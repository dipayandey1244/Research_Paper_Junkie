#!/usr/bin/env python3
"""Fine-tune a Qwen2.5 pointwise 3-label reranker in the paper's P or P+H form.

Expected CSV columns: split (train|validation|test), query_id, query,
product_title, product_description, label. P+H also uses hint_reasoning,
hint_features, and top_brands. Labels are exactly: irrelevant,
relevant but not best, relevant and best. This trainer requires prepared
query/product judgments and cached hints; it does not fetch proprietary data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

LABELS = {"irrelevant": 0, "relevant but not best": 1, "relevant and best": 2}


def normalize_label(raw: str) -> int:
    key = " ".join(raw.strip().lower().replace("_", " ").split())
    if key not in LABELS:
        raise ValueError(f"Unsupported label {raw!r}; expected one of {tuple(LABELS)}")
    return LABELS[key]


def format_pair(row: dict[str, str], use_hints: bool) -> str:
    query = row["query"].strip()
    if use_hints:
        reasoning = row.get("hint_reasoning", "").strip()
        features = row.get("hint_features", "").strip()
        brands = [b.strip() for b in row.get("top_brands", "").split("|") if b.strip()][:3]
        query = (f"{query} semantic features: {features} "
                 f"brand context: {', '.join(brands)}\nHint interpretation: {reasoning}")
    return f"{query}\nproduct: {row['product_title'].strip()} {row['product_description'].strip()}"


def load_split(path: Path, split: str, use_hints: bool) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "").strip().lower() != split:
                continue
            required = ("query_id", "query", "product_title", "product_description", "label")
            missing = [key for key in required if not row.get(key, "").strip()]
            if missing:
                raise ValueError(f"{split} row missing fields {missing}: {row}")
            out.append({"query_id": row["query_id"], "text": format_pair(row, use_hints),
                        "labels": normalize_label(row["label"])})
    if not out:
        raise ValueError(f"No rows with split={split!r} found in {path}")
    return out


def classification_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = np.asarray(logits).argmax(axis=-1)
    f1s = []
    for label in range(3):
        tp = np.sum((predicted == label) & (labels == label))
        fp = np.sum((predicted == label) & (labels != label))
        fn = np.sum((predicted != label) & (labels == label))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
    return {"macro_f1": float(np.mean(f1s)), "accuracy": float(np.mean(predicted == labels))}


def ranking_metrics(rows: list[dict], logits: np.ndarray, k: int) -> dict[str, float]:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, scores in zip(rows, logits):
        # P(label=RELEVANT_AND_BEST) is the pointwise ranking score.
        z = np.asarray(scores, dtype=np.float64)
        p_best = np.exp(z[2] - z.max()) / np.exp(z - z.max()).sum()
        groups[row["query_id"]].append((float(p_best), row["labels"]))
    reciprocal, hits = [], []
    for candidates in groups.values():
        ordered = sorted(candidates, key=lambda pair: pair[0], reverse=True)
        ranks = [i for i, (_, label) in enumerate(ordered, 1) if label == 2]
        reciprocal.append(1 / ranks[0] if ranks else 0.0)
        hits.append(float(any(label == 2 for _, label in ordered[:k])))
    return {"mrr": float(np.mean(reciprocal)), f"hit@{k}": float(np.mean(hits)), "queries": len(groups)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="prepared CSV with train/validation/test rows")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="use Qwen2.5-0.5B-Instruct or 3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen-pointwise-hints"))
    parser.add_argument("--without-hints", action="store_true", help="paper's P baseline; default is P+H")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects paper batch size: 32 for 0.5B, 2 for 3B")
    parser.add_argument("--hit-k", type=int, default=10)
    args = parser.parse_args()
    use_hints = not args.without_hints
    train_rows = load_split(args.data, "train", use_hints)
    valid_rows = load_split(args.data, "validation", use_hints)
    test_rows = load_split(args.data, "test", use_hints)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(rows: list[dict]) -> Dataset:
        ds = Dataset.from_list(rows)
        def tokenize(batch):
            return tokenizer(batch["text"], max_length=args.max_length, truncation=True, padding="max_length")
        ds = ds.map(tokenize, batched=True, remove_columns=["text", "query_id"])
        ds.set_format("torch")
        return ds

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id
    batch = args.batch_size or (2 if "3B" in args.model else 32)
    training = TrainingArguments(
        output_dir=str(args.output_dir), learning_rate=1e-5, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch, num_train_epochs=args.epochs,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1", greater_is_better=True,
        save_total_limit=2, logging_steps=50, report_to="none",
        fp16=bool(__import__("torch").cuda.is_available()),
    )
    trainer = Trainer(model=model, args=training, train_dataset=encode(train_rows),
                      eval_dataset=encode(valid_rows), processing_class=tokenizer,
                      compute_metrics=classification_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    # Keep model selection on validation only; report ranking quality once on test.
    predictions = trainer.predict(encode(test_rows))
    metrics = ranking_metrics(test_rows, predictions.predictions, args.hit_k)
    metrics["mode"] = "P+H" if use_hints else "P"
    metrics["split"] = "test"
    (args.output_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
