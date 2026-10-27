#!/usr/bin/env python3
"""Fine-tune a Qwen2.5 pointwise 3-label reranker in the paper's P or P+H form.

Expected CSV columns: split (train|validation|test), query_id, product_id,
query, product_title, product_description, label. For P+H's paper-best query
strategy, add hint_query (one reformulation per row) and top_brands. To run the
paper's feature-ablation strategy, supply hint_features and top_brands, then
pass --hint-strategy features. Labels are exactly: irrelevant, relevant but
not best, relevant and best. The CSV repeats query/product pairs for each
reformulation; test scores are averaged by product before ranking.
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


def format_pair(row: dict[str, str], use_hints: bool, hint_strategy: str) -> str:
    query = row["query"].strip()
    if use_hints:
        brands = [b.strip() for b in row.get("top_brands", "").split("|") if b.strip()][:3]
        if len(brands) < 3:
            raise ValueError("P+H requires three suggested brands in top_brands")
        if hint_strategy == "queries":
            query = row.get("hint_query", "").strip()
            if not query:
                raise ValueError("P+H --hint-strategy queries requires hint_query in every row")
        else:
            features = row.get("hint_features", "").strip()
            if not features:
                raise ValueError("P+H --hint-strategy features requires hint_features in every row")
            query = f"{query} semantic features: {features}"
        query = f"{query}\nbrand context: {', '.join(brands)}"
    return f"{query}\nproduct: {row['product_title'].strip()} {row['product_description'].strip()}"


def load_split(path: Path, split: str, use_hints: bool, hint_strategy: str) -> list[dict]:
    out = []
    baseline_seen = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "").strip().lower() != split:
                continue
            required = ("query_id", "product_id", "query", "product_title", "product_description", "label")
            missing = [key for key in required if not row.get(key, "").strip()]
            if missing:
                raise ValueError(f"{split} row missing fields {missing}: {row}")
            example = {"query_id": row["query_id"], "product_id": row["product_id"],
                       "text": format_pair(row, use_hints, hint_strategy),
                       "labels": normalize_label(row["label"])}
            if not use_hints:
                # P has one example per query/product. If the input CSV also
                # contains reformulation rows for P+H, collapse those variants
                # so the baseline is neither trained nor evaluated repeatedly.
                key = (example["query_id"], example["product_id"])
                prior = baseline_seen.get(key)
                if prior:
                    if prior["labels"] != example["labels"] or prior["text"] != example["text"]:
                        raise ValueError(f"Conflicting baseline rows for query/product {key}")
                    continue
                baseline_seen[key] = example
            out.append(example)
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
    # QE/P+H can create several rows per original query/product (one per
    # generated reformulation). Average its best-label probability first.
    groups: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    for row, scores in zip(rows, logits):
        # P(label=RELEVANT_AND_BEST) is the pointwise ranking score.
        z = np.asarray(scores, dtype=np.float64)
        p_best = np.exp(z[2] - z.max()) / np.exp(z - z.max()).sum()
        groups[row["query_id"]][row["product_id"]].append((float(p_best), row["labels"]))
    reciprocal, hits = [], []
    for product_scores in groups.values():
        candidates = []
        for variants in product_scores.values():
            labels = {label for _, label in variants}
            if len(labels) != 1:
                raise ValueError("A product must keep one gold label across query reformulations")
            candidates.append((float(np.mean([score for score, _ in variants])), variants[0][1]))
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
    parser.add_argument("--hint-strategy", choices=("queries", "features"), default="queries",
                        help="P+H uses reformulated queries by default; features reproduces the paper's ablation")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects paper batch size: 32 for 0.5B, 2 for 3B")
    parser.add_argument("--hit-k", type=int, default=10)
    args = parser.parse_args()
    use_hints = not args.without_hints
    train_rows = load_split(args.data, "train", use_hints, args.hint_strategy)
    valid_rows = load_split(args.data, "validation", use_hints, args.hint_strategy)
    test_rows = load_split(args.data, "test", use_hints, args.hint_strategy)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(rows: list[dict]) -> Dataset:
        ds = Dataset.from_list(rows)
        def tokenize(batch):
            return tokenizer(batch["text"], max_length=args.max_length, truncation=True, padding="max_length")
        ds = ds.map(tokenize, batched=True, remove_columns=["text", "query_id", "product_id"])
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
    metrics["mode"] = (f"P+H ({args.hint_strategy})" if use_hints else "P")
    metrics["split"] = "test"
    (args.output_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
