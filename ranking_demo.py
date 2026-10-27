#!/usr/bin/env python3
"""Faithful, dependency-free walk-through of SuperHints' retrieval stage.

This implements BM25 and the paper's QE-BM25 score aggregation over a small,
invented product catalog. Hint records follow the paper's four-block schema;
their contents and query judgments are illustrative, not Amazon data. The
separate train_pointwise.py script fine-tunes a small sequence classifier
using the paper's pointwise + hints formulation when a real CSV is supplied.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    description: str
    brand: str

    @property
    def document(self) -> str:
        return f"{self.title} {self.description} {self.brand}"


@dataclass(frozen=True)
class FeatureHint:
    name: str
    synonyms: tuple[str, ...]
    importance: int


@dataclass(frozen=True)
class HintBundle:
    """The paper's reasoning, brands, features, and query-variants blocks."""
    reasoning: str
    brands: tuple[str, ...]
    features: tuple[FeatureHint, ...]
    generated_queries: tuple[str, ...]


CATALOG = (
    Product("A1", "AeroLite 14 ultrabook", "thin lightweight laptop with 16 hour battery, durable aluminum shell", "Aster"),
    Product("A2", "Voyager Pro travel laptop", "portable notebook, long battery life, sturdy magnesium body", "Northstar"),
    Product("A3", "Workhorse 15 business laptop", "durable laptop with long battery, security and repair options", "Northstar"),
    Product("A4", "Creator Studio 16", "powerful workstation with discrete graphics, heavy chassis, color display", "PixelForge"),
    Product("A5", "BudgetBook 14", "affordable everyday notebook, basic battery, plastic case", "ValueTek"),
    Product("A6", "Feather 13 laptop", "lightweight notebook for travel, compact charger and all day battery", "Aster"),
    Product("A7", "Rugged Field Laptop", "reinforced water resistant laptop, tested for drops, long battery", "Terra"),
    Product("A8", "Gaming Max 17", "gaming laptop, high performance graphics, large display, heavy build", "PixelForge"),
    Product("A9", "OfficeBook 14", "standard office notebook with webcam and everyday performance", "ValueTek"),
    Product("A10", "Summit Carbon 13", "premium carbon fiber laptop, lightweight, long battery, travel ready", "Northstar"),
    Product("A11", "Mini Student Laptop", "small inexpensive notebook for school, short battery life", "ValueTek"),
    Product("A12", "Endurance Laptop 14", "laptop with exceptional battery life and durable travel construction", "Aster"),
)

# These two curated records make every demo decision inspectable. Ten variants
# per query mirror the paper's QE-BM25 setup; they are not LLM-generated hints.
HINTS: Mapping[str, HintBundle] = {
    "best lightweight laptop for travel": HintBundle(
        reasoning="Quality in a travel laptop means low carry weight, useful battery life, and a sturdy portable build.",
        brands=("Aster", "Northstar", "Terra"),
        features=(FeatureHint("lightweight", ("light", "low weight", "portable", "easy to carry"), 10),
                  FeatureHint("battery life", ("long battery", "all day battery", "endurance"), 9),
                  FeatureHint("durability", ("sturdy", "rugged", "reinforced", "durable build"), 7)),
        generated_queries=(
            "lightweight laptop for travel long battery", "portable notebook easy to carry all day battery",
            "thin low weight laptop travel ready", "durable ultrabook with long battery life",
            "travel laptop sturdy lightweight construction", "compact notebook long lasting battery",
            "portable laptop reinforced durable build", "light laptop with endurance battery",
            "premium lightweight travel notebook", "easy to carry laptop with all day power",
        ),
    ),
    "best durable laptop for field work": HintBundle(
        reasoning="A field-work laptop should withstand rough handling and keep running away from a desk.",
        brands=("Terra", "Northstar", "Aster"),
        features=(FeatureHint("durability", ("rugged", "reinforced", "drop tested", "sturdy"), 10),
                  FeatureHint("battery life", ("long battery", "endurance", "all day power"), 8),
                  FeatureHint("weather resistance", ("water resistant", "sealed", "outdoor use"), 7)),
        generated_queries=(
            "rugged laptop for field work long battery", "reinforced notebook drop tested durable",
            "water resistant laptop outdoor work", "sturdy portable computer all day battery",
            "field laptop durable construction", "sealed rugged notebook long lasting power",
            "laptop for rough conditions extended battery", "reinforced water resistant computer",
            "durable mobile workstation for outdoor work", "field ready laptop endurance battery",
        ),
    ),
}

# Labels: 2 = relevant and best, 1 = relevant but not best, 0 = irrelevant.
QRELS: Mapping[str, Mapping[str, int]] = {
    "best lightweight laptop for travel": {
        "A1": 1, "A2": 2, "A3": 1, "A4": 0, "A5": 1, "A6": 1,
        "A7": 1, "A8": 0, "A9": 1, "A10": 1, "A11": 1, "A12": 1,
    },
    "best durable laptop for field work": {
        "A1": 1, "A2": 1, "A3": 1, "A4": 0, "A5": 0, "A6": 1,
        "A7": 1, "A8": 0, "A9": 0, "A10": 1, "A11": 0, "A12": 2,
    },
}

TOKEN = re.compile(r"[a-z0-9]+")


class BM25:
    """Small Okapi BM25 index (k1=1.5, b=0.75) over product text."""
    def __init__(self, products: Sequence[Product], k1: float = 1.5, b: float = 0.75):
        self.products = tuple(products)
        self.k1, self.b = k1, b
        self.tokens = [TOKEN.findall(p.document.lower()) for p in products]
        self.lengths = [len(doc) for doc in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.tf = [Counter(doc) for doc in self.tokens]
        df = Counter(token for doc in self.tf for token in doc)
        n = len(products)
        # Robertson/Sparck Jones IDF with a positive floor for tiny corpora.
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        terms = TOKEN.findall(query.lower())
        scores: dict[str, float] = {}
        for product, tf, dl in zip(self.products, self.tf, self.lengths):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    norm = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                    score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / norm
            scores[product.product_id] = score
        return scores

    def rank(self, query: str) -> list[tuple[Product, float]]:
        scores = self.scores(query)
        return sorted(((p, scores[p.product_id]) for p in self.products), key=lambda row: (-row[1], row[0].product_id))


def qe_bm25(index: BM25, hints: HintBundle, max_candidates: int = 50_000) -> list[tuple[Product, float]]:
    """Average per-variant BM25 scores, mirroring the paper's QE-BM25.

    For each of the ten generated queries, append the three suggested brands,
    score the corpus, retain its top max_candidates, then average scores by ID.
    Missing scores contribute zero, like the paper's sparse score matrix.
    """
    if len(hints.generated_queries) != 10:
        raise ValueError("QE-BM25 expects exactly 10 generated query variants")
    total = Counter()
    ids = {p.product_id: p for p in index.products}
    for query in hints.generated_queries:
        scores = index.scores(query + " " + " ".join(hints.brands[:3]))
        top = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:max_candidates]
        for product_id, score in top:
            total[product_id] += score / len(hints.generated_queries)
    return sorted(((p, total[p.product_id]) for p in index.products), key=lambda row: (-row[1], row[0].product_id))


def mrr(rankings: Iterable[Sequence[str]], qrels: Iterable[Mapping[str, int]], k: int | None = None) -> float:
    """MRR for first relevant-and-best (label 2) product per query."""
    reciprocal = []
    for ranking, labels in zip(rankings, qrels):
        limit = ranking if k is None else ranking[:k]
        reciprocal.append(next((1.0 / rank for rank, product_id in enumerate(limit, 1)
                                if labels.get(product_id, 0) == 2), 0.0))
    return sum(reciprocal) / len(reciprocal) if reciprocal else 0.0


def hit_at_k(rankings: Iterable[Sequence[str]], qrels: Iterable[Mapping[str, int]], k: int = 10) -> float:
    """Fraction of queries with a relevant-and-best product in top K."""
    hits = [any(labels.get(product_id, 0) == 2 for product_id in ranking[:k])
            for ranking, labels in zip(rankings, qrels)]
    return sum(hits) / len(hits) if hits else 0.0


def evaluate(index: BM25) -> None:
    queries = list(QRELS)
    labels = [QRELS[q] for q in queries]
    baseline = [[p.product_id for p, _ in index.rank(q)] for q in queries]
    augmented = [[p.product_id for p, _ in qe_bm25(index, HINTS[q])] for q in queries]
    print("Illustrative synthetic data — not paper or Amazon Shopping Queries results")
    print(f"{'Retriever':<18} {'MRR (best)':>12} {'Hit@3':>10} {'Hit@10':>10}")
    for name, rows in (("BM25", baseline), ("QE-BM25 + hints", augmented)):
        print(f"{name:<18} {mrr(rows, labels):>12.3f} {hit_at_k(rows, labels, 3):>10.3f} {hit_at_k(rows, labels, 10):>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", choices=tuple(HINTS), nargs="?", default=next(iter(HINTS)))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true", help="compare BM25 and QE-BM25 on included toy qrels")
    args = parser.parse_args()
    index = BM25(CATALOG)
    if args.evaluate:
        evaluate(index)
        return
    hints = HINTS[args.query]
    print(f"query: {args.query}\n")
    print(f"hint interpretation: {hints.reasoning}")
    print(f"brands: {', '.join(hints.brands[:3])}")
    print("features: " + "; ".join(f"{f.name} (importance {f.importance}) — {', '.join(f.synonyms)}" for f in hints.features))
    for label, results in (("BM25 original query", index.rank(args.query)),
                           ("QE-BM25: averaged 10 hint queries", qe_bm25(index, hints))):
        print(f"\n{label}")
        for rank, (product, score) in enumerate(results[:args.top_k], 1):
            print(f"{rank:>2}. {product.product_id} | {score:.4f} | {product.title} — {product.brand}")


if __name__ == "__main__":
    main()
