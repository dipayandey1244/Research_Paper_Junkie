#!/usr/bin/env python3
"""Deterministic, dependency-free teaching demo of hint-augmented reranking.

The tiny hand-authored data illustrates the pipeline; it does not reproduce
the paper's reported metrics or proprietary experiments.
"""
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Product:
    id: str
    title: str
    features: tuple[str, ...]


CATALOG = (
    Product("p1", "Trailblazer waterproof hiking boots", ("waterproof", "durable", "hiking", "grip")),
    Product("p2", "Summit lightweight trail runners", ("lightweight", "hiking", "breathable", "grip")),
    Product("p3", "Everyday canvas walking shoes", ("casual", "lightweight", "walking")),
    Product("p4", "Alpine insulated winter boots", ("warm", "waterproof", "winter", "durable")),
    Product("p5", "RidgePro hiking boots with deep traction", ("durable", "hiking", "grip", "waterproof", "quality")),
    Product("p6", "FeatherLite running shoes", ("lightweight", "running", "breathable")),
)
CASES = (
    ("best waterproof hiking boots", ("quality", "waterproof", "durable", "hiking"), "p5"),
    ("most lightweight trail shoes", ("lightweight", "hiking", "grip"), "p2"),
    ("best warm winter boots", ("warm", "waterproof", "winter"), "p4"),
    ("most durable waterproof boots", ("durable", "waterproof", "grip"), "p5"),
)
STOP = {"the", "a", "an", "for", "with", "best", "most", "top", "popular", "rated", "highly"}
HINT_MAP = {"best": ("quality",), "top": ("quality",), "most": ("degree",),
            "popular": ("popular",), "trail": ("hiking",)}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def extract_hints(query: str) -> tuple[str, ...]:
    """Rule-based stand-in for a model that decomposes the query into hints."""
    out: list[str] = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        out.extend(HINT_MAP.get(token, (token,)))
    return tuple(dict.fromkeys(out))


def retrieve(query: str, products: Sequence[Product] = CATALOG) -> list[Product]:
    """Candidate retrieval by lexical overlap, with deterministic tie-breaking."""
    q = tokens(query)
    return sorted(products, key=lambda p: (-len(q & tokens(p.title + " " + " ".join(p.features))), p.id))


def rerank(query: str, hints: Sequence[str], candidates: Sequence[Product]) -> list[Product]:
    """Lightweight second stage that combines hint matches and lexical overlap."""
    q = tokens(query)
    def score(p: Product) -> tuple[int, int, str]:
        lexical = len(q & tokens(p.title + " " + " ".join(p.features)))
        return sum(h in p.features for h in hints), lexical, p.id
    return sorted(candidates, key=score, reverse=True)


def mrr(rankings: Iterable[Sequence[str]], relevant: Iterable[str]) -> float:
    """Mean reciprocal rank of the first relevant item; zero if absent."""
    rr = [next((1.0 / i for i, item in enumerate(ranked, 1) if item == answer), 0.0)
          for ranked, answer in zip(rankings, relevant)]
    return sum(rr) / len(rr) if rr else 0.0


def hit_at_k(rankings: Iterable[Sequence[str]], relevant: Iterable[str], k: int = 3) -> float:
    """Fraction of queries with a relevant result in the top k."""
    hits = [answer in ranked[:k] for ranked, answer in zip(rankings, relevant)]
    return sum(hits) / len(hits) if hits else 0.0


def evaluate() -> None:
    relevant = [case[2] for case in CASES]
    baseline = [retrieve(query) for query, _, _ in CASES]
    hinted = [rerank(query, hints, candidates)
              for (query, hints, _), candidates in zip(CASES, baseline)]
    print(f"{'System':<24} {'MRR':>8} {'Hit@3':>8}")
    print("-" * 42)
    for label, rows in (("Lexical retrieval", baseline), ("Hint reranker", hinted)):
        ids = [[p.id for p in row] for row in rows]
        print(f"{label:<24} {mrr(ids, relevant):>8.3f} {hit_at_k(ids, relevant):>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="best waterproof hiking boots")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true", help="score the included toy query set")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
        return
    candidates = retrieve(args.query)
    hints = extract_hints(args.query)
    ranked = rerank(args.query, hints, candidates)
    print(f"query: {args.query}\nhints: {', '.join(hints)}\n")
    for i, product in enumerate(ranked[:max(0, args.top_k)], 1):
        print(f"{i}. [{product.id}] {product.title} — matched: {', '.join(product.features)}")


if __name__ == "__main__":
    main()
