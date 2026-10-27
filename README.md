# Hint-Augmented Re-ranking · Research Work

Interactive walkthrough and implementation of the key methods in Zhu, Vedula, and Malmasi (AACL 2025), [“Hint-Augmented Re-ranking: Efficient Product Search using LLM-Based Query Decomposition”](https://aclanthology.org/2025.ijcnlp-short.19/).

## Files

- `index.html` — single-file article, paper results, and interactive BM25/QE-BM25 explorer
- `ranking_demo.py` — dependency-free Okapi BM25 and ten-variant QE-BM25 implementation with MRR and Hit@K
- `train_pointwise.py` — Hugging Face/PyTorch fine-tuning path for the paper's pointwise Qwen P and P+H rerankers
- `favicon.svg` — article favicon

## Run the retrieval demo

```bash
python3 ranking_demo.py
python3 ranking_demo.py --evaluate
python3 ranking_demo.py "best durable laptop for field work" --top-k 5
```

The local catalog contains 12 invented product records, two hand-authored four-block hint bundles, and hand-authored three-class relevance judgments. It illustrates BM25 and QE-BM25 score aggregation; its metrics are not paper results.

## Train a pointwise reranker

Prepare a CSV with columns `split,query_id,product_id,query,product_title,product_description,label`; `split` is `train`, `validation`, or `test`, and `label` is `irrelevant`, `relevant but not best`, or `relevant and best`. For paper-best P+H, add `hint_query` (one reformulation per product row) and `top_brands` (brands separated by `|`). Repeat products across reformulations under the same `query_id`; held-out evaluation averages their scores by product. The feature-text ablation uses `hint_features` and `top_brands` instead. The script uses Qwen2.5 sequence classification, paper learning rate and per-device batch-size defaults, validation early stopping, and query-grouped held-out test MRR/Hit@K.

```bash
pip install torch transformers datasets accelerate numpy
python3 train_pointwise.py --data prepared_pairs.csv --model Qwen/Qwen2.5-0.5B-Instruct --hint-strategy queries
python3 train_pointwise.py --data prepared_pairs.csv --model Qwen/Qwen2.5-0.5B-Instruct --without-hints
```

The paper derives its evaluation set from Amazon Shopping Queries, generating 21,407 superlative queries and 1.07M query/product labels. Those derived labels and cached hints are not included here. The authors' linked code repository currently says the release is forthcoming; this project does not claim a full benchmark reproduction.

## Paper findings

The paper reports BM25 → QE-BM25 MAP 24.79 → 35.74 (+10.95 points) and Qwen2.5-3B pointwise → pointwise+hint MRR 68.82 → 74.74 (+5.92 points). Its main table and additional caveats are in the article.
