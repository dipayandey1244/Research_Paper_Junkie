# Hint-Augmented Re-ranking · Research Work

Interactive research walkthrough and dependency-free Python teaching implementation inspired by Yilun Zhu, Nikhita Vedula, and Shervin Malmasi, “Hint-Augmented Re-ranking: Efficient Product Search using LLM-Based Query Decomposition” (AACL 2025).

- `index.html`: single-file article and interactive query sandbox
- `ranking_demo.py`: runnable Python implementation
- `favicon.svg`: article favicon

```bash
python3 ranking_demo.py
python3 ranking_demo.py --evaluate
python3 ranking_demo.py "most lightweight trail shoes" --top-k 3
python3 -m http.server 8080
```

The demos use invented products, four toy queries, and hand-authored labels. Rule-based hints stand in for model-based query decomposition; the scoring function stands in for a trained lightweight ranker. This is not a reproduction of the paper's experiments or private product data. The paper reports +10.9 MAP points and +5.9 MRR points over baselines on its evaluation.

Sources: [Amazon Science](https://www.amazon.science/publications/hint-augmented-re-ranking-efficient-product-search-using-llm-based-query-decomposition) · [ACL Anthology](https://aclanthology.org/2025.ijcnlp-short.19/).
