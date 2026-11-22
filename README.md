# Research Work

The [homepage](https://dipayandey1244.github.io/Research_Paper_Junkie/) links to the Hint-Augmented Re-ranking walkthrough and the [Sleep Framework walkthrough](https://dipayandey1244.github.io/Research_Paper_Junkie/sleep-framework.html), which includes a runnable continual-learning experiment. The [Reinforcement Learning walkthrough](https://dipayandey1244.github.io/Research_Paper_Junkie/reinforcement-learning.html) adds browser-based bandit and Gridworld experiments.

## Hint-Augmented Re-ranking

Interactive walkthrough and implementation of the key methods in Zhu, Vedula, and Malmasi (AACL 2025), [“Hint-Augmented Re-ranking: Efficient Product Search using LLM-Based Query Decomposition”](https://aclanthology.org/2025.ijcnlp-short.19/).

## Files

- `index.html` — homepage for the research walkthroughs
- `hint-augmented-reranking.html` — article, paper results, and interactive BM25/QE-BM25 explorer
- `sleep-framework.html` — Google Research Sleep Framework walkthrough and experiments
- `reinforcement-learning.html` — interactive tutorial on bandits, Q-learning, and DQN
- `rl-lab.js` — simulator, Gridworld, and neural training code
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

Prepare a CSV with columns `split,query_id,product_id,query,product_title,product_description,label`; `split` is `train`, `validation`, or `test`, and `label` is `irrelevant`, `relevant but not best`, or `relevant and best`. For paper-best P+H, add `hint_query` (one reformulation per product row) and `top_brands` (brands separated by `|`). Repeat products across reformulations under the same `query_id`; held-out evaluation averages their scores by product. The feature-text ablation uses `hint_features` and `top_brands` instead. For the P baseline, duplicate reformulation rows are collapsed to one row per query/product. The script uses Qwen2.5 sequence classification, paper learning rate and per-device batch-size defaults, validation early stopping, and query-grouped held-out test MRR/Hit@K.

```bash
pip install torch transformers datasets accelerate numpy
python3 train_pointwise.py --data prepared_pairs.csv --model Qwen/Qwen2.5-0.5B-Instruct --hint-strategy queries
python3 train_pointwise.py --data prepared_pairs.csv --model Qwen/Qwen2.5-0.5B-Instruct --without-hints
```

The paper derives its evaluation set from Amazon Shopping Queries, generating 21,407 superlative queries and 1.07M query/product labels. Those derived labels and cached hints are not included here. The authors' linked code repository currently says the release is forthcoming; this project does not claim a full benchmark reproduction.

## Paper findings

The paper reports BM25 → QE-BM25 MAP 24.79 → 35.74 (+10.95 points) and Qwen2.5-3B pointwise → pointwise+hint MRR 68.82 → 74.74 (+5.92 points). Its main table and additional caveats are in the article.

## Reinforcement learning

[Open the interactive walkthrough](https://dipayandey1244.github.io/Research_Paper_Junkie/reinforcement-learning.html). To run locally, start a static server from the repository root:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/reinforcement-learning.html`. The experiments run locally in the browser.

### Exploration, exploitation, and multi-armed bandits

- **Greedy** exploits the highest estimated reward. **ε-greedy** chooses uniformly at random during an ε fraction of rounds and follows the highest estimate on other rounds.
- **UCB** tries each arm once, then ranks arms by their sample means plus an uncertainty bonus, `c * sqrt(log(t) / count)`.
- **Thompson Sampling** draws once per arm from its Beta posterior and chooses the largest draw. A Beta(1,1) prior becomes Beta(1 + successes, 1 + failures).
- The seeded experiment uses four stationary Bernoulli arms. Compare observed reward, pulls by arm, and cumulative pseudo-regret under several seeds before drawing conclusions.

### Q-learning

The 4×4 grid has a goal (+1), two traps (−1), and a step cost of 0.04. Use **Advance one step** to inspect the reward, TD target, and old and new Q-values. The tabular agent uses ε=0.25, α=0.35, and γ=0.95. It bootstraps from the greedy next-state value on continuing transitions. At a terminal transition the target equals the reward.

### Deep Q-Network

A one-hot state passes through two ReLU layers (16 and 12 units) and a four-output linear head. The online network trains on batches of 16 from a 512-transition replay buffer, using Huber loss and SGD at 0.065. Training starts after 24 experiences. A hard target update copies weights every 40 optimizer steps. Exploration follows `max(0.15, 0.98 ** episode)`. Half of episodes start in the top-left. The others start in a randomly chosen nonterminal cell. Both Gridworld labs evaluate greedily from the same top-left cell. A training episode may be truncated after 35 moves. The artificial time limit leaves the final nonterminal transition eligible for bootstrapping.

The default seed reaches the goal in six greedy moves after 75 tabular episodes or 100 DQN episodes. These are small teaching runs on a deterministic grid. Treat the route and recent batch loss as diagnostics for this setup. Broader comparisons need several seeds and matched training conditions.

### References

- Sutton and Barto, [Reinforcement Learning: An Introduction](https://mitpress.ublish.com/book/reinforcement-learning-an-introduction-2), Chapters 2 and 6.
- Russo et al., [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).
- Watkins and Dayan, [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html).
- Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).
