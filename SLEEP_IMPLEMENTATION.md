# Sleep Framework — runnable LLM experiment

**Status:** engineering approximation, not official Google implementation or faithful reproduction of its MoE/GKD/LTI/dreaming algorithms.

## Quick start
Python 3.10+ recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r sleep_requirements.txt
python sleep_train.py --method sequential --model sshleifer/tiny-gpt2
python sleep_train.py --method replay --model sshleifer/tiny-gpt2
python sleep_train.py --method isolated --model sshleifer/tiny-gpt2
```

For a more capable (but more expensive) model, try `--model distilgpt2`. This model is not instruction-tuned; tiny-gpt2 is primarily a pipeline smoke test and may score zero even when training runs.

## What runs
- A real Hugging Face causal LM, trained via PEFT LoRA on three sequential groups of fictional QA facts.
- **Sequential:** one LoRA adapter is repeatedly trained on each task, no replay.
- **Replay:** one LoRA adapter is trained on the current task plus all previous facts (unbounded rehearsal baseline).
- **Isolated:** frozen backbone and one LoRA adapter per task, selected with *oracle task identity* at evaluation. This does not solve routing.
- Per-stage, per-task exact-match generation accuracy written to `sleep_runs/<method>/metrics.json`.

## Important limitations
- Accuracy is on training questions, not held-out paraphrases; do not present it as generalization.
- The tiny dataset is deliberately fictional and is useful for testing mechanics, not benchmarking continual learning.
- No GKD teacher-student KL, on-policy sampling, LTI reinforcement learning, MoE router, gradient-filtered dreaming, or temporary memory pruning is implemented.
- Isolated adapter evaluation assumes the task ID is already known, an unfair advantage compared with deployed routing.
- Replay consumes more examples as tasks accumulate. Report compute and memory budgets before interpreting comparisons.
- Results depend on model, optimizer, and seed. The scripts have not been executed against downloaded Hugging Face model weights in this environment.

## Research-grade next milestones
1. Replace synthetic training-only questions with held-out paraphrases and a separate pretraining-retention benchmark.
2. Implement teacher-with-context -> student-without-context distillation. Compute teacher and student logits on **student-sampled** prefixes, with token-level KL and temperature scaling. Include teacher-forced KL as ablation.
3. Implement MoE-style adapter routing with learned router and expert utilization logs, rather than oracle task ID.
4. Add candidate dream generation, verified factual filtering, gradient-based selection, and rehearsal.
5. Compare matched FLOPs, token budgets, expert capacity, accuracy matrices, forgetting, backward/forward transfer, and confidence intervals across seeds.

## Citation
Ali Behrouz and Vahab Mirrokni, [Google Research blog: The sleep framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/), October 2026.
