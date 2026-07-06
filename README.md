# Search-Agent RLVR — verifiable-reward RL for multi-turn retrieval agents

A compact RLVR research system for training a multi-turn **retrieval agent**:
`<search>` → BM25 over a HotpotQA corpus → `<answer>`, optimized with verifiable exact-match and
token-F1 rewards. The repository contains the GRPO trainer, rollout/evaluation harness, reward
design, paired statistics, and the mechanism analysis behind the final gains.

> This is the **controlled-environment RL-science leg** of a two-repo project: the deterministic
> reward isolates measurement noise, so reward-optimization dynamics can be studied
> mechanistically. The hard-environment headline — a noisy multi-turn tool-calling agent improved
> from pass^1 **0.20 → 0.41 → 0.55** via teacher-distillation warm-start + GRPO — lives in the companion repo
> [**tau2-agentic-rl**](https://github.com/yuyu0529nya/tau2-agentic-rl).

## Headline
- **Held-out Exact-Match 38.7% → 49.3% (+10.7 points)** — McNemar **p < 0.001** (n = 300),
  reconfirmed with multi-trial evaluation at **n = 2400, p < 1e-30**.
- **A statistically decisive RL gain** on a multi-turn tool-use QA loop, re-verifiable from the
  raw rollout/eval artifacts.
- On GSM8K (same self-built loop, exact-match reward): **61.4% → 67.4% pass@1**
  (+6.0 pts, McNemar p < 0.001, n = 1319).

## Why it matters
1. **Clean RL signal.** The reward is deterministic and verifiable, so the training curve is not
   hidden behind simulator or judge noise.
2. **Mechanistic optimization.** The project identifies answer-length pressure as the key
   reward-optimization mechanism, then stabilizes it with token-F1 reward shaping and length-aware
   advantage normalization.
3. **Controlled ablations.** KL-to-base, dense process reward, and length-aware advantage are
   compared head-to-head; the mechanism-targeted length-aware fix produces the strongest and most
   stable endpoint.

## The over-optimization curve
![Search-agent RLVR: held-out EM by GRPO iteration](reports/search_agent_overopt_curve.svg)

*The binary-reward run: held-out Exact-Match climbs from base 0.390 to 0.460 (p = 0.01),
then exposes a clear reward-optimization pressure as answers shrink to ~7 chars.
Token-F1 + length-aware advantage keeps the gain while stabilizing the behavior.*
Detailed round-by-round results (binary → F1 → KL / process-reward / LATA ablation, plus the
multi-trial nail-down) are in [`reports/search_agent_rlvr_findings.md`](reports/search_agent_rlvr_findings.md).

## Trainer components (`scripts/grpo/`)
- **`grpo_update.py`** — GRPO from scratch: group-relative advantage `(r − mean)/(std + ε)`,
  **outcome-variance advantage gating** (drop no-contrast groups), **length-aware advantage**,
  QLoRA, batched loss, optional KL-to-base anchor.
- **`search_agent.py` / `search_retriever.py` / `qa_reward.py`** — the retrieval-agent
  episode loop, a pure-stdlib BM25 retriever over HotpotQA, and the verifiable QA reward
  (normalized EM + token-F1).
- **`run_search_agent.sh`** — end-to-end: serve → base eval → N-iter on-policy
  collect+update → eval → analyze.
- **`gsm8k_collect.py` / `gsm8k_eval.py` / `run_gsm8k.sh`** — the GSM8K RLVR pipeline.
- **`analyze_search_eval.py` / `summarize_eval.py` / `recheck_search_em.py`** — paired
  evaluation with bootstrap CIs + McNemar, behavioral (answer-length) diffs, and an
  independent re-check of EM straight from the raw rollouts.

## Tech
Qwen2.5 (1.5B / 7B) · vLLM · PEFT / QLoRA · bitsandbytes · transformers · HotpotQA ·
on-policy GRPO with verifiable rewards (RLVR).

*Trained adapters, model weights, and rollout/eval artifacts are kept out of the repo for
size — this repository is the code and the findings write-ups (`reports/`).*
