# GSM8K RLVR — Clean, Significant GRPO Win

Qwen2.5-1.5B-Instruct, my own GRPO loop, deterministic exact-match reward. This is the
positive-result counterpart to the tau2-bench airline study (`grpo_rl_phase_findings.md`):
same trainer, but a *verifiable* reward — which is exactly what the airline analysis predicted
was the missing ingredient.

## TL;DR
On the full GSM8K test set (1319 problems, greedy decoding, exact-match), 4 iterations of
on-policy GRPO raised **pass@1 from 61.4% (810/1319) to 67.4% (889/1319), +6.0 points,
McNemar exact p < 0.001** (net +79 tasks: 204 improved / 125 worsened; paired bootstrap 95% CI
[+0.033, +0.086], excludes 0). This validates that the GRPO trainer is correct and that the
airline null results were a reward/eval-signal problem, not an algorithm problem.

## Setup
- **Model**: Qwen2.5-1.5B-Instruct (chosen deliberately: 7B is near the GSM8K ceiling, so an RL
  gain would be invisible; 1.5B leaves measurable headroom).
- **Reward**: exact-match on the final integer (`#### <n>`). Deterministic — no LLM judge, no
  user-simulator — so the eval noise that capped the airline project is absent.
- **Algorithm**: REINFORCE-with-group-baseline (GRPO core), group size N=8 per question,
  advantage = (reward − group_mean)/(group_std + eps) with **outcome-variance gating** (all-
  correct / all-wrong groups contribute no gradient), advantage-weighted assistant-token NLL,
  QLoRA. Reused unchanged from the airline trainer.
- **Training**: 4 ON-POLICY iterations — each iteration serves the current policy, samples fresh
  rollouts (128 train questions × 8), then updates the (chained) LoRA. LR 2e-5, temperature 1.0.

## The null → diagnosis → win path (this is the interesting part)
- **First attempt (1 off-policy update, LR 1e-5, ~77 steps): NULL** — 0.60 → 0.60, 17 tasks up /
  17 down. The reward signal was clean (76/128 groups had outcome variance, 608 usable rollouts,
  balanced advantages), so this was a *too-weak update*, not a signal problem.
- **Fix**: switch to multi-iteration on-policy GRPO (re-collect with the improving policy each
  iteration) and raise LR to 2e-5.
- **Result**: training success rose monotonically across iterations (0.66 → 0.70 → 0.73 → 0.73),
  and the held-out test gain became significant on the full set.

## Results
| | pass@1 (greedy, exact-match) |
|---|---|
| base Qwen2.5-1.5B | 61.4% (810/1319) |
| **+ 4-iter on-policy GRPO** | **67.4% (889/1319)** |
| delta | **+6.0 pts**, McNemar p<0.001, 95% CI [+3.3, +8.6] pts |

(On a 200-question subset the same adapter showed +4 pts at p=0.31 — underpowered; the full
1319-question eval was needed to confirm significance, an instance of the same statistical-power
lesson from the airline study.)

## What this proves
1. The GRPO loop (group-relative advantage + outcome-variance gating + QLoRA, single GPU) is
   correct and produces a real, significant policy improvement.
2. The airline nulls were caused by the *reward/eval signal* (LLM user-simulator noise + a
   train/held-out skill-coverage gap + only 50 tasks), not by the optimization — swapping in a
   verifiable reward on a large test set immediately yields a clean win with the same trainer.

## Resume line
"Built a GRPO loop (group-relative advantage with outcome-variance advantage gating); using a
verifiable exact-match reward (RLVR) and 4-iteration on-policy training, raised Qwen2.5-1.5B
GSM8K pass@1 from 61.4% to 67.4% (+6.0, McNemar p<0.001, n=1319)."

## Appendix
- Code: `scripts/grpo/{gsm8k_collect,gsm8k_eval}.py`, `run_gsm8k.sh`, `grpo_update.py`,
  `summarize_eval.py`.
- Artifacts: `autodl_artifacts/gsm8k_r2_westc_20260625/` (base/adapter eval on 1319, per-iter
  rollouts + grpo_metrics, winning adapter_iter4, logs).
