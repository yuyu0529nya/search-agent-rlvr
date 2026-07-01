# Next GPU session — round 2 (process reward / LATA / KL anchor)

Flagship status: F1 reward already won — best iter5 held-out EM 0.483 (+0.097, p=0.0001),
F1 0.606; collapse deferred to iter6. Data finding: retrieval is NOT the bottleneck
(rHit 0.74→0.85) — the gap is answer EXTRACTION (iter5 retrieves gold 85% but answers 48%).

All code below is built + tested locally (no GPU). Packaged v14. One batched GPU session,
priority order. Goal: spend on compute, not on writing/idle — so write-and-test is minimized.

## Session plan (open one card, run in order)
0. scp v14 `scripts/grpo/`. (parallel-chunk if link slow.)

1. **KL smoke (~5 min, do FIRST — only thing untested locally).** Confirm `disable_adapter`
   + the second (reference) forward works and doesn't OOM, and the KL term shows in the loss:
   ```
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python scripts/grpo/grpo_update.py \
     --rollouts outputs/search_7b_f1/rollouts_iter1.jsonl --base-model /root/autodl-tmp/models/qwen25-7b-instruct \
     --out-adapter /tmp/kl_smoke --reward-mode continuous --gate --batch-size 2 --max-seq-len 2560 \
     --kl-coef 0.05 --progress-every 5
   ```
   Pass = it finishes, GPU mem sane, loss finite. If OOM: the ref forward doubles activation —
   drop --batch-size to 1 for KL runs, or lower --max-seq-len.

2. **KL-anchored F1 run (MAIN, ~50 min) — the headline experiment.** Does the KL anchor stop
   the iter-6 collapse and let us push past 0.483 with MORE iters?
   ```
   RUN=search_7b_f1kl REWARD=f1 KL_COEF=0.05 ITERS=8 BATCH=2 \
     POLICY_MODEL=/root/autodl-tmp/models/qwen25-7b-instruct WORKDIR=/root/autodl-tmp/yuyu \
     PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin HF_ENDPOINT=https://hf-mirror.com \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/grpo/run_search_agent.sh
   then: RUN=search_7b_f1kl ... bash scripts/grpo/eval_adapters.sh   (full EM+F1+rHit curve)
   ```
   Success = no iter-6-style collapse over 8 iters AND best EM >= 0.483 (or clearly flatter tail).

3. **Process-reward run (optional, ~50 min).** Tests the retrieval-hit dense reward. Honest
   expectation: limited EM gain (retrieval already saturated) but may add stability + the rHit
   column will show if it lifts retrieval further.
   ```
   RUN=search_7b_f1proc REWARD=f1 PROC_BETA=0.3 ITERS=6 BATCH=2 ... bash scripts/grpo/run_search_agent.sh
   then eval_adapters.sh
   ```

4. **LATA ablation (optional, cheap toggle).** Add `--lata` is already wired via grpo_update;
   to A/B it cleanly, run one variant with LATA on. (Or fold into a comparison table.)

5. Pull artifacts (parallel chunks), analyze, shut down.

## What each buys (for the report)
- KL: the principled anti-collapse fix → "diagnose collapse → reward fix (F1) → regularization fix (KL)".
- Process reward: ties to the dense-process-reward literature; the rHit/EM gap is itself a finding.
- LATA: ablation rigor.

## Decision rule
KL is the highest-value (anti-collapse). If GPU budget is tight, run ONLY step 1+2 (KL) and
stop. Process reward + LATA are nice-to-haves; the flagship is already a complete win without them.

## Compare against (local, from the F1 run)
base EM 0.387 / F1 0.532 / rHit 0.743 ; best iter5 EM 0.483 / F1 0.606 / rHit 0.853 ;
iter6 collapse EM 0.380 / ans 8.4 chars. Tables: `analyze_search_eval.py`.
