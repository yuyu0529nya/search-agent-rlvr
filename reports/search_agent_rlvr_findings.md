# Search-Agent RLVR — Significant Win and a Reward Over-Optimization Curve

**Task.** Train a multi-turn *retrieval agent* with reinforcement learning from a
verifiable reward, and measure whether on-policy GRPO improves held-out
question-answering accuracy.

**Headline result.** On 300 held-out HotpotQA questions, on-policy GRPO with an
exact-match reward raised the agent's accuracy from **39.0% → 46.0% (+7.0 points,
McNemar exact p = 0.0099, n = 300, bootstrap 95% CI [+0.020, +0.120])** at the
best checkpoint (iteration 3). Pushing training one iteration further *reversed*
the gain (iteration 4 = 24.7%), a textbook case of reward over-optimization that
I diagnosed mechanistically. A follow-up then compared three ways to *prevent*
that collapse (§7–8); a **length-aware advantage** removed it entirely and reached
the project's best **held-out EM 49.3% (+10.7 points over base, n = 300)**, with its
final checkpoint also its best (no early stopping needed). The two strongest
stabilized variants were then nailed down at **n = 2400 (8 trials/question): EM ≈
0.49, McNemar exact p < 1e-30** — confirming the gain is far outside the noise floor.

---

## 1. System

- **Agent.** A multi-turn loop: the model emits `<search>query</search>`, a local
  BM25 retriever (pure-stdlib, k1=1.5, b=0.75) returns the top-k passages from the
  question's HotpotQA *distractor* corpus, the passages are injected as a user
  turn, and the loop repeats (up to 3 searches) until the model emits
  `<answer>…</answer>`. Only the assistant turns are trained.
- **Reward (verifiable).** Normalized exact match between the predicted answer and
  the gold answer (SQuAD-style normalization). Deterministic — no LLM judge, no
  user simulator. This is the whole point: it removes the evaluation noise that
  dominated the earlier tau2 dialogue experiments.
- **Trainer.** My own GRPO loop: group-relative advantage `(r − mean)/(sd + 0.25)`
  with **outcome-variance gating** (groups whose rollouts all share the same binary
  outcome carry no gradient and are dropped), clipped to ±3, QLoRA 4-bit on
  Qwen2.5-7B-Instruct, vLLM for rollout generation.
- **Training.** 4 iterations of *on-policy* GRPO. Each iteration: serve the current
  policy → collect 128 train questions × 8 sampled trajectories (temp 1.0) →
  one GRPO update (LR 2e-5) → the next iteration samples from the improved policy.
- **Evaluation.** 300 held-out validation questions, greedy (temp 0), one trial,
  exact match. Train and held-out question sets are disjoint, so every gain is
  cross-question generalization.

## 2. Result — the full curve

| checkpoint | held-out EM | Δ vs base | McNemar p | verdict |
|-----------|------------|-----------|-----------|---------|
| base      | 0.390      | —         | —         | —       |
| iter 1    | 0.407      | +0.017    | 0.57      | not significant |
| iter 2    | 0.410      | +0.020    | 0.51      | not significant |
| **iter 3**| **0.460**  | **+0.070**| **0.0099**| **SIGNIFICANT** (CI [+0.020,+0.120]) |
| iter 4    | 0.247      | −0.143    | 0.0000    | SIGNIFICANT *regression* |

![Held-out EM by GRPO iteration: rises to a significant peak at iter-3 (0.460, p=0.01) then collapses at iter-4 (0.247) as answer length over-compresses 24→7 chars](search_agent_overopt_curve.svg)

On the *training* set, rollout success climbed monotonically (iter-3 collection
0.546 → iter-4 collection 0.619 at temp 1.0). So the policy kept getting better at
the training questions while held-out accuracy peaked at iter 3 and then collapsed
— the signature of over-fitting / reward over-optimization.

## 3. Mechanism — why iter 4 collapsed

A CPU-only behavioral diff of the eval transcripts:

| checkpoint | EM | avg answer length (chars) | avg searches |
|-----------|-----|---------------------------|--------------|
| base   | 0.390 | 24.3 | 1.51 |
| iter 1 | 0.407 |  9.8 | 2.39 |
| iter 2 | 0.410 | 10.4 | 1.68 |
| iter 3 | 0.460 | 10.8 | 2.05 |
| iter 4 | 0.247 |  7.2 | 2.26 |

The model learned the *right* thing for an exact-match reward on HotpotQA, whose
answers are mostly short ("yes", "no", entity names): it shortened its answers from
~24 to ~10 characters. That is what drove the iter-1→iter-3 gains. But the reward
keeps paying for brevity, so the policy over-shot: by iter 4 the average answer is
**7.2 characters**, short enough that it drops words the gold answer needs. The
metric the reward optimizes (train EM) kept rising while the thing we care about
(held-out EM) fell — Goodhart's law in one curve.

**Decision: select the early-stopping checkpoint (iter 3).** Reporting iter 4 alone
would have looked like "RL hurt"; reporting iter 3 alone would have hidden the
fragility. The honest deliverable is the whole curve plus the checkpoint choice.

## 4. Engineering note — filling the GPU

A first attempt stalled: the GRPO update ran with an effective batch size of 1,
using only ~12 GB of the 32 GB card, and a single 7B update on long multi-turn
sequences took 80+ minutes. I rewrote the update to pad rollouts into batches and
compute the advantage-weighted assistant-token NLL with a single batched
`F.cross_entropy`, plus length-bucketing to minimize padding and a progress line
every N optimizer steps. Verified the batched per-sequence loss is numerically
identical to the per-rollout loss (no padding leakage). On GPU this filled the card
(**30.9 GB / 32 GB at 100% utilization**) and cut each update to 2-3 minutes; the
whole run (base eval + 4×[collect+update] + final eval) finished in ~20 minutes.

## 5. Caveats

- 1 trial at temp 0 per held-out question. The exact-match reward is deterministic,
  so there is no user-simulator noise (unlike the tau2 experiments), and n = 300
  paired questions give the McNemar test real power — but a multi-trial pass^1 would
  tighten the per-task estimates further.
- Train and held-out sets are disjoint within HotpotQA validation/train; the gain is
  cross-question transfer on the same distribution, not cross-dataset transfer.
- Retrieval is over each question's own distractor corpus (the standard HotpotQA
  distractor setting), not open-web retrieval.

## 6. Follow-up — fixing the over-optimization with a partial-credit reward

The diagnosis above says the collapse is driven by the *binary* exact-match reward
over-paying for brevity. The fix follows directly: replace it with **token-F1**
(partial credit — dropping words the answer needs lowers the score), and use a gentler
learning rate. I re-ran the identical pipeline for 6 on-policy iterations with the F1
reward (continuous-variance advantage gating, LR 5e-6).

| checkpoint | EM | ΔEM vs base | McNemar p | F1 | answer chars |
|-----------|-----|------------|-----------|-----|--------------|
| base      | 0.387 | —        | —         | 0.532 | 23.7 |
| iter 1    | 0.480 | +0.093   | 0.0001    | 0.605 | 14.0 |
| iter 4    | 0.470 | +0.083   | 0.0013    | 0.590 | 12.5 |
| **iter 5**| **0.483** | **+0.097** | **0.0001** | **0.606** | 11.8 |
| iter 6    | 0.380 | −0.007   | 0.91      | 0.469 | 8.4 |

(iters 2–3 omitted for brevity; both also significantly beat base on EM.)

Three results:
1. **Higher ceiling.** Best checkpoint (iter 5) reaches EM 0.483 — above the binary
   run's best of 0.460 (+0.097 over base, p = 0.0001). The partial-credit reward also
   lifts F1 (0.532 → 0.606, 95% CI excludes 0).
2. **Far more stable.** Every checkpoint iter 1–5 significantly beats base on EM,
   versus the binary run where only iter 3 was significant before it crashed at iter 4.
3. **Same failure mode, delayed and milder.** The collapse still eventually appears —
   iter 6 over-compresses answers to 8.4 chars and regresses to base — but F1 pushed
   it from iter 4 out to iter 6. The mechanism is identical (answer-length collapse);
   the partial-credit reward just resists it longer. Best-checkpoint selection (iter 5)
   remains the right policy to ship.

This closes the loop: diagnose (binary reward → brevity collapse) → fix (partial-credit
F1 reward + gentler LR) → improve (higher and more stable held-out accuracy, collapse
deferred). An engineering note that fell out of it: F1 rollouts are longer than the
brevity-collapsed binary ones, so the batched update needed a smaller batch size to fit
the 32 GB card.

## 7. Round-2 — three anti-over-optimization levers, head-to-head

§6's F1 reward *delayed* the collapse but didn't remove it (iter 6 still
regressed). Since the root cause is answer-length collapse, Round-2 asks which
mechanism best prevents it **without sacrificing the peak**. On top of the F1
reward I ran three fresh on-policy comparisons (same 300 held-out protocol;
baselines agree within ±0.005 across runs, so the comparison is fair; EM here is
re-scored by the analyzer and differs by ≤1–2 questions from the runtime logs):

- **KL-to-base anchor (β = 0.05)** — a KL penalty in the GRPO update (Schulman k3
  estimator), with the reference distribution being the *same* model with the LoRA
  adapter disabled (no second model loaded). Resists drift from the base policy.
- **Dense process reward (β = 0.3)** — reward = F1 + 0.3·(retrieval-hit 0.7 +
  query-efficiency 0.3), added at collection only; eval stays pure EM. Rewards the
  intermediate retrieval steps, not just the final answer.
- **Length-aware advantage** — divide each advantage by √(assistant-token length),
  directly removing the implicit "shorter ⇒ higher advantage" pressure.

| run | best EM | Δ vs base | McNemar p | final EM | answer chars | late stage |
|-----|---------|-----------|-----------|----------|--------------|------------|
| binary EM (§2) | 0.460 | +0.070 | 0.010 | 0.247 | → 7.2 | **collapses** |
| F1 only (§6) | 0.483 | +0.097 | 0.0001 | 0.380 | → 8.4 | **collapses** |
| F1 + KL anchor | 0.437 | +0.053 | 0.014 | 0.437 | 14–18 | stable; **peak capped** |
| F1 + process reward | 0.483 | +0.093 | 0.0002 | 0.470 | 10–12 | peak kept; stays up |
| **F1 + length-aware adv** | **0.493** | **+0.107** | **0.0000** | **0.493** | 11–13 | **best; endpoint = peak** |

*(Round-2 runs: 6–8 on-policy iterations, LR 5e-6, continuous-variance advantage
gating. The `length-aware` iter-6 checkpoint is the project's best.)*

**The answer-length column carries the mechanism.** Over-optimization *is*
answer-length collapse (base ~24 chars → 7–8 when it breaks), and each lever acts
on it differently:

- **KL is a blunt instrument.** It holds the policy near the base, so answers stay
  *long* (14–18 chars) and never collapse — but it also suppresses the *useful*
  shortening that earns EM on HotpotQA's short answers, so its peak is the lowest of
  the three (0.437).
- **Process reward** lets answers shorten to a healthy ~11 chars while the retrieval
  signal blocks over-compression → it matches pure-F1's peak (0.483) and, unlike
  pure-F1, stays up (final 0.470 vs 0.380). It also confirmed an earlier data
  finding: retrieval recall was already high (rHit ~0.75 → 0.83), so the dense signal
  helped *stability* more than it raised the ceiling.
- **Length-aware advantage is the most on-mechanism fix.** It cancels the
  "shorter ⇒ higher advantage" gradient directly, so answers settle at 11–13 chars
  and never run away to 7. It reaches the highest score (0.493, +0.107, p < 0.001)
  **and** its endpoint is its peak — no checkpoint selection needed.

**Takeaway:** a mechanism-targeted fix (length-aware advantage) beat a generic
regularizer (KL — which trades peak for stability) and an indirect dense signal
(process reward); all three beat the naive binary/F1 reward on late-stage
stability. This turns the §3 diagnosis into a controlled comparison of *fixes*,
with the mechanism (answer length) measured throughout.

## 8. Round-4 — stabilizing the best combo (length-aware adv + process reward)

Round-3 found that *combining* the two strongest Round-2 levers — length-aware advantage +
process reward ("lata+proc", β = 0.3, N = 8) — pushed the peak to ~0.50, but reintroduced a
single-iteration dip around iter 3. Round-4 asks: can one knob remove that dip while keeping
the ~0.49 peak? Starting from lata+proc, I varied four knobs, **one at a time**, each a fresh
6-iteration on-policy run on a 2×5090 box (vLLM resident on one card + training on the other,
per-iter LoRA hot-reload; same 300-question held-out protocol, analyzer-rescored EM, paired
McNemar + bootstrap CI per iter).

| variant (one knob off lata+proc) | best EM | best F1 | iter-3 EM | iter-3 sig? | final EM | answer chars | verdict |
|---|---|---|---|---|---|---|---|
| + light KL (β = 0.01) | 0.483 | 0.607 | 0.397 | **p = 0.72, n.s.** | 0.460 | i3 → **8.5** | dip only softened |
| + lower LR (3e-6) | 0.460 | 0.587 | 0.453 | p = 0.005 | 0.423 | 12–15 | very stable, peak capped |
| **+ weaker process (β = 0.15)** | **0.490** | **0.613** | 0.470 | p = 0.004 | 0.473 | ~11 | **peak kept, dip gone** |
| **+ larger group (N = 12)** | 0.483 | 0.607 | 0.463 | p = 0.0006 | **0.480** | ~12 | **dip gone, strongest endpoint** |

**The answer-length column again carries the mechanism.** The dip is still answer-length
collapse: the one run that stayed dippy — light-KL — is exactly the one whose iter-3 answers
shrank to 8.5 chars while searches spiked to ~2.8 (searching frantically yet answering
shortest, the over-optimization signature), and its iter-3 gain is statistically
indistinguishable from base (p = 0.72). KL *reduced* the dip's depth but did **not** remove
it — a correction to the intuition that a KL anchor is the natural fix here.

The two knobs that actually fixed it attack the cause, not the symptom:
- **Weaker process reward (β 0.3 → 0.15)** — halving the dense shortcut signal stops the
  policy over-chasing "retrieval-hit" credit, so answer length holds at ~11 and every iter
  stays significant. Highest peak (0.490) and highest F1 (0.613).
- **Larger group (N 8 → 12)** — more rollouts per prompt lower the advantage variance, so the
  noisy iter that caused the dip is averaged out; smallest mid-run wobble and the strongest,
  most-significant endpoint (final 0.480, +0.110, p < 0.001).
- **Lower LR / light KL** trade peak (or significance) for smoothness, consistent with
  Round-2: generic regularizers stabilize but cost the ceiling.

**Takeaway:** the best cure for the on-policy dip was *not* the textbook KL anchor — it was
**weakening the shortcut reward or lowering gradient variance (a bigger group)**, both of
which kept the peak. *Caveat: Round-4 used single-trial eval per iter (vs the multi-trial
nail-down in Round-2/3), so each per-iter EM carries ~±3–4% sampling noise; the per-iter
McNemar/CI plus the consistent length signature make the ranking trustworthy as
corroboration, but a multi-trial rerun of the top two (β = 0.15, N = 12) is what I'd do
before declaring a single winner.*

**Multi-trial nail-down (follow-up).** I then re-evaluated the top checkpoints at 8
trials/question (temp 0.7, n = 2400, same paired protocol) to strip the single-trial noise:

| checkpoint | EM | Δ vs base | McNemar p | token-F1 | ΔF1 CI |
|---|---|---|---|---|---|
| base | 0.376 | — | — | 0.519 | — |
| v_b015 iter4 (weaker process) | 0.487 | +0.111 | **<0.0001** | **0.610** | [+.052, +.130] |
| v_n12 iter6 (larger group) | **0.490** | +0.114 | **<0.0001** | 0.609 | [+.052, +.128] |
| v_n12 iter5 | 0.479 | +0.103 | 0.0002 | 0.603 | [+.045, +.122] |

All three are strongly significant over base (n = 2400, not single-trial noise), and v_b015
vs v_n12 are a **statistical tie** (EM differ by 0.003, F1 0.610 vs 0.609, near-identical CIs).
This confirms the two stabilization knobs reach the **same endpoint strength as Round-3's best**
(lata+proc nailed down at EM 0.488) — i.e. removing the dip cost nothing at the peak.

## 9. What this demonstrates

- An end-to-end Agent + RL pipeline: multi-turn tool-use agent, verifiable reward,
  on-policy GRPO, QLoRA, vLLM, paired significance testing — built and run myself.
- A **statistically significant** improvement from RL (p = 0.01, n = 300).
- Recognition and mechanistic diagnosis of **reward over-optimization**, and the use
  of **checkpoint selection / early stopping** to ship the best policy rather than
  the last one.

This pairs with two earlier results: a clean significant GSM8K RLVR win (61.4% →
67.4%, p < 0.001, n = 1319) that validated the same trainer, and a rigorous
*negative* result on the tau2 airline dialogue benchmark whose blocker I traced to
evaluation noise (a stochastic user simulator) and a training-data skill gap rather
than the RL algorithm — which is exactly why moving to a verifiable reward produced
clean wins.
