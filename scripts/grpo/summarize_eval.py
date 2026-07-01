"""Summarize a tau2 eval rollouts JSONL into per-task pass rates + a paired McNemar test vs base.

Fixes the R4 MEASUREMENT-BLINDNESS problem: report multi-trial per-task pass^1 with SE, a
LIVE-pool headline (drop tasks no checkpoint can solve -> they only add variance), and paired
fail->succ / succ->fail flips vs base with an exact McNemar p, so a real effect is
distinguishable from the ~+-0.15 noise band of a 20x1 eval.

Stdlib only. Each input line: {"task_id","trial","reward",...}.

Examples:
  python summarize_eval.py --eval outputs/adapter_airline_eval_iter3.jsonl --base outputs/base_airline_eval.jsonl
  python summarize_eval.py --eval outputs/base_airline_eval.jsonl   # base alone
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

# the 10 airline held-out tasks any R4 checkpoint (base or adapter) ever solved -> the only
# tasks with discriminating power. Override with --live-pool once re-measured.
DEFAULT_LIVE = "0,1,5,10,16,26,31,36,40,46"


def load_pass(path: str) -> dict[str, list[float]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        succ = 1.0 if float(d.get("reward", 0.0)) >= 1.0 - 1e-6 else 0.0
        by_task[str(d.get("task_id"))].append(succ)
    return by_task


def rate(by_task: dict[str, list[float]]) -> dict[str, float]:
    return {t: (sum(v) / len(v) if v else 0.0) for t, v in by_task.items()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial, p=0.5). b,c = discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    tail = sum(math.comb(n, k) for k in range(x + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(deltas, n_boot=10000, seed=0):
    """Seeded paired bootstrap 95% CI of the mean per-task delta (resample tasks)."""
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def sign_test(deltas):
    """Exact two-sided sign test on per-task deltas (robust, tie-dropping)."""
    pos = sum(1 for d in deltas if d > 1e-9)
    neg = sum(1 for d in deltas if d < -1e-9)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    x = min(pos, neg)
    p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(x + 1)) * (0.5 ** n))
    return pos, neg, p


def _key(t: str):
    return int(t) if t.lstrip("-").isdigit() else t


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-task pass^1 + paired McNemar vs base for tau2 eval jsonl.")
    ap.add_argument("--eval", required=True, help="candidate/adapter eval jsonl")
    ap.add_argument("--base", default=None, help="base eval jsonl for the paired comparison")
    ap.add_argument("--live-pool", default=DEFAULT_LIVE, help="comma task ids for the discriminating headline pool")
    ap.add_argument("--solve-threshold", type=float, default=0.5, help="task = solved if mean success >= this")
    args = ap.parse_args()

    ev = load_pass(args.eval)
    ev_rate = rate(ev)
    live = [t.strip() for t in args.live_pool.split(",") if t.strip()]
    all_tasks = sorted(ev_rate, key=_key)

    def mean_over(tasks):
        vals = [ev_rate[t] for t in tasks if t in ev_rate]
        return (sum(vals) / len(vals)) if vals else 0.0

    def se_over(tasks):
        vals = [ev_rate[t] for t in tasks if t in ev_rate]
        if not vals:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(max(m * (1 - m), 0.0) / len(vals))

    n_trials = max((len(v) for v in ev.values()), default=0)
    print(f"=== EVAL {args.eval}  (max trials/task = {n_trials}) ===")
    for t in all_tasks:
        print(f"  task {t:>3}: pass^1={ev_rate[t]:.3f}  (n={len(ev[t])})")
    print(f"OVERALL  pass^1 = {mean_over(all_tasks):.3f} +- {se_over(all_tasks):.3f}  over {len(all_tasks)} tasks")
    if live:  # live-pool is airline-specific; skip for GSM8K etc. (pass --live-pool "")
        live_present = [t for t in live if t in ev_rate]
        print(f"LIVE-{len(live)} pass^1 = {mean_over(live):.3f} +- {se_over(live):.3f}  (pool={args.live_pool}; present={len(live_present)}/{len(live)})")

    if args.base:
        base_rate = rate(load_pass(args.base))
        thr = args.solve_threshold
        tasks = sorted(set(ev_rate) | set(base_rate), key=_key)

        def solved(rmap, t):
            return rmap.get(t, 0.0) >= thr
        fwd = [t for t in tasks if solved(ev_rate, t) and not solved(base_rate, t)]   # fail->succ (good)
        bwd = [t for t in tasks if solved(base_rate, t) and not solved(ev_rate, t)]   # succ->fail (regress)
        p = mcnemar_exact(len(bwd), len(fwd))
        base_overall = sum(base_rate.get(t, 0.0) for t in tasks) / len(tasks) if tasks else 0.0
        print(f"\n=== PAIRED vs base {args.base}  (solved = mean >= {thr}) ===")
        print(f"base OVERALL pass^1 = {base_overall:.3f}   cand OVERALL pass^1 = {mean_over(tasks):.3f}")
        print(f"fail->succ (NEW wins): {len(fwd)} {fwd}")
        print(f"succ->fail (regress) : {len(bwd)} {bwd}")
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"net flips = {len(fwd) - len(bwd):+d}   McNemar exact p = {p:.4f}  ({sig})")

        # paired per-task RATE test: controls task difficulty AND averages user-sim noise
        # within each task's trials -> the right test once NUM_TRIALS>1 with a pinned user temp.
        common = sorted(set(ev_rate) & set(base_rate), key=_key)
        deltas = [ev_rate[t] - base_rate[t] for t in common]
        md = sum(deltas) / len(deltas) if deltas else 0.0
        lo, hi = bootstrap_ci(deltas)
        pos, neg, sp = sign_test(deltas)
        print(f"\n=== PAIRED per-task RATE test (n={len(common)} tasks; controls task + user-sim noise) ===")
        print(f"mean per-task delta (cand - base) = {md:+.3f}   bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"tasks improved / worsened = {pos} / {neg}   sign-test p = {sp:.4f}")
        print(f"VERDICT: {'SIGNIFICANT (95% CI excludes 0)' if (lo > 0 or hi < 0) else 'not significant (95% CI spans 0)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
