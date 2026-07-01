#!/usr/bin/env python3
"""Standalone (stdlib-only) recheck of search-agent EM from saved reward field.

eval jsonl rows = {task_id, trial, reward, messages}. No `gold` stored, so EM == reward.
Reports per-file: overall EM (mean over all rows), per-task EM (mean per task_id),
n rows, n distinct tasks, and paired McNemar (task solved iff mean reward >= 0.5) vs base.
"""
import json, sys, math
from statistics import mean
from pathlib import Path


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def by_task(rows):
    d = {}
    for r in rows:
        d.setdefault(str(r["task_id"]), []).append(float(r["reward"]))
    return d


def overall(rows):
    return mean(float(r["reward"]) for r in rows) if rows else 0.0


def task_mean(bt):
    return {t: mean(v) for t, v in bt.items()}


def mcnemar_exact(b, c):
    """two-sided exact binomial p on discordant pairs b,c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Binom(n,0.5), two-sided
    cum = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cum)


def paired(base_bt, cand_bt):
    bm, cm = task_mean(base_bt), task_mean(cand_bt)
    common = sorted(set(bm) & set(cm))
    up = sum(1 for t in common if cm[t] >= 0.5 > bm[t])
    down = sum(1 for t in common if bm[t] >= 0.5 > cm[t])
    return up, down, mcnemar_exact(up, down), len(common)


def main():
    base_path = sys.argv[1]
    evals = sys.argv[2:]
    base_rows = load(base_path)
    base_bt = by_task(base_rows)
    base_em = overall(base_rows)
    print(f"{'file':14s} {'rows':>5s} {'tasks':>6s} {'EM':>7s} {'dEM':>8s} {'up/down':>9s} {'McNp':>8s}")
    print(f"{'base':14s} {len(base_rows):>5d} {len(base_bt):>6d} {base_em:>7.4f}")
    for item in evals:
        label, path = item.split("=", 1)
        if not Path(path).exists():
            print(f"{label:14s}  MISSING {path}"); continue
        rows = load(path)
        bt = by_task(rows)
        em = overall(rows)
        up, down, p, ncommon = paired(base_bt, bt)
        star = " *" if p < 0.05 else ""
        print(f"{label:14s} {len(rows):>5d} {len(bt):>6d} {em:>7.4f} {em-base_em:>+8.4f} "
              f"{f'{up}/{down}':>9s} {p:>8.4f}{star}")


if __name__ == "__main__":
    main()
