#!/usr/bin/env python3
"""Offline analysis for search-agent eval jsonls.

Given a base eval file and one or more checkpoint eval files (each row =
{task_id, trial, reward, messages, [gold]}), report for every checkpoint:
  - EM   (exact match)   : overall + paired McNemar vs base
  - F1   (token-F1)      : overall + paired per-task delta + bootstrap 95% CI vs base
  - behavior             : mean answer length (chars), searches, assistant turns

EM/F1 are recomputed from the transcript + stored `gold` when present (so one eval
pass yields BOTH metrics); if `gold` is absent (older eval files) EM falls back to the
saved `reward` field and F1 is skipped. Pure python — no GPU/torch.

Usage:
  python analyze_search_eval.py --base out/base_eval.jsonl \
    --evals iter1=out/adapter_iter1_eval.jsonl iter2=out/adapter_iter2_eval.jsonl ...
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import mcnemar_exact, bootstrap_ci, sign_test  # noqa: E402
from qa_reward import qa_reward  # noqa: E402
from search_agent import process_score  # noqa: E402  (retrieval-hit / process metrics)

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)


def episode_stats(messages):
    a = [m for m in messages if m.get("role") == "assistant"]
    text = " ".join((m.get("content") or "") for m in a)
    mm = _ANS.findall(text)
    ans = mm[-1].strip() if mm else ""
    return ans, len(ans), text.count("<search>"), len(a)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def summarize(rows):
    """-> dict: em_by_task, f1_by_task (or None), behavior means, n."""
    em_by, f1_by = {}, {}
    chars, searches, turns, rhits = [], [], [], []
    have_gold = False
    for d in rows:
        ans, c, s, t = episode_stats(d.get("messages", []))
        chars.append(c); searches.append(s); turns.append(t)
        tid = str(d["task_id"])
        gold = d.get("gold")
        if gold:
            have_gold = True
            em = qa_reward(f"<answer>{ans}</answer>", gold, mode="em")
            f1 = qa_reward(f"<answer>{ans}</answer>", gold, mode="f1")
            f1_by.setdefault(tid, []).append(f1)
            rhits.append(process_score(d.get("messages", []), gold)["retrieval_hit"])
        else:
            em = float(d["reward"])
        em_by.setdefault(tid, []).append(em)
    return {
        "n": len(rows),
        "em_by": em_by,
        "f1_by": f1_by if have_gold else None,
        "chars": mean(chars) if chars else 0.0,
        "searches": mean(searches) if searches else 0.0,
        "turns": mean(turns) if turns else 0.0,
        "rhit": mean(rhits) if rhits else None,  # fraction of Qs whose retrieved passages contained a gold
    }


def task_mean(by):
    return {t: mean(v) for t, v in by.items()}


def overall(by):
    vals = [x for v in by.values() for x in v]
    return mean(vals) if vals else 0.0


def paired_em(base_by, cand_by):
    """McNemar on per-task solved (mean>=0.5). -> (net, p, up, down)."""
    bm, cm = task_mean(base_by), task_mean(cand_by)
    common = sorted(set(bm) & set(cm))
    up = sum(1 for t in common if cm[t] >= 0.5 > bm[t])
    down = sum(1 for t in common if bm[t] >= 0.5 > cm[t])
    return up - down, mcnemar_exact(up, down), up, down


def paired_f1(base_by, cand_by):
    """per-task F1 delta -> (mean delta, ci_lo, ci_hi, sign_p)."""
    bm, cm = task_mean(base_by), task_mean(cand_by)
    common = sorted(set(bm) & set(cm))
    deltas = [cm[t] - bm[t] for t in common]
    lo, hi = bootstrap_ci(deltas)
    return (mean(deltas) if deltas else 0.0), lo, hi, sign_test(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--evals", nargs="+", required=True,
                    help="label=path entries, e.g. iter1=out/adapter_iter1_eval.jsonl")
    args = ap.parse_args()

    base = summarize(load(args.base))
    base_em_overall = overall(base["em_by"])
    base_f1_overall = overall(base["f1_by"]) if base["f1_by"] else None

    def rh(x):
        return f"{x:.3f}" if x is not None else "  -  "

    print(f"{'ckpt':10s} {'n':>4s} {'EM':>6s} {'dEM':>6s} {'McNp':>7s} {'F1':>6s} {'dF1':>7s} "
          f"{'F1_CI':>16s} {'rHit':>6s} {'ans_ch':>6s} {'srch':>5s}")
    bf = f"{base_f1_overall:.3f}" if base_f1_overall is not None else "  -  "
    print(f"{'base':10s} {base['n']:>4d} {base_em_overall:>6.3f} {'':>6s} {'':>7s} "
          f"{bf:>6s} {'':>7s} {'':>16s} {rh(base['rhit']):>6s} {base['chars']:>6.1f} {base['searches']:>5.2f}")

    for item in args.evals:
        if "=" not in item:
            print(f"  (skip malformed --evals entry: {item})"); continue
        label, path = item.split("=", 1)
        if not Path(path).exists():
            print(f"{label:10s}  (missing: {path})"); continue
        s = summarize(load(path))
        em_o = overall(s["em_by"])
        net, p, up, down = paired_em(base["em_by"], s["em_by"])
        if s["f1_by"] and base["f1_by"]:
            f1_o = overall(s["f1_by"])
            dF1, lo, hi, _ = paired_f1(base["f1_by"], s["f1_by"])
            f1s, df1s, cis = f"{f1_o:.3f}", f"{dF1:+.3f}", f"[{lo:+.3f},{hi:+.3f}]"
        else:
            f1s, df1s, cis = "  -  ", "   -   ", "       -        "
        star = " *" if p < 0.05 else ""
        print(f"{label:10s} {s['n']:>4d} {em_o:>6.3f} {em_o-base_em_overall:>+6.3f} "
              f"{p:>7.4f} {f1s:>6s} {df1s:>7s} {cis:>16s} {rh(s['rhit']):>6s} {s['chars']:>6.1f} {s['searches']:>5.2f}"
              f"  (EM up/down {up}/{down}){star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
