"""Greedy GSM8K eval (exact-match accuracy) for a vLLM-served policy.

Writes the same {task_id,trial,reward,messages} jsonl as gsm8k_collect, so summarize_eval.py
can pair base vs adapter (per-question McNemar). Deterministic reward -> a trustworthy delta,
unlike the airline user-sim eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_collect as G  # noqa: E402  (reuse PROMPT + extract/reward/load/call helpers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy GSM8K exact-match eval for a served policy.")
    ap.add_argument("--data", default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--served-model", default="policy")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = G.load_gsm8k(args.data, args.split, args.n_questions)
    print(f"[gsm8k-eval] {len(items)} questions, greedy temp {args.temperature}")

    def work(idx_item):
        idx, (q, ans) = idx_item
        gold = G.extract_gold(ans)
        messages = [{"role": "user", "content": G.PROMPT.format(q=q)}]
        try:
            comp = G.call_vllm(args.api_base, args.served_model, messages,
                               args.temperature, args.max_tokens, n=1)[0]
        except Exception as e:
            print(f"[gsm8k-eval] q{idx} failed: {repr(e)[:120]}", file=sys.stderr)
            comp = ""
        r = 1.0 if G.is_correct(G.extract_pred(comp), gold) else 0.0
        return {"task_id": str(idx), "trial": 0, "reward": r,
                "messages": messages + [{"role": "assistant", "content": comp}]}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rewards = []
    with out_path.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futs = [ex.submit(work, it) for it in enumerate(items)]
        for fut in as_completed(futs):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rewards.append(row["reward"])

    acc = sum(rewards) / max(len(rewards), 1)
    print(f"[gsm8k-eval] wrote {len(rewards)} -> {out_path}")
    print(f"[gsm8k-eval] accuracy (pass^1) = {acc:.3f}  ({int(sum(rewards))}/{len(rewards)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
