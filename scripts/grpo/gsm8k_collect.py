"""Collect GSM8K rollouts for verifiable-reward RL (RLVR) — single-turn, exact-match reward.

No user-simulator, no tau2: each rollout is one (question -> chain-of-thought + answer) turn,
scored by DETERMINISTIC exact-match on the final integer. This removes the user-sim / judge
noise that capped the airline project, so GRPO can actually show a clean positive curve.

Output JSONL matches the GRPO trainer's format (one rollout/line), so grpo_update.py consumes
it unchanged:  {"task_id","trial","reward","messages":[user, assistant]}

A vLLM server must already serve the policy at --api-base (run_gsm8k.sh handles start/stop).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT = ("Solve the math problem step by step. "
          "End your response with a line of exactly the form '#### <final integer answer>'.\n\n"
          "Problem: {q}")

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_num(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return None


def extract_pred(text: str) -> str | None:
    """Predicted answer: number after the last '####', else the last number in the text."""
    if "####" in text:
        m = _NUM.search(text.split("####")[-1])
        if m:
            return _norm_num(m.group())
    nums = _NUM.findall(text or "")
    return _norm_num(nums[-1]) if nums else None


def extract_gold(answer_field: str) -> str | None:
    """GSM8K gold answer ends with '#### <n>'."""
    if "####" in answer_field:
        m = _NUM.search(answer_field.split("####")[-1])
        return _norm_num(m.group()) if m else None
    nums = _NUM.findall(answer_field or "")
    return _norm_num(nums[-1]) if nums else None


def is_correct(pred: str | None, gold: str | None) -> bool:
    return pred is not None and gold is not None and pred == gold


def load_gsm8k(data: str, split: str, n: int | None) -> list[tuple[str, str]]:
    """Returns [(question, gold_answer_field), ...]. `data` is 'gsm8k' (HF) or a .jsonl path
    with {"question","answer"} per line."""
    items: list[tuple[str, str]] = []
    if data.endswith(".jsonl"):
        for line in Path(data).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                items.append((d["question"], d["answer"]))
    else:
        from datasets import load_dataset  # noqa: F401
        ds = load_dataset("gsm8k", "main", split=split)
        items = [(ex["question"], ex["answer"]) for ex in ds]
    if n and n < len(items):
        items = items[:n]
    return items


def call_vllm(api_base: str, model: str, messages: list[dict], temperature: float,
              max_tokens: int, n: int = 1, timeout: int = 240) -> list[str]:
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "n": n}
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read())
    return [(c.get("message", {}).get("content") or "") for c in d.get("choices", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect GSM8K RLVR rollouts (exact-match reward).")
    ap.add_argument("--data", default="gsm8k", help="'gsm8k' (HF) or a .jsonl with {question,answer}")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-questions", type=int, default=64)
    ap.add_argument("--num-trials", type=int, default=8, help="group size N per question")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--served-model", default="policy")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_gsm8k(args.data, args.split, args.n_questions)
    print(f"[gsm8k] {len(items)} questions x {args.num_trials} trials @ temp {args.temperature}")

    def work(idx_item):
        idx, (q, ans) = idx_item
        gold = extract_gold(ans)
        messages = [{"role": "user", "content": PROMPT.format(q=q)}]
        try:
            comps = call_vllm(args.api_base, args.served_model, messages,
                              args.temperature, args.max_tokens, n=args.num_trials)
        except Exception as e:
            print(f"[gsm8k] q{idx} call failed: {repr(e)[:120]}", file=sys.stderr)
            comps = []
        rows = []
        for j, comp in enumerate(comps):
            r = 1.0 if is_correct(extract_pred(comp), gold) else 0.0
            rows.append({"task_id": str(idx), "trial": j, "reward": r,
                         "messages": messages + [{"role": "assistant", "content": comp}]})
        return rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rewards, by_task = [], {}
    with out_path.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        for rows in as_completed_map(ex, work, list(enumerate(items))):
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rewards.append(row["reward"])
                by_task.setdefault(row["task_id"], []).append(row["reward"])

    succ = sum(1 for r in rewards if r >= 1.0 - 1e-6)
    dead = sum(1 for v in by_task.values() if len(set(v)) <= 1)
    print(f"[gsm8k] wrote {len(rewards)} rollouts -> {out_path}")
    print(f"[gsm8k] reward: success {succ}/{len(rewards)} mean={succ/max(len(rewards),1):.3f}")
    print(f"[gsm8k] tasks with ZERO intra-group variance (no GRPO signal): {dead}/{len(by_task)}")
    return 0


def as_completed_map(ex, fn, items):
    futs = {ex.submit(fn, it): it for it in items}
    for fut in as_completed(futs):
        yield fut.result()


if __name__ == "__main__":
    raise SystemExit(main())
