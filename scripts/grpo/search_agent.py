"""Multi-turn SEARCH AGENT episodes for RLVR (Search-R1 style) — collect rollouts OR eval.

Episode: system prompt tells the model it may call a local search tool by emitting
'<search>query</search>'; the env runs BM25 over a local corpus and returns passages as a
user turn; the model reasons over rounds and finally emits '<answer>...</answer>'. Reward =
verifiable QA exact-match (qa_reward). Rollouts are written in the GRPO trainer's format
({task_id,trial,reward,messages}) — assistant turns (the model's queries+reasoning+answer) are
the only trained tokens; injected search results are user turns (not trained).

Deterministic local corpus (BM25) => reproducible, unlike live web search. vLLM serves the policy.
Run with --mock to self-test the orchestration with a scripted fake model (no GPU/model needed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_reward import qa_reward, normalize  # noqa: E402
from search_retriever import BM25  # noqa: E402

SYSTEM = (
    "You are a research agent. Answer the question using a local search tool.\n"
    "To search, output exactly: <search>your query</search> and stop. You will then receive "
    "passages. You may search up to {max_searches} times.\n"
    "When you can answer, output exactly: <answer>your final short answer</answer>.\n"
    "Think briefly before each action."
)

_SEARCH = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL | re.IGNORECASE)
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _retrieved_passages(messages):
    """The passages BM25 injected as user turns (content starts with 'Search results:')."""
    return [m.get("content") or "" for m in messages
            if m.get("role") == "user" and (m.get("content") or "").startswith("Search results:")]


def _queries(messages):
    """The <search> queries the agent issued (from its assistant turns)."""
    qs = []
    for m in messages:
        if m.get("role") == "assistant":
            qs.extend(q.strip() for q in _SEARCH.findall(m.get("content") or ""))
    return qs


def process_score(messages, golds):
    """Dense PROCESS reward in [0,1] (Search-R1 / PRM-Lite style): rewards good INTERMEDIATE
    behavior so the gradient isn't only on the final-answer token overlap.
      - retrieval_hit (0.7): did any retrieved passage actually contain a gold answer? This
        gives signal even when the final answer extraction fails (the key densification).
      - efficiency  (0.3): fraction of distinct queries (penalizes repeating the same search).
    Returns a dict; callers use ['score']."""
    npass = normalize(" \n ".join(_retrieved_passages(messages)))
    hit = 0.0
    for g in golds:
        ng = normalize(g)
        if ng and ng in npass:
            hit = 1.0
            break
    qs = [normalize(q) for q in _queries(messages)]
    efficiency = (len(set(qs)) / len(qs)) if qs else 0.0  # no search => no process credit
    return {"retrieval_hit": hit, "efficiency": efficiency, "score": 0.7 * hit + 0.3 * efficiency}


def call_vllm(api_base, model, messages, temperature, max_tokens, timeout=240):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stop": ["</search>", "</answer>"]}
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"] or ""


def run_episode(question, bm25, gen_fn, max_searches=3, top_k=3):
    """gen_fn(messages)->assistant_text. Returns (messages, final_answer_text). The model's
    stop tokens are stripped, so we re-append the closing tag we matched."""
    messages = [{"role": "system", "content": SYSTEM.format(max_searches=max_searches)},
                {"role": "user", "content": f"Question: {question}"}]
    searches = 0
    final = ""
    for _ in range(max_searches + 1):
        text = gen_fn(messages)
        ans = _ANSWER.search(text + "</answer>")  # stop token was stripped; restore for parsing
        sea = _SEARCH.search(text + "</search>")
        # decide which tag the model emitted first (closest to start)
        a_pos = text.lower().find("<answer>")
        s_pos = text.lower().find("<search>")
        emitted_answer = a_pos != -1 and (s_pos == -1 or a_pos < s_pos)
        if emitted_answer and ans:
            messages.append({"role": "assistant", "content": text.split("<answer>")[0] + f"<answer>{ans.group(1)}</answer>"})
            final = ans.group(1)
            break
        if sea and searches < max_searches:
            q = sea.group(1).strip()
            messages.append({"role": "assistant", "content": text.split("<search>")[0] + f"<search>{q}</search>"})
            hits = bm25.search(q, k=top_k)
            obs = "\n".join(f"[{i+1}] {p}" for i, (p, _) in enumerate(hits)) or "(no results)"
            messages.append({"role": "user", "content": f"Search results:\n{obs}\nContinue, or give <answer>."})
            searches += 1
            continue
        # no usable tag (or out of searches): treat the text as the final answer
        messages.append({"role": "assistant", "content": text})
        final = ans.group(1) if ans else text
        break
    return messages, final


def main() -> int:
    ap = argparse.ArgumentParser(description="Search-agent RLVR episodes (collect or eval).")
    ap.add_argument("--mode", choices=["collect", "eval"], default="collect")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--config", default="distractor")
    ap.add_argument("--n-questions", type=int, default=128)
    ap.add_argument("--num-trials", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-searches", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--served-model", default="policy")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--eval-trials", type=int, default=1,
                    help="trials per question in EVAL mode (multi-trial pass^k); default 1 = legacy single-trial")
    ap.add_argument("--eval-temp", type=float, default=0.0,
                    help="sampling temperature in EVAL mode; >0 needed for multi-trial diversity (e.g. 0.7)")
    ap.add_argument("--reward-mode", default="em", choices=["em", "f1", "em_or_f1half"])
    ap.add_argument("--process-beta", type=float, default=0.0,
                    help="weight on the dense process reward (retrieval-hit + query-efficiency); "
                         "training reward = outcome + beta*process. 0 = outcome only (default). "
                         "Use only for COLLECT; keep eval at 0 so the headline stays pure EM.")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true", help="self-test orchestration with a scripted fake model")
    args = ap.parse_args()

    if args.mock:
        return _selftest()

    from search_retriever import build_hotpotqa_corpus
    items, corpus = build_hotpotqa_corpus(args.split, args.n_questions, args.config)
    bm25 = BM25(corpus)
    print(f"[search] {len(items)} questions, corpus {len(corpus)} passages")
    trials = args.eval_trials if args.mode == "eval" else args.num_trials
    temp = args.eval_temp if args.mode == "eval" else args.temperature

    def gen_fn(messages):
        return call_vllm(args.api_base, args.served_model, messages, temp, args.max_tokens)

    def work(idx_item):
        idx, it = idx_item
        rows = []
        for j in range(trials):
            try:
                msgs, final = run_episode(it["question"], bm25, gen_fn, args.max_searches, args.top_k)
            except Exception as e:
                print(f"[search] q{idx} t{j} failed: {repr(e)[:100]}", file=sys.stderr)
                continue
            r = qa_reward(f"<answer>{final}</answer>", it["aliases"], mode=args.reward_mode)
            if args.process_beta > 0:
                r = r + args.process_beta * process_score(msgs, it["aliases"])["score"]
            row = {"task_id": str(it["id"]), "trial": j, "reward": float(r), "messages": msgs}
            if args.mode == "eval":
                row["gold"] = it["aliases"]  # store golds so EM and F1 can both be recomputed offline
            rows.append(row)
        return rows

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rewards, by_task = [], {}
    with out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futs = [ex.submit(work, p) for p in enumerate(items)]
        for fut in as_completed(futs):
            for row in fut.result():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rewards.append(row["reward"]); by_task.setdefault(row["task_id"], []).append(row["reward"])
    succ = sum(1 for r in rewards if r >= 1.0 - 1e-6)
    print(f"[search] wrote {len(rewards)} -> {out}")
    print(f"[search] reward: success {succ}/{len(rewards)} mean={sum(rewards)/max(len(rewards),1):.3f}")
    if args.mode == "collect":
        dead = sum(1 for v in by_task.values() if len(set(1.0 if x>=1-1e-6 else 0.0 for x in v)) <= 1)
        print(f"[search] tasks with ZERO intra-group variance: {dead}/{len(by_task)}")
    else:
        print(f"[search] accuracy (pass^1) = {sum(rewards)/max(len(rewards),1):.3f}")
    return 0


def _selftest() -> int:
    docs = ["Marie Curie: Marie Curie was a physicist and chemist who discovered polonium and radium.",
            "Nobel Prize: Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911.",
            "Distractor: The Eiffel Tower is in Paris."]
    bm = BM25(docs)
    # scripted fake model: first turn searches, second turn answers from the results
    state = {"n": 0}
    def fake(messages):
        state["n"] += 1
        if state["n"] == 1:
            return "I should look this up. <search>what did Marie Curie discover"  # stop stripped </search>
        return "Based on the passages, <answer>polonium and radium"
    msgs, final = run_episode("What did Marie Curie discover?", bm, fake, max_searches=3, top_k=2)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant"], roles
    assert "<search>" in msgs[2]["content"] and "</search>" in msgs[2]["content"]
    assert "Search results:" in msgs[3]["content"] and "Marie Curie" in msgs[3]["content"]
    assert final.strip() == "polonium and radium"
    assert qa_reward(f"<answer>{final}</answer>", ["polonium and radium"], "em") == 1.0
    # episode that never searches, answers directly
    msgs2, f2 = run_episode("Q?", bm, lambda m: "<answer>42", max_searches=3)
    assert f2.strip() == "42" and [m["role"] for m in msgs2] == ["system", "user", "assistant"]
    print("search_agent orchestration self-test: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
