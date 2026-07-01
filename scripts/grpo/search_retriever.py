"""Self-contained BM25 retriever for the search-agent RLVR — pure stdlib, no deps, reproducible.

Builds an in-memory BM25 index over a passage corpus and serves top-k for a query. Used as the
agent's `search` tool (local corpus -> deterministic, no external API, fully reproducible, unlike
live web search). Corpus is built from the QA dataset's own paragraphs (HotpotQA / 2WikiMQA).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tok = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.dl = [len(t) for t in self.tok]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf = [Counter(t) for t in self.tok]
        df: Counter = Counter()
        for t in self.tok:
            df.update(set(t))
        self.idf = {w: math.log((self.N - df[w] + 0.5) / (df[w] + 0.5) + 1.0) for w in df}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = tokenize(query)
        scored = []
        for i in range(self.N):
            tf, dl = self.tf[i], self.dl[i]
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.docs[i], sc) for sc, i in scored[:k]]


def build_hotpotqa_corpus(split: str = "validation", n_questions: int | None = None,
                          config: str = "distractor"):
    """Returns (items, corpus). items = [{'id','question','answer','aliases'}]; corpus = list[str]
    of unique 'Title: sentence-joined-paragraph' passages pooled across the questions.
    Requires `datasets` (on the box); set HF_ENDPOINT=https://hf-mirror.com if HF is blocked."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", config, split=split, trust_remote_code=True)
    if n_questions:
        ds = ds.select(range(min(n_questions, len(ds))))
    items, seen, corpus = [], set(), []
    for ex in ds:
        items.append({"id": str(ex.get("id")), "question": ex["question"],
                      "answer": ex["answer"], "aliases": [ex["answer"]]})
        ctx = ex.get("context", {})
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        for t, ss in zip(titles, sents):
            passage = f"{t}: " + " ".join(ss)
            key = passage[:200]
            if key not in seen:
                seen.add(key)
                corpus.append(passage)
    return items, corpus


if __name__ == "__main__":
    docs = [
        "Paris: Paris is the capital and most populous city of France.",
        "Berlin: Berlin is the capital of Germany.",
        "Eiffel Tower: The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Python: Python is a high-level programming language.",
    ]
    bm = BM25(docs)
    top = bm.search("what is the capital of France", k=2)
    print("query: capital of France -> top:", [d.split(':')[0] for d, _ in top])
    assert top and top[0][0].startswith("Paris"), top
    top2 = bm.search("iron tower in Paris", k=1)
    assert top2[0][0].startswith("Eiffel"), top2
    print("BM25 self-test: ALL PASS")
