"""Verifiable QA reward for the search-agent RLVR (Search-R1 style).

Normalized exact-match + token-F1 (SQuAD convention). Deterministic and verifiable — the same
clean-reward property that produced the GSM8K win, now for open-ended short-answer QA. Supports
multiple gold aliases (HotpotQA / 2WikiMQA answers + aliases).
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, collapse whitespace."""
    s = (s or "").lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(text: str) -> str:
    """Pull the agent's final answer: prefer <answer>...</answer>, else 'the answer is X', else
    the last non-empty line."""
    if text is None:
        return ""
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:the\s+)?answer(?:\s+is)?\s*:?\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def qa_reward(pred_text: str, golds, mode: str = "em") -> float:
    """Reward for one rollout. `golds` = a gold string or list of acceptable aliases.
    mode 'em' = binary exact-match (recommended primary signal, like GSM8K);
    mode 'f1' = token-F1 (denser, for shaping); mode 'em_or_f1half' = em, else 0.5*f1."""
    if isinstance(golds, str):
        golds = [golds]
    golds = [g for g in golds if g is not None and str(g).strip()]
    if not golds:
        return 0.0
    pred = extract_answer(pred_text)
    em = max(exact_match(pred, g) for g in golds)
    if mode == "em":
        return em
    f1 = max(token_f1(pred, g) for g in golds)
    if mode == "f1":
        return f1
    if mode == "em_or_f1half":
        return em if em > 0 else 0.5 * f1
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    # stdlib self-test
    assert normalize("The  Eiffel Tower.") == "eiffel tower"
    assert exact_match("the answer is Paris", "Paris") == 0.0  # exact_match is on raw strings; use qa_reward for extraction
    assert exact_match("Paris.", "paris") == 1.0
    assert abs(token_f1("New York City", "New York") - (2 * (2/3) * 1 / (2/3 + 1))) < 1e-9
    assert extract_answer("reasoning...\n<answer>Barack Obama</answer>") == "Barack Obama"
    assert extract_answer("So the answer is: 1969.") == "1969"
    assert extract_answer("blah\nFinal: Tokyo") == "Final: Tokyo"
    assert qa_reward("<answer>Paris</answer>", ["Paris", "paris, france"]) == 1.0
    assert qa_reward("I think it is the answer is George Washington", ["George Washington"]) == 1.0
    assert qa_reward("<answer>wrong</answer>", ["right"], mode="em") == 0.0
    assert 0.0 < qa_reward("<answer>New York</answer>", ["New York City"], mode="f1") < 1.0
    print("qa_reward self-test: ALL PASS")
