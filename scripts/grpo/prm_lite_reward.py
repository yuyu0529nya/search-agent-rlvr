"""PRM-Lite style rule-based PROCESS reward for tau2 tool-agent rollouts.

This is OUR re-implementation (in our own words) of a rule-based dense process
reward for long-horizon tool agents. It directly targets the failure we diagnosed
(over-calling collapse): redundancy / length / cheap-reasoning / no-reasoning are
penalized, data-chaining and diverse reads are rewarded.

Used in M3 stage R3 (reward shaping). R1/R2 default to the binary outcome reward,
so this module is standalone and not required for the first GRPO smoke.

Final shaped reward (when enabled):  reward = outcome(0/1) + beta * process_score
process_score is the MEAN of per-step scores (so long trajectories aren't flattened),
clamped to [-0.5, 0.5].

Tool categories are domain-configurable (airline / retail) since tool names differ.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---- domain tool taxonomies (extend as needed) -----------------------------
TOOL_TAXONOMY: dict[str, dict[str, frozenset]] = {
    "airline": {
        "read": frozenset({
            "list_all_airports", "search_direct_flight", "search_onestop_flight",
            "get_user_details", "get_reservation_details", "calculate",
        }),
        "write": frozenset({
            "book_reservation", "cancel_reservation", "update_reservation_baggages",
            "update_reservation_passengers", "update_reservation_flights", "send_certificate",
        }),
        "escalation": frozenset({"transfer_to_human_agents"}),
        "think": frozenset({"think", "implicit_think"}),
    },
    "retail": {
        "read": frozenset({
            "get_user_details", "get_order_details", "get_product_details",
            "list_all_product_types", "calculate",
        }),
        "write": frozenset({
            "cancel_pending_order", "exchange_delivered_order", "return_delivered_order",
            "modify_pending_order_address", "modify_pending_order_items",
            "modify_pending_order_payment", "modify_user_address",
        }),
        "escalation": frozenset({"transfer_to_human_agents"}),
        "think": frozenset({"think", "implicit_think"}),
    },
}

_PARAM_PATTERNS = {
    "reservation_id": re.compile(r"^[A-Z0-9]{6}$"),
    "user_id": re.compile(r"^[a-z]+_[a-z]+_[0-9]+$"),
    "order_id": re.compile(r"^#?W[0-9]+$"),
    "flight_number": re.compile(r"^[A-Z]{3}[0-9]{3}$"),
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
}
_PLACEHOLDER_KEYWORDS = frozenset({
    "previous", "unknown", "placeholder", "none", "null", "n/a", "any", "some",
    "first", "last", "default", "example", "sample", "test", "dummy", "temp",
})


def _param_str(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False).lower()


def _is_placeholder_param(field_name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lower = value.lower()
    if any(kw in lower for kw in _PLACEHOLDER_KEYWORDS):
        return True
    for key, pattern in _PARAM_PATTERNS.items():
        if key in field_name.lower():
            return not pattern.match(value)
    return False


def _has_placeholder(params: dict) -> bool:
    return any(_is_placeholder_param(k, v) for k, v in (params or {}).items())


def _is_redundant(prev_actions: list[dict], tool: str, params: dict, think: frozenset, window: int = 3) -> bool:
    sig = (tool, _param_str(params))
    for prev in prev_actions[-window:]:
        if prev.get("tool") in think:
            continue
        if (prev.get("tool"), prev.get("param_str", "")) == sig:
            return True
    return False


def compute_process_score(action_history: list[dict], domain: str = "airline") -> float:
    """Mean per-step reasoning-quality score, clamped to [-0.5, 0.5].

    Each action dict should have: tool, parameters, param_str, is_error,
    content (assistant text before the call), extracted_entities (dict).
    """
    tax = TOOL_TAXONOMY.get(domain, TOOL_TAXONOMY["airline"])
    READ, WRITE, ESC, THINK = tax["read"], tax["write"], tax["escalation"], tax["think"]
    if not action_history:
        return 0.0

    per_step = []
    for i, a in enumerate(action_history):
        tool = a.get("tool", "")
        params = a.get("parameters", {}) or {}
        pstr = a.get("param_str", "")
        s = 0.0

        # placeholder params
        if tool not in THINK and _has_placeholder(params):
            s += -0.05 if tool in WRITE else -0.03
        # redundancy (same tool+params recently)
        if tool not in THINK and _is_redundant(action_history[:i], tool, params, THINK):
            s -= 0.03
        # error repetition vs recovery
        if i >= 1 and tool not in THINK:
            prev = action_history[i - 1]
            if prev.get("is_error"):
                same = (prev.get("tool"), prev.get("param_str", "")) == (tool, pstr)
                s += -0.04 if same else 0.05
        # escalation without prior read
        if tool in ESC:
            did_read = any(p.get("tool") in READ for p in action_history[:i])
            s += -0.10 if not did_read else -0.05
        # data chain: param value reused from earlier extracted entities
        if i >= 1 and tool not in THINK and params:
            seen = set()
            for p in action_history[:i]:
                for ents in (p.get("extracted_entities", {}) or {}).values():
                    seen.update(ents)
            if any(isinstance(v, str) and v in seen for v in params.values()):
                s += 0.08 if tool in WRITE else 0.04
        # first-time read diversity
        if tool in READ:
            seen_reads = {p["tool"] for p in action_history[:i] if p.get("tool") in READ}
            if tool not in seen_reads:
                s += 0.01
        # think bonus with anti-hacking
        if tool in THINK:
            prev_think = i >= 1 and action_history[i - 1].get("tool") in THINK
            is_last = i == len(action_history) - 1
            if not prev_think and not is_last:
                nxt = action_history[i + 1] if i + 1 < len(action_history) else {}
                np_ = nxt.get("parameters", {})
                if not (_has_placeholder(np_) or _is_redundant(action_history[:i + 1], nxt.get("tool", ""), np_, THINK)):
                    s += 0.01
        # cheap reasoning: tool call with <30 chars of assistant text
        if tool not in THINK:
            c = a.get("content", "")
            if 0 < len(c) < 30:
                s -= 0.02
        per_step.append(s)

    mean_score = sum(per_step) / len(per_step)

    # trajectory-level adjustments
    think_count = sum(1 for a in action_history if a.get("tool") in THINK)
    if think_count == 0 and len(action_history) >= 3:
        mean_score -= 0.05
    if len({a["tool"] for a in action_history if a.get("tool") in READ}) >= 3:
        mean_score += 0.01
    # raised from 8 -> 16: airline tasks legitimately need many reads/writes (observed mean
    # ~6.6, max ~16 turns); only penalize genuinely runaway trajectories, not normal long ones.
    length_threshold = 16
    if len(action_history) > length_threshold:
        mean_score += -0.01 * (len(action_history) - length_threshold)

    return float(max(-0.5, min(0.5, mean_score)))


def _collect_entity_strings(obj: Any, out: set) -> None:
    """Recursively gather entity-like leaf strings (ids, names, numbers) from a parsed
    tool-RESULT JSON, so the data-chaining rule can detect a later param reusing them."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and len(k) >= 4:  # ids like reservation_id are dict KEYS in tau2 results
                out.add(k)
            _collect_entity_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_entity_strings(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (str, int, float)):
        s = str(obj)
        if len(s) >= 4:  # skip trivial tokens ("1", "ok", "true")
            out.add(s)


def _result_entities(content: str) -> set:
    out: set = set()
    try:
        _collect_entity_strings(json.loads(content), out)
    except Exception:
        pass
    return out


def action_history_from_messages(messages: list[dict]) -> list[dict]:
    """Extract a tool-call action history from a tau2/OpenAI-format message list.

    Each assistant tool_call is paired to ITS OWN tool result by tool_call_id (fallback:
    next tool message), so multi-call turns set is_error correctly AND extracted_entities
    is populated from the result JSON -- which revives the data-chaining process rule that
    was previously dead (extracted_entities was hardcoded {}).
    `content` = the assistant text on the same turn (for the cheap-reasoning rule).
    """
    results_by_id: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "tool":
            rid = m.get("tool_call_id") or m.get("id")
            if rid is not None:
                results_by_id[str(rid)] = str(m.get("content") or "")

    history: list[dict] = []
    for idx, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        text = m.get("content") or ""
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc   # tau2 native tool_calls are FLAT ({name,arguments}); OpenAI nests under "function"
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args or "{}")
                except Exception:
                    args = {}
            # this call's OWN result: by id, else the next tool message after this turn
            cid = tc.get("id")
            if cid is not None and str(cid) in results_by_id:
                rc = results_by_id[str(cid)]
            else:
                rc = ""
                for nxt in messages[idx + 1:]:
                    if nxt.get("role") == "tool":
                        rc = str(nxt.get("content") or "")
                        break
            rcl = rc.lower()
            is_error = ("error" in rcl) or ("not found" in rcl) or ("invalid" in rcl)
            history.append({
                "tool": name,
                "parameters": args,
                "param_str": _param_str(args),
                "is_error": is_error,
                "content": text,
                "extracted_entities": {"result": _result_entities(rc)},
            })
    return history


def shaped_reward(outcome: float, messages: list[dict], domain: str = "airline", beta: float = 0.3) -> float:
    """reward = outcome(0/1) + beta * process_score."""
    hist = action_history_from_messages(messages)
    return float(outcome) + beta * compute_process_score(hist, domain=domain)


if __name__ == "__main__":
    # tiny self-test (no GPU / no deps beyond stdlib)
    demo = [
        {"role": "assistant", "content": "Let me look up your account first.",
         "tool_calls": [{"function": {"name": "get_user_details", "arguments": "{\"user_id\": \"yusuf_rossi_9876\"}"}}]},
        {"role": "tool", "content": "{...ok...}"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"function": {"name": "get_user_details", "arguments": "{\"user_id\": \"yusuf_rossi_9876\"}"}}]},
        {"role": "tool", "content": "{...ok...}"},
    ]
    h = action_history_from_messages(demo)
    print("actions:", len(h))
    print("process_score:", compute_process_score(h, "retail"))
    print("shaped (outcome=0):", shaped_reward(0.0, demo, "retail"))
