"""Minimal GRPO policy update over collected rollouts (single GPU, QLoRA).

R1/R2 variant = REINFORCE with a group baseline (the core of GRPO without PPO
clipping / KL — added later if needed). For each task we sample a GROUP of N
rollouts; advantage_i = (reward_i - group_mean) / (group_std + eps). We then do
advantage-weighted, assistant-only token NLL:

    loss = mean_i [ adv_i * sum(NLL over assistant tokens_i) / norm_i ]
    norm_i = L_i        (vanilla)   OR   sqrt(L_i)   (LATA, --lata)

Minimizing adv*NLL raises prob of high-advantage rollouts and lowers it for
low-advantage ones. LATA's sqrt(L) keeps long-trajectory gradients from decaying
linearly (preserves multi-turn reasoning incentive).

Reward modes: binary (tau2 outcome; default) or prm_lite (outcome + beta*process).

Assistant token spans are found by render-twice-diff over the Qwen chat template
(matches our SFT convention). Defensive: spans that don't token-align are skipped.

Reuses model/tokenizer helpers from scripts/train_sft_smoke.py for consistency.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for p in (str(SCRIPT_DIR), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from prm_lite_reward import shaped_reward  # noqa: E402


# inlined from scripts/train_sft_smoke.py so grpo_update is self-contained on a minimal deploy
def cached_snapshot_path(model_name):
    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit)
    if "/" not in model_name:
        return model_name
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_name.replace("/", "--"))
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return model_name
    ref_path = cache_root / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.exists():
            return str(candidate)
    subs = [d for d in snapshots.iterdir() if d.is_dir()]
    return str(subs[0]) if subs else model_name


def load_tokenizer(model_name, local_files_only):
    from transformers import AutoTokenizer
    source = cached_snapshot_path(model_name) if local_files_only else model_name
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True,
                                        local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def load_rollouts(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_advantages(rollouts: list[dict], reward_mode: str, domain: str, beta: float,
                       gate: bool = True, denom_eps: float = 0.25, adv_clip: float = 3.0) -> None:
    """Adds 'reward_shaped' and 'advantage' to each rollout in place.

    GATE (default on): a task group teaches GRPO only if its BINARY outcomes contain BOTH a
    success and a failure. All-same-outcome groups get advantage 0 (then dropped by the
    abs<1e-8 filter in main). This kills the prm_lite "phantom advantage" R4 fabricated on
    all-fail groups (~63% length-concordant = training the model to GIVE UP FASTER, the
    mechanistic cause of the iter1 0.35->0.20 dip). Denominator = (sd + denom_eps) and the
    advantage is clipped to +-adv_clip so tiny shaped-reward gaps can't blow up to unit scale."""
    for r in rollouts:
        if reward_mode == "prm_lite":
            outcome = 1.0 if r["reward"] >= 1.0 - 1e-6 else 0.0
            r["reward_shaped"] = shaped_reward(outcome, r.get("messages", []), domain=domain, beta=beta)
        else:
            r["reward_shaped"] = float(r["reward"])
    by_task: dict[str, list[dict]] = {}
    for r in rollouts:
        by_task.setdefault(str(r["task_id"]), []).append(r)
    for group in by_task.values():
        rs_g = [g["reward_shaped"] for g in group]
        # GATE: skip groups with no contrast. For binary/prm_lite gate on the BINARY outcome
        # (a group teaches only if it has both a success and a failure); for a CONTINUOUS reward
        # (e.g. token-F1) binary-thresholding at 1.0 would wrongly kill every partial-credit group,
        # so gate on the continuous-reward variance instead.
        if reward_mode == "continuous":
            no_contrast = len(group) < 2 or pstdev(rs_g) == 0.0
        else:
            outs = [1.0 if g["reward"] >= 1.0 - 1e-6 else 0.0 for g in group]
            no_contrast = len(group) < 2 or pstdev(outs) == 0.0
        if gate and no_contrast:
            for g in group:
                g["advantage"] = 0.0
            continue
        rs = [g["reward_shaped"] for g in group]
        m = mean(rs)
        sd = pstdev(rs) if len(rs) > 1 else 0.0
        for g in group:
            a = (g["reward_shaped"] - m) / (sd + denom_eps)
            g["advantage"] = max(-adv_clip, min(adv_clip, a))


def normalize_messages(messages: list[dict]) -> list[dict]:
    """tau2 sims carry ~20 extra fields/msg and FLAT tool_calls ({id,name,arguments}).
    Strip to clean OpenAI/Qwen shape and NEST tool_calls so render-twice-diff does not
    depend on the chat template tolerating tau2's dialect (verified: 58/58 on 4.57.6)."""
    out = []
    for m in messages:
        role = m.get("role")
        nm = {"role": role, "content": m.get("content") or ""}
        if role == "assistant" and m.get("tool_calls"):
            ntc = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)  # accept nested or flat
                arguments = fn.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False)
                ntc.append({"id": tc.get("id", ""), "type": "function",
                            "function": {"name": fn.get("name"), "arguments": arguments}})
            nm["tool_calls"] = ntc
        if role == "tool":
            nm["tool_call_id"] = m.get("tool_call_id") or m.get("id") or ""
        out.append(nm)
    return out


def render_with_assistant_mask(tok, messages: list[dict]) -> tuple[list[int], list[int]] | None:
    """Full token ids + per-token assistant mask via render-twice-diff."""
    try:
        full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        return None
    mask = [0] * len(full)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        try:
            pre = tok.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
            inc = tok.apply_chat_template(messages[:i + 1], tokenize=True, add_generation_prompt=False)
        except Exception:
            continue
        lo, hi = len(pre), len(inc)
        if lo < hi <= len(full) and list(full[:lo]) == list(pre):
            for j in range(lo, hi):
                mask[j] = 1
    return list(full), mask


def assert_render_sane(tok) -> None:
    """Fail FAST (not after a wasted rollout collection) if the chat template's
    assistant-token detection is broken. transformers v5 silently yields all-zero
    masks here (verified: 0/58 on 5.12.1 vs 58/58 on 4.57.6) -> pin transformers 4.57.x."""
    probe = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello there, this is a probe reply"}]
    rm = render_with_assistant_mask(tok, probe)
    if rm is None or sum(rm[1]) == 0:
        raise SystemExit(
            f"[grpo] render self-test FAILED: assistant-token mask is empty under "
            f"transformers=={transformers.__version__}. The render-twice-diff masking "
            f"breaks on transformers v5 -> pin 'transformers==4.57.6' (see requirements-training.txt)."
        )


def load_policy(base_model: str, adapter_in: str | None, targets: list[str],
                lora_r: int, lora_alpha: int, lora_dropout: float,
                grad_ckpt: bool) -> torch.nn.Module:
    source = cached_snapshot_path(base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, quantization_config=bnb, device_map={"": 0},
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    if adapter_in:
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
        print(f"[grpo] continued from adapter {adapter_in}")
    else:
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model = get_peft_model(model, cfg)
        print("[grpo] fresh LoRA adapter")
    model.print_trainable_parameters()
    return model


def rollout_loss(model, device, ids: list[int], mask: list[int], advantage: float, lata: bool) -> torch.Tensor | None:
    n_asst = sum(mask)
    if n_asst == 0:
        return None
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    mshift = torch.tensor(mask[1:], dtype=torch.float32, device=device)
    sum_logp = (logp * mshift).sum()
    norm = math.sqrt(n_asst) if lata else float(n_asst)
    # minimize adv * NLL == minimize -adv * logp
    return (-advantage * sum_logp) / norm


def batched_loss(model, device, batch, pad_id, lata, kl_coef=0.0):
    """Advantage-weighted assistant-token NLL over a PADDED batch of rollouts (fills the GPU →
    much faster than the per-rollout path). batch = list of (ids, mask, advantage).
    Equivalent per-sequence math to rollout_loss; pad positions (attn=0, mask=0) don't contribute.
    Returns the mean loss over the batch (or None).

    kl_coef > 0 adds a KL-to-reference anchor: loss += kl_coef * mean_token_KL(policy || base),
    where the reference is THIS model with the LoRA adapter disabled (no second model loaded).
    Uses Schulman's k3 estimator (exp(r)-r-1, r=ref_logp-cur_logp), which is always >=0 and
    low-variance. Anchors the policy to the base so it can't drift into the answer-length-collapse
    degenerate that over-optimization falls into (the iter-6 regression we observed)."""
    B = len(batch)
    maxlen = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    maskf = torch.zeros((B, maxlen), dtype=torch.float32)
    advs = torch.tensor([float(a) for *_, a in batch], dtype=torch.float32)
    for i, (ids, m, _a) in enumerate(batch):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
        maskf[i, :L] = torch.tensor(m, dtype=torch.float32)
    input_ids = input_ids.to(device); attn = attn.to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]                       # [B, T-1, V] (model dtype; CE upcasts internally)
    targets = input_ids[:, 1:]                           # [B, T-1]
    V = logits.size(-1)
    nll = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                          reduction="none").view(B, -1)   # [B, T-1] = -logp per token
    mshift = maskf[:, 1:].to(device)                     # [B, T-1]; 0 on pad + non-assistant
    sum_nll = (nll * mshift).sum(dim=1)                  # [B] = -sum_logp over assistant tokens
    n_asst = mshift.sum(dim=1).clamp(min=1.0)            # [B]
    norm = n_asst.sqrt() if lata else n_asst
    per_seq = (advs.to(device) * sum_nll) / norm          # [B]; == (-adv * sum_logp) / norm
    loss = per_seq.mean()
    if kl_coef > 0.0:
        with torch.no_grad():                            # reference = base policy (LoRA disabled)
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
            ref_nll = F.cross_entropy(ref_logits.reshape(-1, V), targets.reshape(-1),
                                      reduction="none").view(B, -1)
        r = (ref_nll - nll).clamp(-10.0, 10.0)            # ref_logp - cur_logp; ref is constant (no_grad)
        kl_tok = torch.exp(r) - r - 1.0                   # k3 estimator, >= 0
        kl_per_seq = (kl_tok * mshift).sum(dim=1) / n_asst
        loss = loss + kl_coef * kl_per_seq.mean()
    if not torch.isfinite(loss):
        return None
    return loss


def main() -> int:
    ap = argparse.ArgumentParser(description="One GRPO update pass over collected rollouts (QLoRA).")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--base-model", default="/root/autodl-tmp/models/qwen25-7b-instruct")
    ap.add_argument("--adapter-in", default=None, help="existing LoRA to continue (else fresh)")
    ap.add_argument("--out-adapter", required=True)
    ap.add_argument("--tokenizer-model", default=None, help="defaults to base-model")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--reward-mode", choices=["binary", "prm_lite", "continuous"], default="binary",
                    help="binary=gate on 0/1 outcome (default); prm_lite=outcome+process; "
                         "continuous=use the raw reward (e.g. token-F1) and gate on its variance.")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--lata", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rft", action="store_true",
                    help="Rejection-sampling fine-tune: treat EVERY rollout as a positive target (advantage=+1, plain mean-NLL, no group baseline, no LATA). Feed a success-only dataset.")
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                    help="drop task groups with no binary-outcome variance (default ON; --no-gate restores R4 behavior).")
    ap.add_argument("--denom-eps", type=float, default=0.25, help="advantage denominator = (group_std + denom_eps).")
    ap.add_argument("--adv-clip", type=float, default=3.0, help="clip |advantage| to this magnitude.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL-to-base anchor weight (batched path only). 0 = off (default). "
                         "Penalizes drift from the base policy to resist over-optimization collapse.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default=DEFAULT_TARGETS)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="rollouts per forward/backward (padded). >1 fills the GPU and is several x faster; 1 = legacy per-rollout path.")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="print a progress line every N optimizer steps so the run isn't a black box.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    tok = load_tokenizer(args.tokenizer_model or args.base_model, local_files_only=False)
    assert_render_sane(tok)  # abort BEFORE loading the 7B if the template masking is broken

    rollouts = load_rollouts(Path(args.rollouts))
    if args.rft:
        # Rejection-sampling fine-tune: every (successful) rollout is a positive target.
        for r in rollouts:
            r["reward_shaped"] = 1.0
            r["advantage"] = 1.0
        args.lata = False
        print(f"[grpo] RFT mode: {len(rollouts)} rollouts, advantage=+1 (plain mean-NLL), LATA off")
    else:
        compute_advantages(rollouts, args.reward_mode, args.domain, args.beta,
                           gate=args.gate, denom_eps=args.denom_eps, adv_clip=args.adv_clip)
    # keep only rollouts with non-zero advantage and renderable assistant tokens
    prepared = []
    skipped = 0
    for r in rollouts:
        if abs(r["advantage"]) < 1e-8:
            continue
        rm = render_with_assistant_mask(tok, normalize_messages(r["messages"]))
        if rm is None or sum(rm[1]) == 0:
            skipped += 1
            continue
        ids, mask = rm
        if len(ids) > args.max_seq_len:  # right-truncate (keep prompt head); R1-simple
            ids, mask = ids[:args.max_seq_len], mask[:args.max_seq_len]
        prepared.append((ids, mask, float(r["advantage"])))
    print(f"[grpo] rollouts={len(rollouts)} usable={len(prepared)} skipped_render={skipped}")
    if not prepared:
        raise SystemExit("No usable rollouts (no advantage variance or no assistant tokens).")

    model = load_policy(args.base_model, args.adapter_in,
                        [t.strip() for t in args.lora_target_modules.split(",") if t.strip()],
                        args.lora_r, args.lora_alpha, args.lora_dropout, not args.no_grad_ckpt)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if args.batch_size > 1:
        prepared.sort(key=lambda x: len(x[0]))  # length-bucket -> minimal padding waste
    n_units = (len(prepared) + args.batch_size - 1) // args.batch_size if args.batch_size > 1 else len(prepared)
    print(f"[grpo] training: batch_size={args.batch_size} units={n_units} epochs={args.epochs}", flush=True)
    losses = []
    step = 0
    for ep in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        if args.batch_size > 1:
            for bi, bstart in enumerate(range(0, len(prepared), args.batch_size)):
                loss = batched_loss(model, device, prepared[bstart:bstart + args.batch_size], pad_id, args.lata, args.kl_coef)
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
                losses.append(float(loss.detach().cpu()))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % args.progress_every == 0:
                    print(f"[grpo] ep{ep+1} step {step} batch {bi+1}/{n_units} loss={mean(losses[-args.progress_every:]):.4f}", flush=True)
        else:
            for k, (ids, mask, adv) in enumerate(prepared):
                loss = rollout_loss(model, device, ids, mask, adv, args.lata)
                if loss is None or not torch.isfinite(loss):
                    continue
                (loss / args.grad_accum).backward()
                losses.append(float(loss.detach().cpu()))
                if (k + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                    opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if (k + 1) % (args.progress_every * args.grad_accum) == 0:
                    print(f"[grpo] step {step} rollout {k+1}/{len(prepared)} loss={mean(losses[-50:]):.4f}", flush=True)
            # flush remainder (per-rollout path)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
        print(f"[grpo] epoch {ep+1}/{args.epochs} opt_steps={step} mean_loss={mean(losses) if losses else float('nan'):.4f}", flush=True)

    out_dir = Path(args.out_adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    metrics = {
        "rollouts": len(rollouts), "usable": len(prepared), "opt_steps": step,
        "mean_loss": mean(losses) if losses else None,
        "reward_mode": args.reward_mode, "lata": args.lata, "lr": args.lr,
        "rft": args.rft, "gate": args.gate,
        "advantages": {
            "mean_abs": mean([abs(a) for *_, a in prepared]),
            "n_pos": sum(1 for *_, a in prepared if a > 0),
            "n_neg": sum(1 for *_, a in prepared if a < 0),
        },
        "max_cuda_mem_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }
    (out_dir / "grpo_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("[grpo] saved adapter ->", out_dir)
    print("[grpo] metrics:", json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
