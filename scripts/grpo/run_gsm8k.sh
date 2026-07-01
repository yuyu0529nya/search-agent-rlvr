#!/usr/bin/env bash
set -euo pipefail
# Minimal verifiable-reward RL (RLVR): GRPO on GSM8K with exact-match reward.
# Single-turn, NO user-sim, NO tau2 -> deterministic reward -> a clean positive curve is expected.
# Reuses grpo_update.py + summarize_eval.py unchanged. One on-policy GRPO update (ITERS=1).
#   Recommended model: Qwen2.5-1.5B-Instruct (real headroom; 7B is ~ceiling on GSM8K).
#   First-time: download it, e.g.  huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
#   --local-dir /root/autodl-tmp/models/qwen25-1.5b-instruct   (or set POLICY_MODEL to the HF id).
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-1.5b-instruct}"
RUN="${RUN:-gsm8k_r1}"
DATA="${DATA:-gsm8k}"
N_TRAIN_Q="${N_TRAIN_Q:-128}"; N_EVAL_Q="${N_EVAL_Q:-200}"
N="${N:-8}"; ITERS="${ITERS:-4}"; LR="${LR:-2e-5}"; TEMP="${TEMP:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-640}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"; GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"; PORT=8000; HOST=127.0.0.1
export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
OUT="outputs/${RUN}"; mkdir -p "$OUT" outputs/vllm_logs
VP=""

serve() {  # $1 = adapter path (empty = base)
  pkill -9 -f vllm.entrypoints 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 3
  local nm=(--served-model-name policy)
  [ -n "$1" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$1")
  echo "[gsm8k] serve vLLM (adapter='${1:-<base>}')"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" > "outputs/vllm_logs/${RUN}.log" 2>&1 &
  VP=$!
  for _ in $(seq 1 120); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q policy && { echo "[gsm8k] vLLM ready"; return 0; }; sleep 5; done
  echo "[gsm8k] vLLM FAILED"; tail -n 25 "outputs/vllm_logs/${RUN}.log"; exit 4
}
stop() { [ -n "$VP" ] && kill "$VP" 2>/dev/null || true; [ -n "$VP" ] && wait "$VP" 2>/dev/null || true; VP=""; sleep 3; }
trap stop EXIT

echo "######## BASE eval $(date +%H:%M:%S) ########"
serve ""
$PYBIN scripts/grpo/gsm8k_eval.py --data "$DATA" --split test --n-questions "$N_EVAL_Q" \
  --out "$OUT/base_eval.jsonl" --max-concurrency "$MAX_CONCURRENCY" --max-tokens "$MAX_TOKENS"

# On-policy iterative GRPO: each iter collects with the CURRENT policy, then updates it (chained).
CUR=""
for ((it=1; it<=ITERS; it++)); do
  echo "######## ITER $it/$ITERS — collect (policy='${CUR:-<base>}') $(date +%H:%M:%S) ########"
  [ "$it" -gt 1 ] && serve "$CUR"   # iter 1 reuses the base server from the eval above
  $PYBIN scripts/grpo/gsm8k_collect.py --data "$DATA" --split train --n-questions "$N_TRAIN_Q" \
    --num-trials "$N" --temperature "$TEMP" --out "$OUT/rollouts_iter${it}.jsonl" \
    --max-concurrency "$MAX_CONCURRENCY" --max-tokens "$MAX_TOKENS"
  stop
  echo "######## ITER $it/$ITERS — GRPO update (binary+gate, lr=$LR) $(date +%H:%M:%S) ########"
  in=(); [ -n "$CUR" ] && in=(--adapter-in "$CUR")
  $PYBIN scripts/grpo/grpo_update.py --rollouts "$OUT/rollouts_iter${it}.jsonl" --base-model "$MODEL" \
    "${in[@]}" --out-adapter "$OUT/adapter_iter${it}" --reward-mode binary --gate --lr "$LR"
  CUR="$OUT/adapter_iter${it}"
done

echo "######## FINAL adapter eval $(date +%H:%M:%S) ########"
serve "$CUR"
$PYBIN scripts/grpo/gsm8k_eval.py --data "$DATA" --split test --n-questions "$N_EVAL_Q" \
  --out "$OUT/adapter_eval.jsonl" --max-concurrency "$MAX_CONCURRENCY" --max-tokens "$MAX_TOKENS"
stop

echo "######## COMPARE (exact-match, deterministic) ########"
$PYBIN scripts/grpo/summarize_eval.py --eval "$OUT/adapter_eval.jsonl" --base "$OUT/base_eval.jsonl" --live-pool ""
echo "ALLDONE_GSM8K RUN=$RUN"
