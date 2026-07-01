#!/usr/bin/env bash
set -euo pipefail
# Search-agent RLVR (Search-R1 miniature): multi-turn agent searches a LOCAL BM25 corpus,
# verifiable exact-match QA reward, on-policy GRPO. Reuses grpo_update.py + summarize_eval.py.
# Reproducible (local corpus, no live web). Start with 1.5B to validate cheaply, then 7B.
#   First time on box: ensure `datasets` can load hotpot_qa (HF_ENDPOINT=https://hf-mirror.com);
#   if datasets>=3 dropped the loader script, load the parquet directly (see NEXT_RUN.md).
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-1.5b-instruct}"
RUN="${RUN:-search_r1}"; SPLIT="${SPLIT:-validation}"; CONFIG="${CONFIG:-distractor}"
N_TRAIN_Q="${N_TRAIN_Q:-128}"; N_EVAL_Q="${N_EVAL_Q:-300}"
N="${N:-8}"; ITERS="${ITERS:-4}"; LR="${LR:-2e-5}"; TEMP="${TEMP:-1.0}"
BATCH="${BATCH:-4}"; GRPO_SEQ="${GRPO_SEQ:-2560}"   # batched GRPO update: fill the GPU (~4x faster). bump BATCH if VRAM allows.
PROG="${PROG:-5}"   # print a [grpo] progress line every N opt-steps so the update is never a black box.
REWARD="${REWARD:-em}"   # training reward for COLLECT: em (binary exact-match) | f1 (token-F1 partial credit, fixes EM's brevity over-optimization). eval headline stays EM either way.
PROC_BETA="${PROC_BETA:-0}"   # dense PROCESS reward weight (retrieval-hit + query-efficiency); >0 => reward = outcome + beta*process. Applied to COLLECT only.
KL_COEF="${KL_COEF:-0}"        # KL-to-base anchor weight in the GRPO update; >0 resists over-optimization drift.
LATA="${LATA:-0}"              # 1 => length-aware advantage normalization (advantage /= sqrt(#assistant tokens)).
LATA_FLAG=""; [ "$LATA" = "1" ] && LATA_FLAG="--lata"
if [ "$REWARD" = "em" ]; then GRPO_REWARD=binary; else GRPO_REWARD=continuous; fi
# any process shaping makes the reward continuous -> the trainer must gate on continuous variance
if awk "BEGIN{exit !($PROC_BETA > 0)}" 2>/dev/null; then GRPO_REWARD=continuous; fi
MAX_TOKENS="${MAX_TOKENS:-512}"; MAX_SEARCHES="${MAX_SEARCHES:-3}"; TOP_K="${TOP_K:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"; GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
PORT=8000; HOST=127.0.0.1; export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
OUT="outputs/${RUN}"; mkdir -p "$OUT" outputs/vllm_logs; VP=""

serve() {
  pkill -9 -f vllm.entrypoints 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 3
  local nm=(--served-model-name policy)
  [ -n "$1" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$1")
  echo "[search] serve vLLM (adapter='${1:-<base>}')"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" > "outputs/vllm_logs/${RUN}.log" 2>&1 &
  VP=$!
  for _ in $(seq 1 120); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q policy && { echo "[search] vLLM ready"; return 0; }; sleep 5; done
  echo "[search] vLLM FAILED"; tail -n 25 "outputs/vllm_logs/${RUN}.log"; exit 4
}
stop() { [ -n "$VP" ] && kill "$VP" 2>/dev/null || true; [ -n "$VP" ] && wait "$VP" 2>/dev/null || true; VP=""; sleep 3; }
trap stop EXIT
SA="$PYBIN scripts/grpo/search_agent.py --config $CONFIG --max-searches $MAX_SEARCHES --top-k $TOP_K --max-tokens $MAX_TOKENS --max-concurrency $MAX_CONCURRENCY"

echo "######## BASE eval $(date +%H:%M:%S) ########"
serve ""
$SA --mode eval --split "$SPLIT" --n-questions "$N_EVAL_Q" --out "$OUT/base_eval.jsonl"

CUR=""
for ((it=1; it<=ITERS; it++)); do
  echo "######## ITER $it/$ITERS collect (policy='${CUR:-<base>}') $(date +%H:%M:%S) ########"
  [ "$it" -gt 1 ] && serve "$CUR"
  $SA --mode collect --split train --n-questions "$N_TRAIN_Q" --num-trials "$N" --temperature "$TEMP" --reward-mode "$REWARD" --process-beta "$PROC_BETA" --out "$OUT/rollouts_iter${it}.jsonl"
  stop
  echo "######## ITER $it/$ITERS GRPO update $(date +%H:%M:%S) ########"
  in=(); [ -n "$CUR" ] && in=(--adapter-in "$CUR")
  $PYBIN scripts/grpo/grpo_update.py --rollouts "$OUT/rollouts_iter${it}.jsonl" --base-model "$MODEL" \
    "${in[@]}" --out-adapter "$OUT/adapter_iter${it}" --reward-mode "$GRPO_REWARD" --gate --lr "$LR" \
    --batch-size "$BATCH" --max-seq-len "$GRPO_SEQ" --progress-every "$PROG" --kl-coef "$KL_COEF" $LATA_FLAG
  CUR="$OUT/adapter_iter${it}"
done

echo "######## FINAL adapter eval $(date +%H:%M:%S) ########"
serve "$CUR"
$SA --mode eval --split "$SPLIT" --n-questions "$N_EVAL_Q" --out "$OUT/adapter_eval.jsonl"
stop
echo "######## COMPARE ########"
$PYBIN scripts/grpo/summarize_eval.py --eval "$OUT/adapter_eval.jsonl" --base "$OUT/base_eval.jsonl" --live-pool ""
echo "ALLDONE_SEARCH RUN=$RUN"
