#!/usr/bin/env bash
set -euo pipefail
# DUAL-GPU search-agent RLVR. vLLM stays RESIDENT on GPU $VLLM_GPU for the whole run;
# GRPO training runs on GPU $TRAIN_GPU; each iteration's freshly trained LoRA is HOT-RELOADED
# into the running server (no serve/stop churn — the single-GPU version restarted vLLM every
# iteration). On-policy is preserved: collect_i uses policy_{i-1}, train produces policy_i.
#
# Requires vLLM runtime LoRA updating. Env knobs mirror run_search_agent.sh, plus:
#   VLLM_GPU (default 0)  TRAIN_GPU (default 1)  EVAL_TRIALS (default 1)  EVAL_TEMP (default 0)
# Reuses search_agent.py (--served-model / --api-base) and grpo_update.py UNCHANGED.
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
RUN="${RUN:-search_dual}"; SPLIT="${SPLIT:-validation}"; CONFIG="${CONFIG:-distractor}"
N_TRAIN_Q="${N_TRAIN_Q:-128}"; N_EVAL_Q="${N_EVAL_Q:-300}"
N="${N:-8}"; ITERS="${ITERS:-6}"; LR="${LR:-5e-6}"; TEMP="${TEMP:-1.0}"
BATCH="${BATCH:-4}"; GRPO_SEQ="${GRPO_SEQ:-2560}"; PROG="${PROG:-10}"
REWARD="${REWARD:-f1}"; PROC_BETA="${PROC_BETA:-0}"; KL_COEF="${KL_COEF:-0}"; LATA="${LATA:-0}"
EVAL_TRIALS="${EVAL_TRIALS:-1}"; EVAL_TEMP="${EVAL_TEMP:-0}"
LATA_FLAG=""; [ "$LATA" = "1" ] && LATA_FLAG="--lata"
if [ "$REWARD" = "em" ]; then GRPO_REWARD=binary; else GRPO_REWARD=continuous; fi
if awk "BEGIN{exit !($PROC_BETA > 0)}" 2>/dev/null; then GRPO_REWARD=continuous; fi
MAX_TOKENS="${MAX_TOKENS:-512}"; MAX_SEARCHES="${MAX_SEARCHES:-3}"; TOP_K="${TOP_K:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"; GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
LORA_RANK="${LORA_RANK:-64}"
VLLM_GPU="${VLLM_GPU:-0}"; TRAIN_GPU="${TRAIN_GPU:-1}"
PORT="${PORT:-8000}"; HOST=127.0.0.1; export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
OUT="outputs/${RUN}"; mkdir -p "$OUT" outputs/vllm_logs; VP=""

serve_resident() {
  # SHARED BOX SAFE: never kill other users' GPU procs. Only clean OUR OWN leftover vLLM
  # (matched by this run's PYBIN + port), so a teammate's vllm/.venv on the box is untouched.
  pkill -9 -f "$PYBIN -m vllm.entrypoints.openai.api_server --model $MODEL --host $HOST --port $PORT" 2>/dev/null || true
  sleep 2
  echo "[dual] starting RESIDENT vLLM on GPU $VLLM_GPU port $PORT (served-model=basemodel, LoRA hot-reload ON)"
  CUDA_VISIBLE_DEVICES=$VLLM_GPU VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 --enforce-eager --no-enable-flashinfer-autotune \
    --enable-lora --max-loras 2 --max-lora-rank "$LORA_RANK" --served-model-name basemodel \
    > "outputs/vllm_logs/${RUN}.log" 2>&1 &
  VP=$!
  for _ in $(seq 1 120); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q basemodel && { echo "[dual] vLLM ready"; return 0; }; sleep 5; done
  echo "[dual] vLLM FAILED to start"; tail -n 30 "outputs/vllm_logs/${RUN}.log"; exit 4
}
stop_vllm() { [ -n "$VP" ] && kill "$VP" 2>/dev/null || true; [ -n "$VP" ] && wait "$VP" 2>/dev/null || true; VP=""; }
trap stop_vllm EXIT

load_lora() {  # $1=lora_name  $2=abs_path  — unload same name first so reloads are idempotent
  curl -s -X POST "$OPENAI_API_BASE/unload_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$1\"}" >/dev/null 2>&1 || true
  local resp; resp=$(curl -s -X POST "$OPENAI_API_BASE/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$1\",\"lora_path\":\"$2\"}")
  echo "[dual] load_lora $1 <- $2 : ${resp:0:100}"
  for _ in $(seq 1 12); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q "\"$1\"" && { echo "[dual] LoRA $1 registered"; return 0; }; sleep 2; done
  echo "[dual] ERROR: LoRA $1 did not register (is VLLM_ALLOW_RUNTIME_LORA_UPDATING set? rank<=$LORA_RANK?)"; return 1
}

SA="$PYBIN scripts/grpo/search_agent.py --config $CONFIG --max-searches $MAX_SEARCHES --top-k $TOP_K --max-tokens $MAX_TOKENS --max-concurrency $MAX_CONCURRENCY --api-base $OPENAI_API_BASE"

train_iter() {  # $1=rollouts  $2=adapter_in(may be empty)  $3=out_adapter  — pinned to GPU $TRAIN_GPU
  local in=(); [ -n "$2" ] && in=(--adapter-in "$2")
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PYBIN scripts/grpo/grpo_update.py --rollouts "$1" --base-model "$MODEL" \
    "${in[@]}" --out-adapter "$3" --reward-mode "$GRPO_REWARD" --gate --lr "$LR" \
    --batch-size "$BATCH" --max-seq-len "$GRPO_SEQ" --progress-every "$PROG" --kl-coef "$KL_COEF" $LATA_FLAG
}

serve_resident   # ONCE — stays up for the whole run

echo "######## BASE eval $(date +%H:%M:%S) ########"
$SA --mode eval --served-model basemodel --split "$SPLIT" --n-questions "$N_EVAL_Q" \
    --eval-trials "$EVAL_TRIALS" --eval-temp "$EVAL_TEMP" --out "$OUT/base_eval.jsonl"

CUR=""
for ((it=1; it<=ITERS; it++)); do
  POL="basemodel"; [ "$it" -gt 1 ] && POL="policy_$((it-1))"
  echo "######## ITER $it/$ITERS collect (policy=$POL) $(date +%H:%M:%S) ########"
  $SA --mode collect --served-model "$POL" --split train --n-questions "$N_TRAIN_Q" \
      --num-trials "$N" --temperature "$TEMP" --reward-mode "$REWARD" --process-beta "$PROC_BETA" \
      --out "$OUT/rollouts_iter${it}.jsonl"
  echo "######## ITER $it/$ITERS GRPO update on GPU $TRAIN_GPU $(date +%H:%M:%S) ########"
  train_iter "$OUT/rollouts_iter${it}.jsonl" "$CUR" "$OUT/adapter_iter${it}"
  CUR="$OUT/adapter_iter${it}"
  load_lora "policy_${it}" "$(readlink -f "$CUR")"   # hot-reload into the resident server
  # edit-and-eval inline: the resident server already holds policy_$it, so eval costs no serve/stop
  if [ "${EVAL_EVERY:-1}" = "1" ] || [ "$it" -eq "$ITERS" ]; then
    echo "######## ITER $it/$ITERS held-out eval (resident vLLM) $(date +%H:%M:%S) ########"
    $SA --mode eval --served-model "policy_${it}" --split "$SPLIT" --n-questions "$N_EVAL_Q" \
        --eval-trials "$EVAL_TRIALS" --eval-temp "$EVAL_TEMP" --out "$OUT/iter${it}_eval.jsonl"
  fi
done

cp -f "$OUT/iter${ITERS}_eval.jsonl" "$OUT/adapter_eval.jsonl" 2>/dev/null || true
echo "######## analyze full curve $(date +%H:%M:%S) ########"
EVALS=""; for ((i=1; i<=ITERS; i++)); do [ -f "$OUT/iter${i}_eval.jsonl" ] && EVALS="$EVALS iter${i}=$OUT/iter${i}_eval.jsonl"; done
$PYBIN scripts/grpo/analyze_search_eval.py --base "$OUT/base_eval.jsonl" --evals $EVALS || true
echo "ALLDONE_SEARCH_DUAL RUN=$RUN"
