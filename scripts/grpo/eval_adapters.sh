#!/usr/bin/env bash
set -uo pipefail
# Evaluate EVERY saved adapter_iter* checkpoint of a search-agent run on the held-out
# split, then print the full EM/F1/behavior table (analyze_search_eval.py). Auto-globs
# checkpoints so it works for any ITERS (4, 6, ...). Eval reward stays EM and rows store
# `gold` so F1 is recoverable. Reuses run_search_agent.sh's serve pattern.
#   RUN=search_7b_f1 POLICY_MODEL=.../qwen25-7b-instruct WORKDIR=/root/autodl-tmp/yuyu \
#     PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin \
#     bash scripts/grpo/eval_adapters.sh
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
RUN="${RUN:-search_7b}"; SPLIT="${SPLIT:-validation}"; CONFIG="${CONFIG:-distractor}"
N_EVAL_Q="${N_EVAL_Q:-300}"; MAX_TOKENS="${MAX_TOKENS:-512}"; MAX_SEARCHES="${MAX_SEARCHES:-3}"
TOP_K="${TOP_K:-3}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"; GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
PORT=8000; HOST=127.0.0.1; export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
OUT="outputs/${RUN}"; mkdir -p outputs/vllm_logs

serve() {  # $1 = adapter path
  pkill -9 -f vllm.entrypoints 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 4
  VLLM_USE_FLASHINFER_SAMPLER=0 $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 --enforce-eager --no-enable-flashinfer-autotune \
    --served-model-name basemodel --enable-lora --lora-modules "policy=$1" \
    > "outputs/vllm_logs/eval_adapters_${RUN}.log" 2>&1 &
  for _ in $(seq 1 120); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q policy && return 0; sleep 5; done
  echo "[eval] vLLM FAILED"; tail -n 20 "outputs/vllm_logs/eval_adapters_${RUN}.log"; return 1
}
stop() { pkill -9 -f vllm.entrypoints 2>/dev/null || true; for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done; sleep 4; }
trap stop EXIT
SA="$PYBIN scripts/grpo/search_agent.py --config $CONFIG --max-searches $MAX_SEARCHES --top-k $TOP_K --max-tokens $MAX_TOKENS --max-concurrency $MAX_CONCURRENCY"

shopt -s nullglob
ADAPTERS=("$OUT"/adapter_iter*/)
if [ ${#ADAPTERS[@]} -eq 0 ]; then echo "[eval] no $OUT/adapter_iter* found"; exit 3; fi
echo "[eval] found ${#ADAPTERS[@]} checkpoints: ${ADAPTERS[*]}"

EVAL_ARGS=()
for ad in "${ADAPTERS[@]}"; do
  ad="${ad%/}"; it="$(basename "$ad" | sed 's/adapter_//')"   # e.g. iter3
  ev="$OUT/${it}_eval.jsonl"
  if [ -f "$ev" ]; then echo "[eval] $it already evaluated -> $ev (skip)"; else
    echo "######## EVAL $it $(date +%H:%M:%S) ########"
    serve "$ad" || exit 4
    $SA --mode eval --split "$SPLIT" --n-questions "$N_EVAL_Q" --out "$ev"
    stop
  fi
  EVAL_ARGS+=("${it}=${ev}")
done

echo "######## FULL CURVE (EM + F1 + behavior) ########"
$PYBIN scripts/grpo/analyze_search_eval.py --base "$OUT/base_eval.jsonl" --evals "${EVAL_ARGS[@]}"
echo "ALLDONE_EVAL_ADAPTERS RUN=$RUN"
