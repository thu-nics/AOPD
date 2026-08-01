#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/paths.sh"
MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-4B}"
API_BASE="${API_BASE:-http://127.0.0.1:${VLLM_PORT:-8001}/v1}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/awm}"
SPLIT="${SPLIT:-smoke}"
DATA_FILE="${DATA_FILE:-$DATA_DIR/awm_${SPLIT}.parquet}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_eval_${SPLIT}_$(date -u +%Y%m%dT%H%M%S)}"
CONCURRENCY="${CONCURRENCY:-8}"
START_VLLM="${START_VLLM:-1}"
VLLM_PORT="${VLLM_PORT:-8001}"
SEED="${SEED:-300}"
TP_SIZE="${TP_SIZE:-2}"

vllm_pid=""
cleanup() {
    if [[ -n "$vllm_pid" ]]; then
        kill "$vllm_pid" 2>/dev/null || true
        wait "$vllm_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if ! "$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    echo "Start it with examples/awm/scripts/start_server.sh to enable pinned logical time." >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/../cli/prepare_data.py" \
    --data-dir "$AWM_DATA_DIR" \
    --output-dir "$DATA_DIR" \
    --local-files-only \
    --verify-only

if [[ "$START_VLLM" == "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len 32000 \
        --max-num-batched-tokens 32000 \
        --gpu-memory-utilization 0.9 >"$OUTPUT_DIR/vllm.log" 2>&1 &
    vllm_pid=$!
    ready=0
    for _ in $(seq 1 120); do
        if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$VLLM_PORT/health', timeout=2)" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "ERROR: persistent vLLM server exited; see $OUTPUT_DIR/vllm.log" >&2
            exit 1
        fi
        sleep 2
    done
    if [[ "$ready" != "1" ]]; then
        echo "ERROR: persistent vLLM server did not become healthy; see $OUTPUT_DIR/vllm.log" >&2
        exit 1
    fi
fi

selection_args=()
if [[ -n "$SELECTION_MANIFEST" ]]; then
    selection_args+=(--selection-manifest "$SELECTION_MANIFEST")
fi

"$PYTHON" "$SCRIPT_DIR/../cli/eval_awm.py" \
    --data "$DATA_FILE" \
    --manifest "$DATA_DIR/manifest.json" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --api-base "$API_BASE" \
    --awm-base-url "$AWM_BASE_URL" \
    --concurrency "$CONCURRENCY" \
    --seed "$SEED" \
    "${selection_args[@]}" "$@"
