#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the evaluated model directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
API_BASE="${API_BASE:-http://127.0.0.1:${VLLM_PORT:-8001}/v1}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/awm}"
SPLIT="${SPLIT:-all}"
DATA_FILE="${DATA_FILE:-$DATA_DIR/awm_${SPLIT}.parquet}"
TASK_LIMIT="${TASK_LIMIT:-}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_eval_${SPLIT}_$(date -u +%Y%m%dT%H%M%S)}"
CONCURRENCY="${CONCURRENCY:-8}"
START_VLLM="${START_VLLM:-1}"
VLLM_PORT="${VLLM_PORT:-8001}"
SEED="${SEED:-300}"
TP_SIZE="${TP_SIZE:-2}"
MAX_HISTORY_EXCHANGES="${MAX_HISTORY_EXCHANGES:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-27904}"
VERIFIER_MODE="${VERIFIER_MODE:-sql}"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.deepseek.com}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-DEEPSEEK_API_KEY}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-deepseek}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"

if [[ "$SPLIT" != "all" ]]; then
    echo "ERROR: prepared AWM data now materializes only SPLIT=all; use TASK_LIMIT for a smaller eval." >&2
    exit 1
fi

vllm_pid=""
cleanup() {
    if [[ -n "$vllm_pid" ]]; then
        kill "$vllm_pid" 2>/dev/null || true
        wait "$vllm_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$VERIFIER_MODE" != "sql" && "$VERIFIER_MODE" != "code" ]]; then
    echo "ERROR: VERIFIER_MODE must be sql or code" >&2
    exit 1
fi
if [[ "$VERIFIER_MODE" == "sql" && -z "${!JUDGE_API_KEY_ENV:-}" ]]; then
    echo "ERROR: VERIFIER_MODE=sql requires non-empty $JUDGE_API_KEY_ENV" >&2
    exit 1
fi

server_check_args=(--base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR")
if [[ "$VERIFIER_MODE" == "sql" ]]; then
    server_check_args+=(
        --expected-terminal-provider "$JUDGE_PROVIDER"
        --expected-terminal-model "$JUDGE_MODEL"
    )
fi
if ! "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
    "${server_check_args[@]}" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    echo "Start it with examples/awm/runtime/start_server.sh to enable pinned logical time." >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/../data/prepare_data.py" \
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
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
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
if [[ -n "$MAX_HISTORY_EXCHANGES" ]]; then
    selection_args+=(--max-history-exchanges "$MAX_HISTORY_EXCHANGES")
fi
if [[ -n "$SELECTION_MANIFEST" ]]; then
    selection_args+=(--selection-manifest "$SELECTION_MANIFEST")
fi
if [[ -n "$TASK_LIMIT" ]]; then
    selection_args+=(--limit "$TASK_LIMIT")
fi

"$PYTHON" "$SCRIPT_DIR/eval_awm.py" \
    --data "$DATA_FILE" \
    --manifest "$DATA_DIR/manifest.json" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --api-base "$API_BASE" \
    --awm-base-url "$AWM_BASE_URL" \
    --verifier-mode "$VERIFIER_MODE" \
    --judge-api-base "$JUDGE_API_BASE" \
    --judge-api-key-env "$JUDGE_API_KEY_ENV" \
    --judge-provider "$JUDGE_PROVIDER" \
    --judge-model "$JUDGE_MODEL" \
    --concurrency "$CONCURRENCY" \
    --seed "$SEED" \
    --max-prompt-tokens "$MAX_PROMPT_LENGTH" \
    "${selection_args[@]}" "$@"
