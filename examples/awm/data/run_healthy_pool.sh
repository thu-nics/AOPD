#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"

SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_context_selection}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_healthy_pool}"
AWM_BASE_URL="${AWM_BASE_URL:-}"
AWM_HOST="${AWM_HOST:-127.0.0.1}"
AWM_PORT="${AWM_PORT:-}"
MANAGE_AWM_SERVER="${MANAGE_AWM_SERVER:-1}"
MODEL="${MODEL:-deepseek-v4-flash}"
API_BASE="${API_BASE:-https://api.deepseek.com}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-12}"
ATTEMPTS="${ATTEMPTS:-3}"
RESUME="${RESUME:-auto}"
EXPERT_TRIALS="${EXPERT_TRIALS-$REPO_ROOT/runs/awm_final_pool/trials.jsonl}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR%/}.server.log}"
SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap stop_server EXIT

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON" >&2
    exit 1
fi
if [[ -z "${!API_KEY_ENV:-}" ]]; then
    echo "ERROR: missing required environment variable $API_KEY_ENV" >&2
    exit 1
fi
for path in \
    "$SELECTION_DIR/awm_context_candidates.parquet" \
    "$SELECTION_DIR/candidate_manifest.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing selection artifact $path; run run_selection.sh first" >&2
        exit 1
    fi
done
if [[ -n "$EXPERT_TRIALS" && ! -f "$EXPERT_TRIALS" ]]; then
    echo "ERROR: configured expert metadata does not exist: $EXPERT_TRIALS" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$MANAGE_AWM_SERVER" == "1" ]]; then
    if [[ -n "$AWM_BASE_URL" ]]; then
        echo "ERROR: AWM_BASE_URL cannot be set with MANAGE_AWM_SERVER=1" >&2
        exit 1
    fi
    AWM_PORT="$("$PYTHON" - "$AWM_HOST" "$AWM_PORT" <<'PY'
import socket
import sys
host = sys.argv[1]
requested = int(sys.argv[2]) if sys.argv[2] else 0
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.bind((host, requested))
    print(sock.getsockname()[1])
PY
)"
    AWM_BASE_URL="http://$AWM_HOST:$AWM_PORT"
    setsid env \
        AWM_HOST="$AWM_HOST" \
        AWM_PORT="$AWM_PORT" \
        AWM_SERVER_RUN_ID="healthy-pool-$$" \
        AWM_TERMINAL_JUDGE_MODEL="$MODEL" \
        AWM_TERMINAL_JUDGE_API_BASE="$API_BASE" \
        AWM_TERMINAL_JUDGE_REASONING_EFFORT=max \
        AWM_TERMINAL_JUDGE_MAX_TOKENS=8192 \
        AWM_TERMINAL_JUDGE_TIMEOUT_SECONDS=300 \
        AWM_TERMINAL_JUDGE_MAX_RETRIES=1 \
        bash "$SCRIPT_DIR/../runtime/start_server.sh" \
        >>"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    for _ in {1..120}; do
        if "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
            --base-url "$AWM_BASE_URL" \
            --data-dir "$AWM_DATA_DIR" \
            --expected-terminal-model "$MODEL" \
            --expected-run-id="healthy-pool-$$" \
            --timeout 1 >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: managed AWM server exited; see $SERVER_LOG" >&2
            exit 1
        fi
        sleep 1
    done
    if ! "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
        --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
        --expected-terminal-model "$MODEL" \
        --expected-run-id="healthy-pool-$$" --timeout 5 >/dev/null; then
        echo "ERROR: managed AWM server did not become healthy; see $SERVER_LOG" >&2
        exit 1
    fi
else
    AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
    "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
        --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
        --expected-terminal-model "$MODEL" >/dev/null
fi

resume_args=()
if [[ "$RESUME" == "1" || ( "$RESUME" == "auto" && -f "$OUTPUT_DIR/config.json" ) ]]; then
    resume_args+=(--resume)
elif [[ "$RESUME" != "0" && "$RESUME" != "auto" ]]; then
    echo "ERROR: RESUME must be auto, 0, or 1" >&2
    exit 1
fi
expert_args=()
if [[ -n "$EXPERT_TRIALS" ]]; then
    expert_args+=(--expert-trials "$EXPERT_TRIALS")
fi

cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/build_healthy_pool.py" \
    --data "$SELECTION_DIR/awm_context_candidates.parquet" \
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
    --awm-data-dir "$AWM_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --awm-base-url "$AWM_BASE_URL" \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --api-key-env "$API_KEY_ENV" \
    --concurrency "$CONCURRENCY" \
    --attempts "$ATTEMPTS" \
    "${expert_args[@]}" \
    "${resume_args[@]}" \
    "$@"
