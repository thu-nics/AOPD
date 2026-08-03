#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/paths.sh"
MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_context_selection}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_deterministic_filter}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
CONCURRENCY="${CONCURRENCY:-12}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"
MAX_JUDGE_TASKS="${MAX_JUDGE_TASKS:-1000}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16384}"
SKIP_JUDGE="${SKIP_JUDGE:-1}"
RESUME="${RESUME:-auto}"

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ "$SKIP_JUDGE" != "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required unless SKIP_JUDGE=1" >&2
    exit 1
fi
if ! "$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    exit 1
fi
for path in "$SELECTION_DIR/awm_context_candidates.parquet" "$SELECTION_DIR/candidate_manifest.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing selection artifact $path; run run_selection.sh first" >&2
        exit 1
    fi
done

resume_args=()
if [[ "$RESUME" == "1" || ( "$RESUME" == "auto" && -f "$OUTPUT_DIR/config.json" ) ]]; then
    resume_args+=(--resume)
elif [[ "$RESUME" != "0" && "$RESUME" != "auto" ]]; then
    echo "ERROR: RESUME must be auto, 0, or 1" >&2
    exit 1
fi
judge_args=()
if [[ "$SKIP_JUDGE" == "1" ]]; then
    judge_args+=(--skip-judge)
elif [[ "$SKIP_JUDGE" != "0" ]]; then
    echo "ERROR: SKIP_JUDGE must be 0 or 1" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" "$SCRIPT_DIR/../cli/audit_integrity.py" \
    --data "$SELECTION_DIR/awm_context_candidates.parquet" \
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
    --awm-data-dir "$AWM_DATA_DIR" \
    --tokenizer "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --awm-base-url "$AWM_BASE_URL" \
    --model "$JUDGE_MODEL" --api-base "$DEEPSEEK_API_BASE" \
    --concurrency "$CONCURRENCY" --judge-concurrency "$JUDGE_CONCURRENCY" \
    --max-judge-tasks "$MAX_JUDGE_TASKS" --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    "${resume_args[@]}" "${judge_args[@]}" "$@"
