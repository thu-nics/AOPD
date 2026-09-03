#!/usr/bin/env bash
# Diagnostic-only legacy screen. Expert outcome is not a training-pool gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_data_processing/01_context_selection}"
INTEGRITY_DIR="${INTEGRITY_DIR:?INTEGRITY_DIR is required for deprecated expert screening}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required for deprecated expert screening}"
MIGRATE_FROM="${MIGRATE_FROM:-}"
EXPERT_MODEL="${EXPERT_MODEL:-deepseek-v4-flash}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
INFRASTRUCTURE_ATTEMPTS="${INFRASTRUCTURE_ATTEMPTS:-3}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-}"
MAX_NEW_TASK_FRACTION="${MAX_NEW_TASK_FRACTION:-}"
RESUME="${RESUME:-auto}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON" >&2
    exit 1
fi
for path in \
    "$SELECTION_DIR/candidate_manifest.json" \
    "$INTEGRITY_DIR/awm_training_pool.parquet" \
    "$INTEGRITY_DIR/integrity_manifest.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing screening input $path" >&2
        exit 1
    fi
done
base_args=(
    --data "$INTEGRITY_DIR/awm_training_pool.parquet"
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json"
    --integrity-manifest "$INTEGRITY_DIR/integrity_manifest.json"
    --output-dir "$OUTPUT_DIR"
)
cd "$REPO_ROOT"
if [[ -n "$MIGRATE_FROM" ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/expert_screening.py" "${base_args[@]}" \
        --migrate-from "$MIGRATE_FROM" "$@"
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required" >&2
    exit 1
fi
if [[ -n "$MAX_NEW_TASKS" && -n "$MAX_NEW_TASK_FRACTION" ]]; then
    echo "ERROR: MAX_NEW_TASKS and MAX_NEW_TASK_FRACTION are mutually exclusive" >&2
    exit 1
fi
if ! "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    exit 1
fi
resume_args=()
if [[ "$RESUME" == "1" || ( "$RESUME" == "auto" && -f "$OUTPUT_DIR/config.json" ) ]]; then
    resume_args+=(--resume)
elif [[ "$RESUME" != "0" && "$RESUME" != "auto" ]]; then
    echo "ERROR: RESUME must be auto, 0, or 1" >&2
    exit 1
fi
limit_args=()
if [[ -n "$MAX_NEW_TASKS" ]]; then
    limit_args+=(--max-new-tasks "$MAX_NEW_TASKS")
elif [[ -n "$MAX_NEW_TASK_FRACTION" ]]; then
    limit_args+=(--max-new-task-fraction "$MAX_NEW_TASK_FRACTION")
fi
exec "$PYTHON" "$SCRIPT_DIR/expert_screening.py" "${base_args[@]}" \
    --tokenizer "$MODEL_PATH" \
    --model "$EXPERT_MODEL" --api-key-env DEEPSEEK_API_KEY \
    --api-base "$DEEPSEEK_API_BASE" --awm-base-url "$AWM_BASE_URL" \
    --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
    --infrastructure-attempts "$INFRASTRUCTURE_ATTEMPTS" \
    "${resume_args[@]}" "${limit_args[@]}" "$@"
