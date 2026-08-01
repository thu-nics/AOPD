#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/paths.sh"
MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_selection_1k}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_expert_qualification}"
EXPERT_MODEL="${EXPERT_MODEL:-deepseek-v4-flash}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-}"
RESUME="${RESUME:-auto}"

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required; launch from tmux session deepseek_api" >&2
    exit 1
fi
if ! "$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    echo "Start it with examples/awm/scripts/start_server.sh to enable pinned logical time." >&2
    exit 1
fi
for path in "$SELECTION_DIR/awm_expert_candidates_1k.parquet" "$SELECTION_DIR/candidate_manifest.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing candidate artifact $path; run run_selection.sh first" >&2
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
limit_args=()
if [[ -n "$MAX_NEW_TASKS" ]]; then
    limit_args+=(--max-new-tasks "$MAX_NEW_TASKS")
fi

cd "$REPO_ROOT"
exec "$PYTHON" "$SCRIPT_DIR/../cli/qualify_expert.py" \
    --data "$SELECTION_DIR/awm_expert_candidates_1k.parquet" \
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
    --tokenizer "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
    --model "$EXPERT_MODEL" --api-base "$DEEPSEEK_API_BASE" \
    --awm-base-url "$AWM_BASE_URL" --concurrency "$CONCURRENCY" \
    "${resume_args[@]}" "${limit_args[@]}" "$@"
