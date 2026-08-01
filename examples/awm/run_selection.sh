#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/opt/venvs/verl-agent-sokoban/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
AWM_DATA_DIR="${AWM_DATA_DIR:-$HOME/.cache/openenv/awm}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/awm}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_selection_1k}"
CONCURRENCY="${CONCURRENCY:-12}"
SCAFFOLD_CUTOFF="${SCAFFOLD_CUTOFF:-16000}"
TARGET_TASKS="${TARGET_TASKS:-1000}"
RESUME="${RESUME:-auto}"

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if ! "$PYTHON" "$SCRIPT_DIR/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    echo "Start it with examples/awm/start_server.sh to enable pinned logical time." >&2
    exit 1
fi
mkdir -p "$DATA_DIR"
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
        --data-dir "$AWM_DATA_DIR" --output-dir "$DATA_DIR" --local-files-only
fi
"$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
    --data-dir "$AWM_DATA_DIR" --output-dir "$DATA_DIR" \
    --local-files-only --verify-only

resume_args=()
if [[ "$RESUME" == "1" || ( "$RESUME" == "auto" && -f "$OUTPUT_DIR/config.json" ) ]]; then
    resume_args+=(--resume)
elif [[ "$RESUME" != "0" && "$RESUME" != "auto" ]]; then
    echo "ERROR: RESUME must be auto, 0, or 1" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" "$SCRIPT_DIR/select_tasks.py" \
    --data "$DATA_DIR/awm_all.parquet" \
    --manifest "$DATA_DIR/manifest.json" \
    --tokenizer "$MODEL_PATH" \
    --awm-base-url "$AWM_BASE_URL" \
    --output-dir "$OUTPUT_DIR" \
    --cutoff "$SCAFFOLD_CUTOFF" \
    --target "$TARGET_TASKS" \
    --concurrency "$CONCURRENCY" \
    "${resume_args[@]}" "$@"
