#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_context_selection}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_deterministic_filter}"
MIGRATE_FROM="${MIGRATE_FROM:-}"
CONCURRENCY="${CONCURRENCY:-12}"
RESUME="${RESUME:-auto}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON" >&2
    exit 1
fi
for path in "$SELECTION_DIR/awm_context_candidates.parquet" "$SELECTION_DIR/candidate_manifest.json"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing selection artifact $path; run run_selection.sh first" >&2
        exit 1
    fi
done

cd "$REPO_ROOT"
if [[ -n "$MIGRATE_FROM" ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/audit_integrity.py" \
        --data "$SELECTION_DIR/awm_context_candidates.parquet" \
        --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
        --output-dir "$OUTPUT_DIR" \
        --migrate-from "$MIGRATE_FROM" "$@"
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
exec "$PYTHON" "$SCRIPT_DIR/audit_integrity.py" \
    --data "$SELECTION_DIR/awm_context_candidates.parquet" \
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
    --awm-data-dir "$AWM_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --awm-base-url "$AWM_BASE_URL" \
    --concurrency "$CONCURRENCY" \
    "${resume_args[@]}" "$@"
