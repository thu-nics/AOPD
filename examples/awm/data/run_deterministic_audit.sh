#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
SELECTION_DIR="${SELECTION_DIR:-$REPO_ROOT/runs/awm_data_processing/01_context_selection}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/awm_data_processing/02_deterministic_audit}"


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
exec "$PYTHON" "$SCRIPT_DIR/audit_deterministic_health.py" \
    --data "$SELECTION_DIR/awm_context_candidates.parquet" \
    --candidate-manifest "$SELECTION_DIR/candidate_manifest.json" \
    --awm-data-dir "$AWM_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" "$@"
