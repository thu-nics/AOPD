#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/tau_bench/qualification}"
WORKERS="${WORKERS:-8}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/.cache/tau2-bench-17e07b1}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
if [[ ! -d "$TAU2_DATA_DIR/tau2/domains" ]]; then
    echo "ERROR: Tau data not found at $TAU2_DATA_DIR; run install_tau2.sh first" >&2
    exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/qualify_expert.py" \
    --output-dir "$OUTPUT_DIR" \
    --model "${EXPERT_MODEL:-deepseek/deepseek-v4-flash}" \
    --user-llm "${USER_LLM:-openrouter/qwen/qwen3.6-27b}" \
    --trials 4 \
    --oracle-samples 3 \
    --max-steps "${MAX_STEPS:-30}" \
    --workers "$WORKERS" \
    "$@"
