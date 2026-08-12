#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/opt/venvs/verl-agent/bin/python}"
SOURCE_ROOT="${ENVSCALER_ROOT:-/mnt/public2/yuanhuining/repos/EnvScaler}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/envscaler_filter}"
DETERMINISTIC_DIR="${DETERMINISTIC_DIR:-$OUTPUT_DIR/deterministic}"
MODEL="${MODEL:-deepseek-v4-flash}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"

deterministic_manifest="$DETERMINISTIC_DIR/deterministic_manifest.json"
deterministic_audit="$DETERMINISTIC_DIR/task_audit.jsonl"
if [[ -f "$deterministic_manifest" && -f "$deterministic_audit" ]]; then
  echo "Reusing deterministic EnvScaler audit: $DETERMINISTIC_DIR"
elif [[ -e "$deterministic_manifest" || -e "$deterministic_audit" ]]; then
  echo "ERROR: incomplete deterministic EnvScaler artifacts in $DETERMINISTIC_DIR" >&2
  exit 1
else
  "$PYTHON" examples/envscaler/filter/run_deterministic.py \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$DETERMINISTIC_DIR"
fi

args=(
  --source-root "$SOURCE_ROOT"
  --deterministic-dir "$DETERMINISTIC_DIR"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --concurrency "$CONCURRENCY"
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
"$PYTHON" examples/envscaler/filter/run_screening.py "${args[@]}"
