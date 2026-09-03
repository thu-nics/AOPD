#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON="${PYTHON:-/opt/venvs/verl-agent/bin/python}"
SOURCE_ROOT="${ENVSCALER_ROOT:-/mnt/public2/yuanhuining/repos/EnvScaler}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/envscaler_data_processing/02_static_feasibility_judge}"
DETERMINISTIC_DIR="${DETERMINISTIC_DIR:-$REPO_ROOT/runs/envscaler_data_processing/01_deterministic_audit}"
MODEL="${MODEL:-deepseek-v4-flash}"
API_BASE="${API_BASE:-https://api.deepseek.com}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-16}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-auto}"

deterministic_manifest="$DETERMINISTIC_DIR/deterministic_manifest.json"
if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "ERROR: missing required environment variable $API_KEY_ENV" >&2
  exit 1
fi

deterministic_audit="$DETERMINISTIC_DIR/task_audit.jsonl"
if [[ -f "$deterministic_manifest" && -f "$deterministic_audit" ]]; then
  echo "Reusing deterministic EnvScaler audit: $DETERMINISTIC_DIR"
elif [[ -e "$deterministic_manifest" || -e "$deterministic_audit" ]]; then
  echo "ERROR: incomplete deterministic EnvScaler artifacts in $DETERMINISTIC_DIR" >&2
  exit 1
else
  "$PYTHON" "$SCRIPT_DIR/audit_deterministic_health.py" \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$DETERMINISTIC_DIR"
fi

args=(
  --source-root "$SOURCE_ROOT"
  --deterministic-dir "$DETERMINISTIC_DIR"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --concurrency "$CONCURRENCY"
  --api-base "$API_BASE"
  --api-key-env "$API_KEY_ENV"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --max-retries "$MAX_RETRIES"
  --max-tokens "$MAX_TOKENS"
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" || ( "$RESUME" == "auto" && -f "$OUTPUT_DIR/config.json" ) ]]; then
  args+=(--resume)
elif [[ "$RESUME" != "0" && "$RESUME" != "auto" ]]; then
  echo "ERROR: RESUME must be auto, 0, or 1" >&2
  exit 1
fi
"$PYTHON" "$SCRIPT_DIR/build_static_feasibility_pool.py" "${args[@]}"
