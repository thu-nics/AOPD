#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORACLE_PROVIDER="${ORACLE_PROVIDER:-dashscope}"
export ORACLE_MODEL="${ORACLE_MODEL:-qwen3.6-flash}"
export ORACLE_API_BASE="${ORACLE_API_BASE:-${DASHSCOPE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
export ORACLE_API_KEY_ENV="${ORACLE_API_KEY_ENV:-DASHSCOPE_API_KEY}"
export ORACLE_ENABLE_THINKING="${ORACLE_ENABLE_THINKING:-true}"
export ORACLE_REASONING_EFFORT="${ORACLE_REASONING_EFFORT:-null}"
export ORACLE_THINKING_BUDGET="${ORACLE_THINKING_BUDGET:-4096}"
export ORACLE_TEMPERATURE="${ORACLE_TEMPERATURE:-0.6}"
export ORACLE_TOP_P="${ORACLE_TOP_P:-0.95}"
export ORACLE_PRESENCE_PENALTY="${ORACLE_PRESENCE_PENALTY:-null}"
export ORACLE_MAX_TOKENS="${ORACLE_MAX_TOKENS:-8192}"
exec bash "$SCRIPT_DIR/run_mixed_agentic_opd.sh" "$@"
