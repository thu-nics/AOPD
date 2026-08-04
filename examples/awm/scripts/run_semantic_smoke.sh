#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SMOKE=1
export TAU_VAL_DOMAINS="${TAU_VAL_DOMAINS:-airline}"
export TAU_VAL_NUM_TASKS="${TAU_VAL_NUM_TASKS:-2}"
export VAL_BEFORE_TRAIN=true

exec bash "$SCRIPT_DIR/run_semantic.sh" "$@"
