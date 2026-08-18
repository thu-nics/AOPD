#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_STEPS=1
export TRAIN_BATCH=4
export AWM_PER_STEP=3
export ENVSCALER_PER_STEP=1
export VAL_BATCH="${VAL_BATCH:-2}"
export PPO_MINI_BATCH=4
export SAVE_FREQ=-1
export TEST_FREQ=-1
export VAL_BEFORE_TRAIN=true
export SAVE_BEFORE_VALIDATION=false
export RESUME_MODE=disable

exec bash "$SCRIPT_DIR/run_mixed_agentic_opd.sh" "$@"
