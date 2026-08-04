#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_STEPS="${TRAIN_STEPS:-200}"
export TRAIN_BATCH="${TRAIN_BATCH:-64}"
export VAL_BATCH="${VAL_BATCH:-16}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export SHUFFLE="${SHUFFLE:-false}"
export N_GPUS="${N_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export SP_SIZE="${SP_SIZE:-2}"
export MAX_CKPTS="${MAX_CKPTS:-null}"
export SAVE_BEFORE_VALIDATION="${SAVE_BEFORE_VALIDATION:-true}"
export TAU_VAL_DOMAINS="${TAU_VAL_DOMAINS:-airline}"

VARIANT=semantic exec bash "$SCRIPT_DIR/run_training.sh" "$@"
