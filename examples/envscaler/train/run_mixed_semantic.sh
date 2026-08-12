#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export ENABLE_ENVSCALER=1
export TRAIN_STEPS="${TRAIN_STEPS:-200}"
export TRAIN_BATCH="${TRAIN_BATCH:-64}"
export AWM_PER_STEP="${AWM_PER_STEP:-48}"
export ENVSCALER_PER_STEP="${ENVSCALER_PER_STEP:-16}"
export VAL_BATCH="${VAL_BATCH:-16}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export SAVE_BEFORE_VALIDATION="${SAVE_BEFORE_VALIDATION:-true}"
export SHUFFLE=false
export RESUME_MODE="${RESUME_MODE:-disable}"
export ENVSCALER_ROOT="${ENVSCALER_ROOT:-/mnt/public2/yuanhuining/repos/EnvScaler}"
export ENVSCALER_POOL="${ENVSCALER_POOL:-$REPO_ROOT/runs/envscaler_filter/envscaler_training_pool.parquet}"
export ENVSCALER_MANIFEST="${ENVSCALER_MANIFEST:-$REPO_ROOT/runs/envscaler_filter/health_manifest.json}"

exec bash "$REPO_ROOT/examples/awm/train/run_semantic.sh" "$@"
