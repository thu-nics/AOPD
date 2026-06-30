#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_BATCH="${TRAIN_BATCH:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
export STATE_GROUP_ADV_MODE="${STATE_GROUP_ADV_MODE:-mean_then_batch_whiten}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPUS="${N_GPUS:-8}"
export TP_SIZE="${TP_SIZE:-4}"

bash "$SCRIPT_DIR/vpr_sokoban.sh"
