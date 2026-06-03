#!/bin/bash
# ============================================================================
# Minesweeper —— 标准 GRPO + outcome（胜负）奖励 训练脚本
#
# 标准 GRPO（algorithm.adv_estimator=grpo，verl 原生 compute_grpo_outcome_advantage，
# 组内归一化）+ 纯结果奖励（env.minesweeper.reward_mode=outcome）：
#   合法非终止步=0；揭开所有安全格（通关）=+1；踩雷/非法动作终止=-1；超时=0。
# reward_mode 默认 oracle（VPR 后验稠密奖励）；本脚本覆盖为 outcome。
#
# 可用环境变量覆盖：MODEL_PATH / PYTHON / TRAIN_STEPS / TRAIN_BATCH / ROLLOUT_N /
#                  SAVE_FREQ / GPU_MEM_UTIL
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
ROLLOUT_N="${ROLLOUT_N:-4}"
SAVE_FREQ="${SAVE_FREQ:--1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/vpr_minesweeper"
LOG_DIR="$SCRIPT_DIR/smoke_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/minesweeper_grpo_outcome_${TS}.log"

echo "=== Minesweeper | standard GRPO + outcome reward ==="
echo "Model: $MODEL_PATH | steps: $TRAIN_STEPS batch: $TRAIN_BATCH group: $ROLLOUT_N"
echo "Log:   $LOG_FILE"

if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: Model not found at $MODEL_PATH" >&2; exit 1; fi
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found at $PYTHON" >&2; exit 1; fi
if ! "$PYTHON" -c "import gem" 2>/dev/null; then
    echo "ERROR: 'gem' not found in $PYTHON (pip install 'git+https://github.com/axon-rl/gem.git')" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py" --env-name vpr_minesweeper \
        --train-size "$TRAIN_BATCH" --val-size 1 --output-dir "$DATA_DIR"
fi

VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_minesweeper \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    env.seed=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.minesweeper.reward_mode=outcome \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.test_freq=5 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.logger=["console"] \
    trainer.resume_mode=disable \
    +ray_init.num_cpus=16 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log preserved at: $LOG_FILE"
echo "=== done ==="
