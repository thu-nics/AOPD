#!/bin/bash
# ============================================================================
# TicTacToe —— 标准 GRPO + outcome（episode 级）奖励 训练脚本
#
# 与 VPR 的区别：
#   * algorithm.adv_estimator=grpo  → 用标准 GRPO 的“组内归一化 outcome advantage”
#     （compute_grpo_outcome_advantage），而不是 VPR 的逐-turn 过程 advantage。
#   * env.tictactoe.reward_mode=outcome  → 纯终局胜负奖励：合法非终止步奖励 0，终止步
#     +1（agent 赢）/ -1（输，含对手获胜及非法动作终止）/ 0（平局或步数耗尽）。
#     于是每条轨迹的 episode 级标量（EpisodeRewardManager 放最后一个 token）= 该局胜负，
#     GRPO 对同一 prompt 组（group_n = rollout.n 个共享同一初始局面的副本）做 (r-mean)/std。
#   * 对手：random（已内置；agent 落子后对手随机落子，对手连成线则判负）。
#   * 注意：reward_mode 默认是 "oracle"（VPR 用的稠密逐步奖励）；本脚本显式覆盖为
#     "outcome" 以得到“按胜负给分 + 标准 GRPO”的严格 baseline。
#
# 可用环境变量覆盖：MODEL_PATH / PYTHON / TRAIN_STEPS / TRAIN_BATCH / ROLLOUT_N /
#                  SAVE_FREQ / GPU_MEM_UTIL
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"      # 训练步数（smoke 验证用 2；正式训练自行调大）
TRAIN_BATCH="${TRAIN_BATCH:-2}"       # prompt 数；环境/actor 数 = TRAIN_BATCH * ROLLOUT_N
ROLLOUT_N="${ROLLOUT_N:-4}"           # GRPO 组大小（同 prompt 的副本数，需 >=2）
SAVE_FREQ="${SAVE_FREQ:--1}"          # >0 时按步保存 checkpoint
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"   # vLLM 显存占比；OOM 时调小

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/vpr_tictactoe"
LOG_DIR="$SCRIPT_DIR/smoke_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/tictactoe_grpo_outcome_${TS}.log"

echo "=== TicTacToe | standard GRPO + outcome reward ==="
echo "Model:        $MODEL_PATH"
echo "Train steps:  $TRAIN_STEPS | batch: $TRAIN_BATCH | group(rollout.n): $ROLLOUT_N"
echo "Log:          $LOG_FILE"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH" >&2
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preparing data..."
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
        --env-name vpr_tictactoe --train-size "$TRAIN_BATCH" --val-size 1 \
        --output-dir "$DATA_DIR"
fi

VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_tictactoe \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=64 \
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
    env.tictactoe.reward_mode=outcome \
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
