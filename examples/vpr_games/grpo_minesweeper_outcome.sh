#!/bin/bash
# ============================================================================
# Minesweeper —— 标准 GRPO + outcome（结果）奖励 训练脚本
#
#   * algorithm.adv_estimator=grpo  → verl 原生 compute_grpo_outcome_advantage（组内归一化）。
#   * env.minesweeper.reward_mode=outcome → 纯结果奖励：合法非终止步 0；揭开所有安全格
#     （通关）+1；踩雷/非法动作终止 -1；超时/步数耗尽 0。
#     注意：通关判定 = 揭开全部安全格（覆盖 GEM 默认的 flag-all 判定）。
#   * 无对手；每个 episode 一局 5x5、5 雷的扫雷（20 个安全格）。
#   * KL 正则：actor.use_kl_loss=True, kl_loss_coef=0.001（可用 USE_KL=False 关闭）。
#
# 本次配置（可用同名环境变量覆盖）：
#   * 开启 thinking 模式（enable_thinking=True），最大输出长度 4096。
#   * 100 个 step，每个 step 8x16=128 条 rollout 轨迹（TRAIN_BATCH x ROLLOUT_N）。
#   * 每次验证 64 条轨迹（VAL_BATCH）。
#   * 所有训练产物（hydra 日志 / checkpoint / tensorboard / 控制台日志）放到 ./runs/<timestamp>。
#
# 依赖：需要 GEM（pip install 'git+https://github.com/axon-rl/gem.git'）。
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
VAL_BATCH="${VAL_BATCH:-64}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
MAX_RESP="${MAX_RESP:-4096}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
USE_KL="${USE_KL:-True}"
KL_COEF="${KL_COEF:-0.001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/vpr_minesweeper"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Minesweeper | standard GRPO + outcome reward ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Run dir:      $RUN_DIR"

if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: Model not found at $MODEL_PATH" >&2; exit 1; fi
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found at $PYTHON" >&2; exit 1; fi
if ! "$PYTHON" -c "import gem" 2>/dev/null; then
    echo "ERROR: 'gem' not found in $PYTHON (pip install 'git+https://github.com/axon-rl/gem.git')" >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
    --env-name vpr_minesweeper --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
TENSORBOARD_DIR="$RUN_DIR/tensorboard" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_minesweeper \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length=2048 \
    data.max_response_length="$MAX_RESP" \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len=8192 \
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
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=vpr_minesweeper \
    trainer.experiment_name="grpo_outcome_${TS}" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.logger=["console","tensorboard"] \
    trainer.resume_mode=disable \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus=32 2>&1 | tee "$LOG_FILE"

echo ""
echo "All outputs under: $RUN_DIR  (train.log / ckpt/ / tensorboard/ / hydra/)"
echo "  tensorboard --logdir $RUN_DIR/tensorboard"
echo "=== done ==="
