#!/bin/bash
# ============================================================================
# TicTacToe —— 标准 GRPO + outcome（胜负）奖励 训练脚本
#
#   * algorithm.adv_estimator=grpo  → verl 原生 compute_grpo_outcome_advantage（组内归一化）。
#   * env.tictactoe.reward_mode=outcome → 纯终局胜负：合法非终止步 0，终止步 +1（赢）/
#     -1（输，含对手获胜及非法动作终止）/ 0（平局或步数耗尽）。
#   * 角色：policy 控制 player0 = X（先手，env.tictactoe.agent_player=X）；对手执 O。
#   * 对手：random，且每个 step 用新的随机种子（真随机，跨 step 对局各不相同；同一 GRPO
#     组内 16 个副本仍共享同一种子以保证可比，全程可由 env.seed 复现）。
#     可改 env.tictactoe.opponent=mcts（OpenSpiel C++ MCTS，需 pip install open_spiel）。
#   * KL 正则：actor.use_kl_loss=True, kl_loss_coef=0.001（防策略漂移）。
#
# 本次配置（可用同名环境变量覆盖）：
#   * 开启 thinking 模式（enable_thinking=True），最大输出长度 4096。
#   * 100 个 step，每个 step 8x16=128 条 rollout 轨迹（TRAIN_BATCH x ROLLOUT_N）。
#   * 每次验证 64 条轨迹（VAL_BATCH）。
#   * 所有训练产物（hydra 日志 / checkpoint / tensorboard / 控制台日志）放到 ./runs/<timestamp>。
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"        # 训练步数
TRAIN_BATCH="${TRAIN_BATCH:-8}"          # 每 step 的 prompt 数
ROLLOUT_N="${ROLLOUT_N:-16}"            # GRPO 组大小；每 step 轨迹数 = TRAIN_BATCH x ROLLOUT_N = 128
VAL_BATCH="${VAL_BATCH:-64}"            # 每次验证的轨迹数（val group_n=1）
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"  # 每次梯度更新的 mini-batch（需能整除 128）
MAX_RESP="${MAX_RESP:-4096}"           # 最大输出长度（thinking 模式需要较长预算）
SAVE_FREQ="${SAVE_FREQ:-20}"           # 每多少 step 保存一次 checkpoint
TEST_FREQ="${TEST_FREQ:-20}"           # 每多少 step 验证一次
ENABLE_THINKING="${ENABLE_THINKING:-True}"
USE_KL="${USE_KL:-True}"               # KL 正则开关（防策略漂移）
KL_COEF="${KL_COEF:-0.001}"            # KL loss 系数
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"    # vLLM 显存占比（free_cache_engine=True 训练时会释放）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/vpr_tictactoe"
TS="$(date +%Y%m%dT%H%M%S)"
# 所有产物根目录（默认 当前工作目录下 ./runs/<timestamp>，取绝对路径以兼容 Hydra 改目录）
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== TicTacToe | standard GRPO + outcome reward ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Run dir:      $RUN_DIR  (train.log / ckpt/ / tensorboard/ / hydra config)"

if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: Model not found at $MODEL_PATH" >&2; exit 1; fi
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found at $PYTHON" >&2; exit 1; fi

# 重新生成数据，保证 train>=TRAIN_BATCH、val>=VAL_BATCH 条
"$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
    --env-name vpr_tictactoe --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
TENSORBOARD_DIR="$RUN_DIR/tensorboard" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_tictactoe \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length=1024 \
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
    env.tictactoe.agent_player=X \
    env.tictactoe.opponent=random \
    env.tictactoe.reward_mode=outcome \
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
    trainer.project_name=vpr_tictactoe \
    trainer.experiment_name="grpo_outcome_${TS}" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.logger=["console","tensorboard"] \
    trainer.resume_mode=disable \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus=32 2>&1 | tee "$LOG_FILE"

echo ""
echo "All outputs under: $RUN_DIR"
echo "  - train.log         (控制台日志)"
echo "  - ckpt/             (checkpoint，每 ${SAVE_FREQ} step)"
echo "  - tensorboard/      (tensorboard：tensorboard --logdir $RUN_DIR/tensorboard)"
echo "  - hydra/            (Hydra 配置快照)"
echo "=== done ==="
