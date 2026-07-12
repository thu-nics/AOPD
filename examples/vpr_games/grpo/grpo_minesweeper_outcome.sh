#!/bin/bash
# ============================================================================
# Minesweeper —— 标准 GRPO + outcome（结果）奖励 训练脚本
#
#   * algorithm.adv_estimator=grpo  → verl 原生 compute_grpo_outcome_advantage（组内归一化）。
#   * env.minesweeper.reward_mode=outcome → 纯结果奖励：合法非终止步 0；揭开所有安全格
#     （通关）+1；踩雷/非法动作终止 -1；超时/步数耗尽 0。
#     注意：通关判定 = 揭开全部安全格（覆盖 GEM 默认的 flag-all 判定）。
#   * 无对手；扫雷棋盘规格由 env.minesweeper 配置控制。
#   * KL 正则：由 USE_KL 和 KL_COEF 控制。
#
# 本次配置（可用同名环境变量覆盖）：
#   * thinking 模式、输出长度、训练步数、rollout 规模、验证规模均由下方环境变量控制。
#   * 训练产物写入 RUN_DIR 下的日志、checkpoint、tensorboard 和 Hydra 配置目录。
#
# 依赖：需要 GEM（pip install 'git+https://github.com/axon-rl/gem.git'）。
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"       # 训练步数
TRAIN_BATCH="${TRAIN_BATCH:-32}"         # 每个训练 step 的 prompt 数
ROLLOUT_N="${ROLLOUT_N:-8}"            # GRPO 组大小；每 step 轨迹数 = TRAIN_BATCH x ROLLOUT_N
VAL_BATCH="${VAL_BATCH:-128}"            # 每次验证的轨迹数
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"  # PPO 更新使用的 mini-batch
MAX_RESP="${MAX_RESP:-4096}"           # 生成响应的最大 token 长度
SAVE_FREQ="${SAVE_FREQ:-25}"           # checkpoint 保存间隔
RESUME_MODE="${RESUME_MODE:-disable}"     # disable/auto/resume_path
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"  # RESUME_MODE=resume_path 时指定 global_step_* 目录
TEST_FREQ="${TEST_FREQ:-20}"           # 验证间隔
ENABLE_THINKING="${ENABLE_THINKING:-True}"  # Qwen chat template thinking 开关
USE_KL="${USE_KL:-True}"               # actor KL loss 开关
KL_COEF="${KL_COEF:-0.001}"            # actor KL loss 系数
PPO_MICRO="${PPO_MICRO:-2}"            # actor 训练 micro-batch
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"    # rollout/ref log-prob micro-batch
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"  # vLLM 每批最大 token 预算
RAY_CPUS="${RAY_CPUS:-64}"             # Ray 初始化 CPU 配额
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"    # vLLM 可使用的 GPU 显存比例
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"  # 默认使用 8 张 GPU
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
ORACLE_REWARD="${ORACLE_REWARD:-0}"
ORACLE_FLAG_REWARD="${ORACLE_FLAG_REWARD:-0}"
ORACLE_GUESS_REWARD="${ORACLE_GUESS_REWARD:-0}"
NON_ORACLE_PENALTY="${NON_ORACLE_PENALTY:-0}"
NON_ORACLE_FLAG_PENALTY="${NON_ORACLE_FLAG_PENALTY:-0}"
INVALID_PENALTY="${INVALID_PENALTY:--1}"
MINES="${MINES:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_minesweeper"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Minesweeper | standard GRPO + outcome reward only (32x8) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: vanilla | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Reward: outcome only | oracle_process: $ORACLE_REWARD/$ORACLE_FLAG_REWARD/$ORACLE_GUESS_REWARD | non_oracle_process: $NON_ORACLE_PENALTY/$NON_ORACLE_FLAG_PENALTY | invalid: $INVALID_PENALTY | mines: $MINES"
echo "Run dir:      $RUN_DIR"
echo "Resume:       mode=$RESUME_MODE path=${RESUME_FROM_PATH:-auto/latest-or-none}"

if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: Model not found at $MODEL_PATH" >&2; exit 1; fi
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found at $PYTHON" >&2; exit 1; fi
if [ "$RESUME_MODE" = "resume_path" ] && [ -z "$RESUME_FROM_PATH" ]; then
    echo "ERROR: RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
fi
if [ -n "$RESUME_FROM_PATH" ] && [ ! -d "$RESUME_FROM_PATH" ]; then
    echo "ERROR: RESUME_FROM_PATH not found: $RESUME_FROM_PATH" >&2
    exit 1
fi
if ! "$PYTHON" -c "import gem" 2>/dev/null; then
    echo "ERROR: 'gem' not found in $PYTHON (pip install 'git+https://github.com/axon-rl/gem.git')" >&2
    exit 1
fi

"$PYTHON" "$VPR_GAMES_DIR/prepare_data.py" \
    --env-name vpr_minesweeper --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    env.seed=0 \
    env.rollout.mode=vanilla \
    env.rollout.selection_mode=best \
    env.rollout.random_select_prob=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.minesweeper.reward_mode=outcome \
    env.minesweeper.mines="$MINES" \
    env.minesweeper.oracle_reward="$ORACLE_REWARD" \
    env.minesweeper.oracle_flag_reward="$ORACLE_FLAG_REWARD" \
    env.minesweeper.oracle_guess_reward="$ORACLE_GUESS_REWARD" \
    env.minesweeper.non_oracle_penalty="$NON_ORACLE_PENALTY" \
    env.minesweeper.non_oracle_flag_penalty="$NON_ORACLE_FLAG_PENALTY" \
    env.invalid_penalty="$INVALID_PENALTY" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=vpr_minesweeper \
    trainer.experiment_name="grpo_outcome_32x8_${TS}" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.logger=["console","tensorboard"] \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$LOG_FILE"

echo ""
echo "All outputs under: $RUN_DIR  (train.log / ckpt/ / tensorboard/ / hydra/)"
echo "  tensorboard --logdir $RUN_DIR/tensorboard"
echo "=== done ==="
