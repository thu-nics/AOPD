#!/bin/bash
# ============================================================================
# Minesweeper —— VPR（逐-turn 过程奖励 + VPR advantage）训练脚本
#
#   * algorithm.adv_estimator=vpr（config 默认，不覆盖）→ compute_vpr_turn_level_advantage：
#     逐-turn 归一化的过程 advantage（排除 padding）。
#   * env.minesweeper.reward_mode=oracle → turn-level oracle-action imitation 奖励：
#     safe reveal +2；certain flag +1；min-posterior guess +1；non-oracle reveal 0；non-oracle flag -1；invalid/truncate -2。
#   * 与 grpo_minesweeper_outcome.sh 的区别仅在 adv_estimator(vpr vs grpo) 与 reward_mode
#     (oracle vs outcome)；其余训练超参一致。
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
TRAIN_BATCH="${TRAIN_BATCH:-64}"         # 每个训练 step 的 prompt 数
ROLLOUT_N="${ROLLOUT_N:-4}"            # vanilla: 每 prompt rollout 数；state_group: 每 state 候选数
ROLLOUT_MODE="${ROLLOUT_MODE:-state_group}"  # vanilla 或 state_group（Minesweeper-only）
SELECTION_MODE="${SELECTION_MODE:-mixed}"  # state_group candidate commit: best/random/mixed
RANDOM_SELECT_PROB="${RANDOM_SELECT_PROB:-0}"  # mixed 模式下随机执行 candidate 的概率
RANDOM_SELECT_PROB_SCHEDULE="${RANDOM_SELECT_PROB_SCHEDULE:-}"  # 例如 1:0,50:0.1,100:0.2
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
N_GPUS="${N_GPUS:-8}"                  # trainer 使用的 GPU 数量
TP_SIZE="${TP_SIZE:-2}"                  # 8GPU 下默认使用 2 路 TP、4 路 rollout DP
ORACLE_POLICY="${ORACLE_POLICY:-all_oracle_actions}"  # turn-level oracle action policy
ORACLE_REWARD="${ORACLE_REWARD:-2}"       # safe reveal reward
ORACLE_FLAG_REWARD="${ORACLE_FLAG_REWARD:-1}"  # certain flag reward
ORACLE_GUESS_REWARD="${ORACLE_GUESS_REWARD:-1}"  # min-posterior guess reveal reward
NON_ORACLE_PENALTY="${NON_ORACLE_PENALTY:--1}"  # legal non-oracle reveal reward
NON_ORACLE_FLAG_PENALTY="${NON_ORACLE_FLAG_PENALTY:--1.5}"  # legal non-oracle flag penalty
INVALID_PENALTY="${INVALID_PENALTY:--2}"  # parse/illegal/truncate penalty
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-0}"  # terminal outcome bonus disabled for imitation
STATE_GROUP_ADV_MODE="${STATE_GROUP_ADV_MODE:-mean_then_batch_whiten}"  # group_whiten 或 mean_then_batch_whiten
VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD="${VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD:-null}"  # null disables update skipping
LOSS_MODE="${LOSS_MODE:-vanilla}"  # vanilla 或 gspo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_minesweeper"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Minesweeper | VPR (per-turn oracle reward + VPR advantage) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: $ROLLOUT_MODE | selection: $SELECTION_MODE p_random=$RANDOM_SELECT_PROB schedule=${RANDOM_SELECT_PROB_SCHEDULE:-none} | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Reward:       policy: $ORACLE_POLICY | safe_reveal: $ORACLE_REWARD | certain_flag: $ORACLE_FLAG_REWARD | guess: $ORACLE_GUESS_REWARD | non_oracle_reveal: $NON_ORACLE_PENALTY | non_oracle_flag: $NON_ORACLE_FLAG_PENALTY | invalid/truncate: $INVALID_PENALTY | outcome_scale: $OUTCOME_REWARD_SCALE | loss_mode: $LOSS_MODE"
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
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
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
    env.minesweeper.oracle_policy="$ORACLE_POLICY" \
    env.minesweeper.oracle_reward="$ORACLE_REWARD" \
    env.minesweeper.oracle_flag_reward="$ORACLE_FLAG_REWARD" \
    env.minesweeper.oracle_guess_reward="$ORACLE_GUESS_REWARD" \
    env.minesweeper.non_oracle_penalty="$NON_ORACLE_PENALTY" \
    env.minesweeper.non_oracle_flag_penalty="$NON_ORACLE_FLAG_PENALTY" \
    env.invalid_penalty="$INVALID_PENALTY" \
    env.minesweeper.reward_mode=oracle \
    env.minesweeper.mines=5 \
    algorithm.vpr.outcome_reward_scale="$OUTCOME_REWARD_SCALE" \
    algorithm.vpr.state_group_advantage_mode="$STATE_GROUP_ADV_MODE" \
    algorithm.vpr.skip_update_equal_reward_threshold="$VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD" \
    env.seed=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.rollout.mode="$ROLLOUT_MODE" \
    env.rollout.selection_mode="$SELECTION_MODE" \
    env.rollout.random_select_prob="$RANDOM_SELECT_PROB" \
    env.rollout.random_select_prob_schedule="'${RANDOM_SELECT_PROB_SCHEDULE}'" \
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
    trainer.experiment_name="vpr_${TS}" \
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
