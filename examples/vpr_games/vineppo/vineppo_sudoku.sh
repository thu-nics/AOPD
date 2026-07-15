#!/bin/bash
# ============================================================================
# Sudoku —— VinePPO（逐-turn 过程奖励 + VinePPO advantage）训练脚本
#
#   * algorithm.adv_estimator=vineppo → MC continuation value baseline：
#     A(s,a)=r+gamma*V(next_state)-V(state)，并在 batch 内归一化。
#   * env.sudoku.reward_mode 由 REWARD_MODE 控制，默认 outcome；可设 oracle 使用逐 turn imitation 奖励：
#     forced cell correct digit +2；MRV cell correct digit +1；
#     legal non-MRV digit correct +0.5；wrong digit -1；
#     cell_not_blank/out_of_range -2；parse/truncate -2。
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
TRAIN_STEPS="${TRAIN_STEPS:-100}"       # 训练步数
TRAIN_BATCH="${TRAIN_BATCH:-128}"         # 每个训练 step 的 prompt 数
ROLLOUT_N="${ROLLOUT_N:-1}"            # vanilla 独立轨迹数
VINE_K="${VINE_K:-5}"  # 每个 state 的 MC continuation 次数
VINE_TRAIN_TRAJ="${VINE_TRAIN_TRAJ:-32}"  # null 表示训练所有采样轨迹；设为 32 时只对 32 条轨迹做 MC 和 actor loss
VINE_MC_ENABLE_THINKING="${VINE_MC_ENABLE_THINKING:-True}"
REWARD_MODE="${REWARD_MODE:-outcome}"
VAL_BATCH="${VAL_BATCH:-64}"            # 每次验证的轨迹数
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"  # PPO 更新使用的 mini-batch
MAX_RESP="${MAX_RESP:-4096}"           # 生成响应的最大 token 长度
SAVE_FREQ="${SAVE_FREQ:-25}"           # checkpoint 保存间隔
RESUME_MODE="${RESUME_MODE:-disable}"     # disable/auto/resume_path
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"  # RESUME_MODE=resume_path 时指定 global_step_* 目录
TEST_FREQ="${TEST_FREQ:-20}"           # 验证间隔
ENABLE_THINKING="${ENABLE_THINKING:-True}"  # Qwen chat template thinking 开关
USE_KL="${USE_KL:-True}"               # actor KL loss 开关
KL_COEF="${KL_COEF:-0.001}"            # actor KL loss 系数
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-1}"  # terminal outcome bonus disabled for imitation
FORCED_REWARD="${FORCED_REWARD:-2}"
MRV_REWARD="${MRV_REWARD:-1}"
LEGAL_NON_ORACLE_REWARD="${LEGAL_NON_ORACLE_REWARD:-0}"
WRONG_DIGIT_PENALTY="${WRONG_DIGIT_PENALTY:--1}"
CELL_ERROR_PENALTY="${CELL_ERROR_PENALTY:--2}"
INVALID_PENALTY="${INVALID_PENALTY:--1}"
NUM_BLANKS="${NUM_BLANKS:-10}"
PPO_MICRO="${PPO_MICRO:-2}"            # actor 训练 micro-batch
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"    # rollout/ref log-prob micro-batch
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"  # vLLM 每批最大 token 预算
RAY_CPUS="${RAY_CPUS:-64}"             # Ray 初始化 CPU 配额
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"    # vLLM 可使用的 GPU 显存比例
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"  # 默认使用 8 张 GPU
N_GPUS="${N_GPUS:-8}"                  # trainer 使用的 GPU 数量
TP_SIZE="${TP_SIZE:-2}"                  # 8GPU 下默认使用 2 路 TP、4 路 rollout DP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_sudoku"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Sudoku | VinePPO (per-turn oracle reward + VinePPO advantage) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: vanilla | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | vine_k: $VINE_K | train_traj: $VINE_TRAIN_TRAJ | mc_thinking: $VINE_MC_ENABLE_THINKING | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Reward:       mode:$REWARD_MODE | forced:$FORCED_REWARD | mrv:$MRV_REWARD | legal_non_mrv_correct:$LEGAL_NON_ORACLE_REWARD | wrong_digit:$WRONG_DIGIT_PENALTY | cell_error:$CELL_ERROR_PENALTY | invalid/truncate:$INVALID_PENALTY | outcome_scale:$OUTCOME_REWARD_SCALE"
echo "Sudoku:      blanks:$NUM_BLANKS"
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
    --env-name vpr_sudoku --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
TENSORBOARD_DIR="$RUN_DIR/tensorboard" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_sudoku \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    env.rollout.n="$ROLLOUT_N" \
    env.rollout.mode=vanilla \
    env.invalid_penalty="$INVALID_PENALTY" \
    env.sudoku.reward_mode="$REWARD_MODE" \
    env.sudoku.clues="$NUM_BLANKS" \
    env.sudoku.forced_reward="$FORCED_REWARD" \
    env.sudoku.mrv_reward="$MRV_REWARD" \
    env.sudoku.legal_non_oracle_reward="$LEGAL_NON_ORACLE_REWARD" \
    env.sudoku.wrong_digit_penalty="$WRONG_DIGIT_PENALTY" \
    env.sudoku.cell_error_penalty="$CELL_ERROR_PENALTY" \
    algorithm.vpr.outcome_reward_scale="$OUTCOME_REWARD_SCALE" \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=vineppo \
    algorithm.vineppo.num_rollouts_per_state="$VINE_K" \
    algorithm.vineppo.max_train_trajectories="$VINE_TRAIN_TRAJ" \
    algorithm.vineppo.mc_enable_thinking="$VINE_MC_ENABLE_THINKING" \
    algorithm.vineppo.normalize_adv=True \
    algorithm.vineppo.snapshot_fields_cleanup=True \
    reward_model.enable=False \
    env.history_length=0 \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=vpr_sudoku \
    trainer.experiment_name="vineppo_${TS}" \
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
