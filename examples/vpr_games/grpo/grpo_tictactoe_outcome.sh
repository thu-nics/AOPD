#!/bin/bash
# ============================================================================
# TicTacToe —— 标准 GRPO + outcome（结果）奖励 训练脚本
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_BATCH="${VAL_BATCH:-128}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
MAX_RESP="${MAX_RESP:-4096}"
SAVE_FREQ="${SAVE_FREQ:-25}"
RESUME_MODE="${RESUME_MODE:-disable}"     # disable/auto/resume_path
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"  # RESUME_MODE=resume_path 时指定 global_step_* 目录
TEST_FREQ="${TEST_FREQ:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
USE_KL="${USE_KL:-True}"
KL_COEF="${KL_COEF:-0.001}"
PPO_MICRO="${PPO_MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
RAY_CPUS="${RAY_CPUS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
OPPONENT="${OPPONENT:-mcts}"
MCTS_SIMS="${MCTS_SIMS:-100}"
INVALID_PENALTY="${INVALID_PENALTY:--1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_tictactoe"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== TicTacToe | standard GRPO + outcome reward (32x8) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: vanilla | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Reward: outcome only | opponent: $OPPONENT | mcts_sims: $MCTS_SIMS | invalid: $INVALID_PENALTY"
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

"$PYTHON" "$VPR_GAMES_DIR/prepare_data.py" \
    --env-name vpr_tictactoe --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
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
    +env.rollout.mode=vanilla \
    +env.rollout.selection_mode=best \
    +env.rollout.random_select_prob=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.tictactoe.agent_player=X \
    env.tictactoe.opponent="$OPPONENT" \
    env.tictactoe.mcts.max_simulations="$MCTS_SIMS" \
    env.tictactoe.reward_mode=outcome \
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
    trainer.project_name=vpr_tictactoe \
    trainer.experiment_name="grpo_outcome_${TS}" \
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
