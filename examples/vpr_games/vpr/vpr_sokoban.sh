#!/bin/bash
# ============================================================================
# Sokoban —— VPR（逐-turn 最短路径 oracle reward + VPR advantage）训练脚本
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_MODE="${ROLLOUT_MODE:-state_group}"
SELECTION_MODE="${SELECTION_MODE:-mixed}"
RANDOM_SELECT_PROB="${RANDOM_SELECT_PROB:-0}"
VAL_BATCH="${VAL_BATCH:-64}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
MAX_RESP="${MAX_RESP:-4096}"
SAVE_FREQ="${SAVE_FREQ:-25}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
TEST_FREQ="${TEST_FREQ:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
USE_KL="${USE_KL:-True}"
KL_COEF="${KL_COEF:-0.001}"
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-0}"
STATE_GROUP_ADV_MODE="${STATE_GROUP_ADV_MODE:-mean_then_batch_whiten}"
VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD="${VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD:-null}"  # null disables update skipping
ORACLE_REWARD="${ORACLE_REWARD:-2}"
LEGAL_NON_ORACLE_REWARD="${LEGAL_NON_ORACLE_REWARD:-0}"
VPR_REWARD_NOISE_PROB="${VPR_REWARD_NOISE_PROB:-0}"
INVALID_PENALTY="${INVALID_PENALTY:--2}"
DIM_ROOM="${DIM_ROOM:-7,7}"
NUM_BOXES="${NUM_BOXES:-3}"
SEARCH_DEPTH="${SEARCH_DEPTH:-25}"
MAX_STEPS="${MAX_STEPS:-36}"
PPO_MICRO="${PPO_MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
RAY_CPUS="${RAY_CPUS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_sokoban"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$(pwd)/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Sokoban | VPR (shortest-path oracle reward + VPR advantage) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: $ROLLOUT_MODE | selection: $SELECTION_MODE p_random=$RANDOM_SELECT_PROB | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP"
echo "Reward:       oracle:$ORACLE_REWARD | legal_non_oracle:$LEGAL_NON_ORACLE_REWARD | noise_prob:$VPR_REWARD_NOISE_PROB | invalid/truncate:$INVALID_PENALTY | outcome_scale:$OUTCOME_REWARD_SCALE"
echo "Sokoban:      dim_room:[$DIM_ROOM] | boxes:$NUM_BOXES | search_depth:$SEARCH_DEPTH | max_steps:$MAX_STEPS"
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
if ! "$PYTHON" -c "import gym_sokoban" 2>/dev/null; then
    echo "ERROR: 'gym_sokoban' not found in $PYTHON" >&2
    exit 1
fi

"$PYTHON" "$VPR_GAMES_DIR/prepare_data.py" \
    --env-name vpr_sokoban --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
TENSORBOARD_DIR="$RUN_DIR/tensorboard" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_sokoban \
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
    env.max_steps="$MAX_STEPS" \
    env.rollout.n="$ROLLOUT_N" \
    env.rollout.mode="$ROLLOUT_MODE" \
    env.rollout.selection_mode="$SELECTION_MODE" \
    env.rollout.random_select_prob="$RANDOM_SELECT_PROB" \
    env.invalid_penalty="$INVALID_PENALTY" \
    env.sokoban.dim_room=[$DIM_ROOM] \
    env.sokoban.num_boxes="$NUM_BOXES" \
    env.sokoban.search_depth="$SEARCH_DEPTH" \
    env.sokoban.reward_mode=oracle \
    env.sokoban.oracle_reward="$ORACLE_REWARD" \
    env.sokoban.legal_non_oracle_reward="$LEGAL_NON_ORACLE_REWARD" \
    env.sokoban.reward_noise_prob="$VPR_REWARD_NOISE_PROB" \
    algorithm.vpr.outcome_reward_scale="$OUTCOME_REWARD_SCALE" \
    algorithm.vpr.state_group_advantage_mode="$STATE_GROUP_ADV_MODE" \
    algorithm.vpr.skip_update_equal_reward_threshold="$VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD" \
    algorithm.use_kl_in_reward=False \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=vpr_sokoban \
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
