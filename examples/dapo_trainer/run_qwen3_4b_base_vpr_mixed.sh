#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-dapo_vpr_mixed}"
TRAINING_VARIANT="${TRAINING_VARIANT:-VPR state-group}"
PROJECT_NAME="${PROJECT_NAME:-dapo-vpr-mixed}"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/.cache/model_tests/Qwen3-4B-Base}"
DAPO_TRAIN="${DAPO_TRAIN:-$REPO_ROOT/data/dapo/dapo-math-17k-unique.parquet}"
DAPO_VAL="${DAPO_VAL:-$REPO_ROOT/data/dapo/aime-2024-unique.parquet}"
RUN_NAME="${RUN_NAME:-dapo_vpr_mixed_base_math8_games4_6_8_18_boxed}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%S)}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"

TRAIN_STEPS="${TRAIN_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
MATH_TRAJ="${MATH_TRAJ:-64}"
SOKOBAN_TRAJ="${SOKOBAN_TRAJ:-6}"
SUDOKU_TRAJ="${SUDOKU_TRAJ:-8}"
MINESWEEPER_TRAJ="${MINESWEEPER_TRAJ:-18}"
VAL_MATH="${VAL_MATH:-30}"
VAL_PER_GAME="${VAL_PER_GAME:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MATH_ROLLOUT_N="${MATH_ROLLOUT_N:-8}"
GAME_ROLLOUT_N="${GAME_ROLLOUT_N:-4}"
GAME_ACTION_FORMAT="${GAME_ACTION_FORMAT:-boxed}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO="${PPO_MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESPONSE="${MAX_RESPONSE:-8192}"
OVERLONG_BUFFER="${OVERLONG_BUFFER:-4096}"
MAX_GEN_BATCHES="${MAX_GEN_BATCHES:-10}"
LR="${LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
DIM_ROOM="${DIM_ROOM:-6,6}"
NUM_BOXES="${NUM_BOXES:-2}"
SOKOBAN_MAX_STEPS="${SOKOBAN_MAX_STEPS:-24}"
SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS="${SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS:-15}"
NUM_BLANKS="${NUM_BLANKS:-40}"
SUDOKU_MAX_STEPS="${SUDOKU_MAX_STEPS:-40}"
SUDOKU_TRAIN_ROLLOUT_MAX_STEPS="${SUDOKU_TRAIN_ROLLOUT_MAX_STEPS:-15}"
SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS="${SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS:-}"
NUM_MINES="${NUM_MINES:-4}"
MINESWEEPER_MAX_STEPS="${MINESWEEPER_MAX_STEPS:-15}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"
N_GPUS="${N_GPUS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_CKPTS="${MAX_CKPTS:-2}"
RAY_CPUS="${RAY_CPUS:-64}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
    TRAIN_STEPS=1
    SAVE_FREQ=-1
    TEST_FREQ=-1
    MATH_TRAJ="${SMOKE_MATH_TRAJ:-4}"
    SOKOBAN_TRAJ="${SMOKE_SOKOBAN_TRAJ:-1}"
    SUDOKU_TRAJ="${SMOKE_SUDOKU_TRAJ:-1}"
    MINESWEEPER_TRAJ="${SMOKE_MINESWEEPER_TRAJ:-2}"
    VAL_MATH=1
    VAL_PER_GAME=1
    ROLLOUT_N="${SMOKE_ROLLOUT_N:-8}"
    MATH_ROLLOUT_N="${SMOKE_MATH_ROLLOUT_N:-8}"
    GAME_ROLLOUT_N="${SMOKE_GAME_ROLLOUT_N:-4}"
    PPO_MINI_BATCH="${SMOKE_PPO_MINI_BATCH:-8}"
    MAX_PROMPT="${SMOKE_MAX_PROMPT:-1024}"
    MAX_RESPONSE="${SMOKE_MAX_RESPONSE:-1024}"
    OVERLONG_BUFFER="${SMOKE_OVERLONG_BUFFER:-256}"
    WARMUP_STEPS=0
    ENABLE_THINKING=True
    DIM_ROOM=6,6
    NUM_BOXES=1
    SOKOBAN_MAX_STEPS=4
    SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS=4
    NUM_BLANKS=2
    SUDOKU_MAX_STEPS=2
    SUDOKU_TRAIN_ROLLOUT_MAX_STEPS=2
    if [[ "$CONFIG_NAME" == "dapo_games_non_vpr_mixed" ]]; then
        SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS="${SMOKE_SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS:-1}"
    fi
    NUM_MINES=1
    MINESWEEPER_MAX_STEPS=4
    RESUME_MODE=disable
fi

TRAIN_BATCH=$((MATH_TRAJ + SOKOBAN_TRAJ + SUDOKU_TRAJ + MINESWEEPER_TRAJ))
VAL_BATCH=$((VAL_MATH + 3 * VAL_PER_GAME))
if (( TRAIN_BATCH % N_GPUS != 0 )); then
    echo "ERROR: mixed TRAIN_BATCH=$TRAIN_BATCH must be divisible by N_GPUS=$N_GPUS" >&2
    exit 1
fi
if ! [[ "$MATH_ROLLOUT_N" =~ ^[1-9][0-9]*$ ]] || \
        ! [[ "$GAME_ROLLOUT_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MATH_ROLLOUT_N and GAME_ROLLOUT_N must be positive integers" >&2
    exit 1
fi
if [[ "$GAME_ACTION_FORMAT" != "action_tag" && "$GAME_ACTION_FORMAT" != "boxed" ]]; then
    echo "ERROR: GAME_ACTION_FORMAT must be action_tag or boxed" >&2
    exit 1
fi
if (( OVERLONG_BUFFER <= 0 || OVERLONG_BUFFER >= MAX_RESPONSE )); then
    echo "ERROR: OVERLONG_BUFFER must be in (0, MAX_RESPONSE)" >&2
    exit 1
fi
if ! [[ "$SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || \
        (( SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS > SOKOBAN_MAX_STEPS )); then
    echo "ERROR: SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS must be in [1, SOKOBAN_MAX_STEPS]" >&2
    exit 1
fi
if ! [[ "$SUDOKU_TRAIN_ROLLOUT_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || \
        (( SUDOKU_TRAIN_ROLLOUT_MAX_STEPS > SUDOKU_MAX_STEPS )); then
    echo "ERROR: SUDOKU_TRAIN_ROLLOUT_MAX_STEPS must be in [1, SUDOKU_MAX_STEPS]" >&2
    exit 1
fi
if [[ "$CONFIG_NAME" == "dapo_games_non_vpr_mixed" ]]; then
    if ! [[ "$SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS" =~ ^[1-9][0-9]*$ ]] || \
            (( SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS > NUM_BLANKS )) || \
            (( SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS > SUDOKU_MAX_STEPS )); then
        echo "ERROR: SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS must be in [1, min(NUM_BLANKS, SUDOKU_MAX_STEPS)]" >&2
        exit 1
    fi
fi
if [[ "$RESUME_MODE" == "resume_path" && -z "$RESUME_FROM_PATH" ]]; then
    echo "ERROR: RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi
for path in "$MODEL_PATH" "$DAPO_TRAIN" "$DAPO_VAL"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard" "$DATA_DIR"
"$PYTHON" "$SCRIPT_DIR/prepare_dapo_vpr_mixed.py" \
    --dapo-train "$DAPO_TRAIN" \
    --dapo-val "$DAPO_VAL" \
    --output-dir "$DATA_DIR" \
    --train-steps "$TRAIN_STEPS" \
    --math-pool-size "$MAX_GEN_BATCHES" \
    --math "$MATH_TRAJ" \
    --sokoban "$SOKOBAN_TRAJ" \
    --sudoku "$SUDOKU_TRAJ" \
    --minesweeper "$MINESWEEPER_TRAJ" \
    --val-math "$VAL_MATH" \
    --val-per-game "$VAL_PER_GAME"

LOG_FILE="$RUN_DIR/train.log"
ACTOR_MAX_TOKENS=$((MAX_PROMPT + MAX_RESPONSE))
INFER_MAX_TOKENS=$((MAX_PROMPT + MAX_RESPONSE))
LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then
    LOGGER='["console"]'
fi

echo "Mixed DAPO run ($TRAINING_VARIANT): $RUN_DIR"
echo "Base groups: math=$MATH_TRAJ sokoban=$SOKOBAN_TRAJ sudoku=$SUDOKU_TRAJ minesweeper=$MINESWEEPER_TRAJ"
TRAIN_HORIZON_OVERRIDES=()
GROUP_SIZE_OVERRIDES=()
OUTCOME_TARGET_OVERRIDES=()
if [[ "$CONFIG_NAME" == "dapo_vpr_mixed" ]]; then
    echo "Candidate groups: math=$MATH_ROLLOUT_N games=$GAME_ROLLOUT_N | first-pass candidates: $((MATH_TRAJ * MATH_ROLLOUT_N + (SOKOBAN_TRAJ + SUDOKU_TRAJ + MINESWEEPER_TRAJ) * GAME_ROLLOUT_N))"
    echo "Sokoban train rollout horizon: $SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS (environment/eval: $SOKOBAN_MAX_STEPS)"
    echo "Sudoku train rollout horizon: $SUDOKU_TRAIN_ROLLOUT_MAX_STEPS (environment/eval: $SUDOKU_MAX_STEPS)"
    GROUP_SIZE_OVERRIDES=(
        "env.rollout.math_n=$MATH_ROLLOUT_N"
        "env.rollout.game_n=$GAME_ROLLOUT_N"
    )
    TRAIN_HORIZON_OVERRIDES=(
        "env.sokoban.train_rollout_max_steps=$SOKOBAN_TRAIN_ROLLOUT_MAX_STEPS"
        "env.sudoku.train_rollout_max_steps=$SUDOKU_TRAIN_ROLLOUT_MAX_STEPS"
    )
else
    echo "Rollouts per outcome group: $ROLLOUT_N | first-pass rollout count: $((TRAIN_BATCH * ROLLOUT_N))"
    echo "Outcome rollout horizons: sokoban=$SOKOBAN_MAX_STEPS sudoku=$SUDOKU_MAX_STEPS minesweeper=$MINESWEEPER_MAX_STEPS"
    echo "Sudoku outcome target: $SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS correct fills"
    OUTCOME_TARGET_OVERRIDES=(
        "env.sudoku.outcome_success_correct_fills=$SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS"
    )
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

"$PYTHON" -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/validation.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    data.shuffle=False \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    reward_model.reward_manager=dapo_turn \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len="$OVERLONG_BUFFER" \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    reward_model.overlong_buffer.log=True \
    algorithm.adv_estimator=dapo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches="$MAX_GEN_BATCHES" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$WARMUP_STEPS" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKENS" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKENS" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    env.seed=0 \
    env.game_action_format="$GAME_ACTION_FORMAT" \
    env.rollout.n="$ROLLOUT_N" \
    "${GROUP_SIZE_OVERRIDES[@]}" \
    env.mixed.trajectory_counts.math="$MATH_TRAJ" \
    env.mixed.trajectory_counts.sokoban="$SOKOBAN_TRAJ" \
    env.mixed.trajectory_counts.sudoku="$SUDOKU_TRAJ" \
    env.mixed.trajectory_counts.minesweeper="$MINESWEEPER_TRAJ" \
    env.mixed.validation_counts.math="$VAL_MATH" \
    env.mixed.validation_counts.sokoban="$VAL_PER_GAME" \
    env.mixed.validation_counts.sudoku="$VAL_PER_GAME" \
    env.mixed.validation_counts.minesweeper="$VAL_PER_GAME" \
    env.sokoban.dim_room=[$DIM_ROOM] \
    env.sokoban.num_boxes="$NUM_BOXES" \
    env.sokoban.max_steps="$SOKOBAN_MAX_STEPS" \
    "${TRAIN_HORIZON_OVERRIDES[@]}" \
    env.sudoku.clues="$NUM_BLANKS" \
    env.sudoku.max_steps="$SUDOKU_MAX_STEPS" \
    "${OUTCOME_TARGET_OVERRIDES[@]}" \
    env.minesweeper.mines="$NUM_MINES" \
    env.minesweeper.max_steps="$MINESWEEPER_MAX_STEPS" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$([[ "$SMOKE" == "1" ]] && echo False || echo True)" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep="$MAX_CKPTS" \
    trainer.logger="$LOGGER" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$LOG_FILE"
