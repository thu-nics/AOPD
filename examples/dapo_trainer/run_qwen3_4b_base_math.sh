#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/.cache/model_tests/Qwen3-4B-Base}"
DAPO_SOURCE_TRAIN="${DAPO_SOURCE_TRAIN:-$REPO_ROOT/.cache/model_tests/dapo-math-17k.parquet}"
DAPO_SOURCE_VAL="${DAPO_SOURCE_VAL:-$REPO_ROOT/data/dapo/aime-2024.parquet}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-unique.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/aime-2024-unique.parquet}"
RUN_NAME="${RUN_NAME:-dapo_qwen3_4b_base}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%S)}"

TRAIN_STEPS="${TRAIN_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
GEN_BATCH="${GEN_BATCH:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO="${PPO_MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESPONSE="${MAX_RESPONSE:-8192}"
OVERLONG_BUFFER="${OVERLONG_BUFFER:-4096}"
MAX_GEN_BATCHES="${MAX_GEN_BATCHES:-10}"
LR="${LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"
N_GPUS="${N_GPUS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_CKPTS="${MAX_CKPTS:-2}"
RAY_CPUS="${RAY_CPUS:-64}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
    RUN_NAME="${RUN_NAME:-dapo_qwen3_4b_base_smoke}"
    TRAIN_STEPS=1
    SAVE_FREQ=-1
    TEST_FREQ=-1
    TRAIN_BATCH="${SMOKE_TRAIN_BATCH:-8}"
    GEN_BATCH="${SMOKE_GEN_BATCH:-8}"
    ROLLOUT_N="${SMOKE_ROLLOUT_N:-8}"
    PPO_MINI_BATCH="${SMOKE_PPO_MINI_BATCH:-16}"
    MAX_PROMPT="${SMOKE_MAX_PROMPT:-1024}"
    MAX_RESPONSE="${SMOKE_MAX_RESPONSE:-2048}"
    OVERLONG_BUFFER="${SMOKE_OVERLONG_BUFFER:-512}"
    WARMUP_STEPS=0
    RESUME_MODE=disable
fi

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi
for path in "$MODEL_PATH" "$DAPO_SOURCE_TRAIN" "$DAPO_SOURCE_VAL"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done
if [[ ! -e "$TRAIN_FILE" || ! -e "$VAL_FILE" ]]; then
    "$PYTHON" "$SCRIPT_DIR/prepare_dapo_math.py" --train-source "$DAPO_SOURCE_TRAIN" --val-source "$DAPO_SOURCE_VAL" --output-dir "$DATA_DIR"
fi
for path in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: prepared data path does not exist: $path" >&2
        exit 1
    fi
done
if [[ "$RESUME_MODE" == "resume_path" && -z "$RESUME_FROM_PATH" ]]; then
    echo "ERROR: RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
fi
if (( TRAIN_BATCH % N_GPUS != 0 )); then
    echo "ERROR: TRAIN_BATCH=$TRAIN_BATCH must be divisible by N_GPUS=$N_GPUS" >&2
    exit 1
fi
if (( (TRAIN_BATCH * ROLLOUT_N) % PPO_MINI_BATCH != 0 )); then
    echo "ERROR: TRAIN_BATCH*ROLLOUT_N must be divisible by PPO_MINI_BATCH" >&2
    exit 1
fi
if (( OVERLONG_BUFFER <= 0 || OVERLONG_BUFFER >= MAX_RESPONSE )); then
    echo "ERROR: OVERLONG_BUFFER must be in (0, MAX_RESPONSE)" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"
ACTOR_MAX_TOKENS=$((MAX_PROMPT + MAX_RESPONSE))
INFER_MAX_TOKENS=$((MAX_PROMPT + MAX_RESPONSE))

echo "DAPO run:      $RUN_DIR"
echo "Model:         $MODEL_PATH"
echo "Train/val:     $TRAIN_FILE | $VAL_FILE"
echo "Batch:         train=$TRAIN_BATCH gen=$GEN_BATCH rollout_n=$ROLLOUT_N"
echo "Lengths:       prompt=$MAX_PROMPT response=$MAX_RESPONSE overlong=$OVERLONG_BUFFER"
echo "Parallelism:   gpus=$N_GPUS tp=$TP_SIZE sp=$SP_SIZE"
echo "Resume:        mode=$RESUME_MODE path=${RESUME_FROM_PATH:-auto/latest-or-none}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then
    LOGGER='["console"]'
fi

"$PYTHON" -m recipe.dapo.main_dapo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.gen_batch_size="$GEN_BATCH" \
    data.filter_overlong_prompts=False \
    reward_model.reward_manager=dapo \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT + MAX_RESPONSE))" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKENS" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKENS" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.logger="$LOGGER" \
    trainer.project_name=dapo-zero-rl \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train="$([[ "$SMOKE" == "1" ]] && echo False || echo True)" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep="$MAX_CKPTS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$LOG_FILE"
