#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VARIANT="${VARIANT:?Set VARIANT to vpr or outcome}"
if [[ "$VARIANT" != "vpr" && "$VARIANT" != "outcome" ]]; then
    echo "ERROR: VARIANT must be vpr or outcome" >&2
    exit 1
fi
CONFIG_NAME="tau_${VARIANT}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-8B checkpoint}"
QUALIFICATION_MANIFEST="${QUALIFICATION_MANIFEST:?Set QUALIFICATION_MANIFEST to qualification_manifest.json}"
PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-tau_${VARIANT}_qwen3_8b}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%S)}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"
ORACLE_CACHE="${ORACLE_CACHE:-$(dirname "$QUALIFICATION_MANIFEST")/oracle_state_cache.jsonl}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/.cache/tau2-bench-17e07b1}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"

TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-30}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-25}"
AIRLINE_TRAJ="${AIRLINE_TRAJ:-4}"
RETAIL_TRAJ="${RETAIL_TRAJ:-4}"
VAL_AIRLINE="${VAL_AIRLINE:-4}"
VAL_RETAIL="${VAL_RETAIL:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO="${PPO_MICRO:-1}"
LOGPROB_MICRO="${LOGPROB_MICRO:-1}"
MAX_PROMPT="${MAX_PROMPT:-24576}"
MAX_RESPONSE="${MAX_RESPONSE:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-8192}"
LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-8192}"
OVERLONG_BUFFER="${OVERLONG_BUFFER:-2048}"
MAX_GEN_BATCHES="${MAX_GEN_BATCHES:-10}"
LR="${LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-4}"
N_GPUS="${N_GPUS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_CKPTS="${MAX_CKPTS:-null}"
RAY_CPUS="${RAY_CPUS:-64}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
SMOKE="${SMOKE:-0}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required for the Tau user simulator and oracle}"
for path in "$MODEL_PATH" "$QUALIFICATION_MANIFEST" "$TAU2_DATA_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done
if ! TAU2_DATA_DIR="$TAU2_DATA_DIR" "$PYTHON" -c 'import tau2; import rank_bm25' >/dev/null 2>&1; then
    echo "ERROR: Tau dependencies are incomplete; run PYTHON=$PYTHON bash examples/tau_bench/install_tau2.sh" >&2
    exit 1
fi
if (( AIRLINE_TRAJ + RETAIL_TRAJ != 8 )); then
    echo "ERROR: formal training requires AIRLINE_TRAJ + RETAIL_TRAJ = 8" >&2
    exit 1
fi
if (( ROLLOUT_N != 4 )); then
    echo "ERROR: formal training requires ROLLOUT_N=4" >&2
    exit 1
fi
if [[ "$RESUME_MODE" == "resume_path" && -z "$RESUME_FROM_PATH" ]]; then
    echo "ERROR: RESUME_FROM_PATH is required for RESUME_MODE=resume_path" >&2
    exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
    TRAIN_STEPS=1
    PPO_MINI_BATCH="${SMOKE_PPO_MINI_BATCH:-8}"
    TRAIN_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
    EVAL_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
    SAVE_FREQ=-1
    TEST_FREQ=-1
    VAL_AIRLINE=1
    VAL_RETAIL=1
    WARMUP_STEPS=0
    MAX_GEN_BATCHES="${SMOKE_MAX_GEN_BATCHES:-1}"
    MAX_PROMPT="${SMOKE_MAX_PROMPT:-8192}"
    MAX_RESPONSE="${SMOKE_MAX_RESPONSE:-1024}"
    MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))
    RESUME_MODE=disable
fi

if (( N_GPUS % SP_SIZE != 0 )); then
    echo "ERROR: N_GPUS must be divisible by SP_SIZE" >&2
    exit 1
fi
MAX_SEQUENCE_TOKENS=$((MAX_PROMPT + MAX_RESPONSE))
if (( MAX_SEQUENCE_TOKENS > PPO_MAX_TOKENS_PER_GPU * SP_SIZE )); then
    echo "ERROR: PPO token budget cannot fit one maximum-length sequence" >&2
    exit 1
fi
if (( MAX_SEQUENCE_TOKENS > LOGPROB_MAX_TOKENS_PER_GPU * SP_SIZE )); then
    echo "ERROR: log-prob token budget cannot fit one maximum-length sequence" >&2
    exit 1
fi

TRAIN_BATCH=$((AIRLINE_TRAJ + RETAIL_TRAJ))
VAL_BATCH=$((VAL_AIRLINE + VAL_RETAIL))
mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard" "$DATA_DIR"
"$PYTHON" "$SCRIPT_DIR/prepare_tau_training.py" \
    --qualification-manifest "$QUALIFICATION_MANIFEST" \
    --output-dir "$DATA_DIR" \
    --train-steps "$TRAIN_STEPS" \
    --airline "$AIRLINE_TRAJ" \
    --retail "$RETAIL_TRAJ" \
    --val-airline "$VAL_AIRLINE" \
    --val-retail "$VAL_RETAIL"

LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then
    LOGGER='["console"]'
fi
LOG_FILE="$RUN_DIR/train.log"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export TAU2_DATA_DIR
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

ORACLE_OVERRIDES=()
if [[ "$VARIANT" == "vpr" ]]; then
    ORACLE_OVERRIDES=("env.tau.oracle.cache_path=$ORACLE_CACHE")
fi

VARIANT_OVERRIDES=(
    "reward_model.reward_manager=dapo_turn"
    "reward_model.overlong_buffer.enable=True"
    "reward_model.overlong_buffer.len=$OVERLONG_BUFFER"
    "algorithm.adv_estimator=dapo"
    "actor_rollout_ref.actor.optim.weight_decay=0.1"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.28"
    "actor_rollout_ref.actor.clip_ratio_c=10.0"
)
if [[ "$VARIANT" == "outcome" ]]; then
    VARIANT_OVERRIDES=(
        "reward_model.reward_manager=turn"
        "reward_model.overlong_buffer.enable=False"
        "algorithm.adv_estimator=grpo"
        "algorithm.filter_groups.enable=False"
        "actor_rollout_ref.actor.optim.weight_decay=0.01"
        "actor_rollout_ref.actor.entropy_coeff=0"
        "actor_rollout_ref.actor.clip_ratio_low=0.2"
        "actor_rollout_ref.actor.clip_ratio_high=0.2"
        "actor_rollout_ref.actor.clip_ratio_c=3.0"
    )
fi

echo "Tau $VARIANT run: $RUN_DIR"
echo "Committed task groups: Airline=$AIRLINE_TRAJ Retail=$RETAIL_TRAJ; group size=$ROLLOUT_N"
echo "Per-GPU dynamic token budgets: PPO=$PPO_MAX_TOKENS_PER_GPU log-prob=$LOGPROB_MAX_TOKENS_PER_GPU; SP=$SP_SIZE"

"$PYTHON" -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/validation.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    data.shuffle=False \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    "${VARIANT_OVERRIDES[@]}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.max_num_gen_batches="$MAX_GEN_BATCHES" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$WARMUP_STEPS" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    +actor_rollout_ref.rollout.min_p=0.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    env.seed=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.tau.qualification_manifest="$QUALIFICATION_MANIFEST" \
    env.tau.train_max_steps="$TRAIN_MAX_STEPS" \
    env.tau.eval_max_steps="$EVAL_MAX_STEPS" \
    env.tau.trajectory_counts.airline="$AIRLINE_TRAJ" \
    env.tau.trajectory_counts.retail="$RETAIL_TRAJ" \
    env.tau.validation_counts.airline="$VAL_AIRLINE" \
    env.tau.validation_counts.retail="$VAL_RETAIL" \
    "${ORACLE_OVERRIDES[@]}" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$([[ "$SMOKE" == "1" ]] && echo False || echo True)" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name="tau-bench" \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep="$MAX_CKPTS" \
    trainer.max_critic_ckpt_to_keep="$MAX_CKPTS" \
    trainer.logger="$LOGGER" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$LOG_FILE"
