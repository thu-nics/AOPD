#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

METHOD="${METHOD:-agentic_opd}"
if [[ "$METHOD" != "agentic_opd" && "$METHOD" != "outcome" ]]; then
    echo "ERROR: METHOD must be agentic_opd or outcome" >&2
    exit 1
fi
DEFAULT_DATA_TRUNCATION=left
if [[ "$METHOD" == "agentic_opd" ]]; then
    DEFAULT_DATA_TRUNCATION=error
fi
DATA_TRUNCATION="${DATA_TRUNCATION:-$DEFAULT_DATA_TRUNCATION}"
CONFIG_NAME="tau_${METHOD}"
DEFAULT_COMPACT_STATE_GROUP_ROWS=false
if [[ "$METHOD" == "agentic_opd" ]]; then
    DEFAULT_COMPACT_STATE_GROUP_ROWS=true
fi
MODEL_PATH="${MODEL_PATH:-}"
PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-tau_${METHOD}}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%S)}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"
ORACLE_CACHE="${ORACLE_CACHE:-$RUN_DIR/cache/teacher.jsonl}"
ORACLE_MATCHER_CACHE="${ORACLE_MATCHER_CACHE:-$RUN_DIR/cache/matcher.jsonl}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/../tau2-bench}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"
TAU_USER_MODEL="${TAU_USER_MODEL:-}"
TAU_USER_API_BASE="${TAU_USER_API_BASE:-}"
TAU_USER_API_KEY_ENV="${TAU_USER_API_KEY_ENV:-TAU_USER_API_KEY}"
TAU_TEACHER_MODEL="${TAU_TEACHER_MODEL:-}"
TAU_TEACHER_API_BASE="${TAU_TEACHER_API_BASE:-}"
TAU_NATIVE_LOG_LEVEL="${TAU_NATIVE_LOG_LEVEL:-WARNING}"
TAU_TEACHER_API_KEY_ENV="${TAU_TEACHER_API_KEY_ENV:-TAU_TEACHER_API_KEY}"
TAU_TEACHER_TEMPERATURE="${TAU_TEACHER_TEMPERATURE:-0.6}"
TAU_TEACHER_TOP_P="${TAU_TEACHER_TOP_P:-0.95}"
TAU_TEACHER_TOP_K="${TAU_TEACHER_TOP_K:-20}"
TAU_TEACHER_MIN_P="${TAU_TEACHER_MIN_P:-0.0}"
TAU_TEACHER_MAX_TOKENS="${TAU_TEACHER_MAX_TOKENS:-8192}"
TAU_TEACHER_VALIDITY_MAX_RETRIES="${TAU_TEACHER_VALIDITY_MAX_RETRIES:-2}"
TAU_MASK_MATCHER_REQUIRED_GROUPS="${TAU_MASK_MATCHER_REQUIRED_GROUPS:-false}"
case "$TAU_MASK_MATCHER_REQUIRED_GROUPS" in
    true|false) ;;
    *) echo "ERROR: TAU_MASK_MATCHER_REQUIRED_GROUPS must be true or false" >&2; exit 1 ;;
esac
if [[ "$METHOD" != "agentic_opd" && "$TAU_MASK_MATCHER_REQUIRED_GROUPS" == true ]]; then
    echo "ERROR: matcher-required group masking is only supported for agentic_opd" >&2
    exit 1
fi
TAU_MATCHER_ENABLE_THINKING="${TAU_MATCHER_ENABLE_THINKING:-null}"
TAU_MATCHER_MAX_TOKENS="${TAU_MATCHER_MAX_TOKENS:-null}"
TAU_MATCHER_REASONING_EFFORT="${TAU_MATCHER_REASONING_EFFORT:-null}"
TAU_MATCHER_MAX_CONCURRENT_REQUESTS="${TAU_MATCHER_MAX_CONCURRENT_REQUESTS:-32}"
TAU_TRANSFER_REWARD_GUARD="${TAU_TRANSFER_REWARD_GUARD:-true}"
TAU_MATCHER_PROVIDER="${TAU_MATCHER_PROVIDER:-openai-compatible}"
TAU_MATCHER_MODEL="${TAU_MATCHER_MODEL:-$TAU_TEACHER_MODEL}"
TAU_MATCHER_API_BASE="${TAU_MATCHER_API_BASE:-$TAU_TEACHER_API_BASE}"
if [[ "$METHOD" == "agentic_opd" && "$TAU_MASK_MATCHER_REQUIRED_GROUPS" == false && "${TAU_MATCHER_API_BASE%/}" != "${TAU_TEACHER_API_BASE%/}" && -z "${TAU_MATCHER_API_KEY_ENV:-}" ]]; then
    echo "ERROR: a separate matcher endpoint requires explicit TAU_MATCHER_API_KEY_ENV" >&2
    exit 1
fi
TAU_MATCHER_API_KEY_ENV="${TAU_MATCHER_API_KEY_ENV:-$TAU_TEACHER_API_KEY_ENV}"

TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-30}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
AIRLINE_TRAJ="${AIRLINE_TRAJ:-5}"
RETAIL_TRAJ="${RETAIL_TRAJ:-11}"
VAL_BATCH="${VAL_BATCH:-16}"
VALIDATION_DOMAINS="${VALIDATION_DOMAINS:-airline,retail}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-1}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-test}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
VALIDATION_NUM_TASKS="${VALIDATION_NUM_TASKS:-}"
PPO_MICRO="${PPO_MICRO:-1}"
LOGPROB_MICRO="${LOGPROB_MICRO:-1}"
MAX_PROMPT="${MAX_PROMPT:-24576}"
MAX_RESPONSE="${MAX_RESPONSE:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-}"
LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-}"
MAX_GEN_BATCHES="${MAX_GEN_BATCHES:-10}"
LR="${LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
TP_SIZE="${TP_SIZE:-}"
SP_SIZE="${SP_SIZE:-}"
N_GPUS="${N_GPUS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_CKPTS="${MAX_CKPTS:-null}"
RAY_CPUS="${RAY_CPUS:-64}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
TAU_USE_PRIVILEGED_TEACHER_CONTEXT="${TAU_USE_PRIVILEGED_TEACHER_CONTEXT:-false}"
TEACHER_REWARD_MODE="${TEACHER_REWARD_MODE:-frequency_weighted}"
FREQUENCY_BONUS_SCALE="${FREQUENCY_BONUS_SCALE:-0.5}"
STATE_GROUP_ADVANTAGE_MODE="${STATE_GROUP_ADVANTAGE_MODE:-mean_then_batch_whiten}"
MIN_EFFECTIVE_STATE_GROUPS="${MIN_EFFECTIVE_STATE_GROUPS:-1}"
COMPACT_STATE_GROUP_ROWS="${COMPACT_STATE_GROUP_ROWS:-$DEFAULT_COMPACT_STATE_GROUP_ROWS}"
SMOKE="${SMOKE:-0}"
case "$N_GPUS" in
    2)
        TP_SIZE="${TP_SIZE:-1}"
        SP_SIZE="${SP_SIZE:-2}"
        PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-16384}"
        LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-16384}"
        ;;
    8)
        TP_SIZE="${TP_SIZE:-2}"
        SP_SIZE="${SP_SIZE:-4}"
        PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-8192}"
        LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-8192}"
        ;;
    *)
        : "${TP_SIZE:?Set TP_SIZE when N_GPUS is neither 2 nor 8}"
        : "${SP_SIZE:?Set SP_SIZE when N_GPUS is neither 2 nor 8}"
        : "${PPO_MAX_TOKENS_PER_GPU:?Set PPO_MAX_TOKENS_PER_GPU for this GPU profile}"
        : "${LOGPROB_MAX_TOKENS_PER_GPU:?Set LOGPROB_MAX_TOKENS_PER_GPU for this GPU profile}"
        ;;
esac


if [[ "$TEACHER_REWARD_MODE" != "appearance" && "$TEACHER_REWARD_MODE" != "frequency_weighted" ]]; then
    echo "ERROR: TEACHER_REWARD_MODE must be appearance or frequency_weighted" >&2
    exit 1
fi
if [[ "$STATE_GROUP_ADVANTAGE_MODE" != "group_whiten" && "$STATE_GROUP_ADVANTAGE_MODE" != "mean_then_batch_whiten" ]]; then
    echo "ERROR: STATE_GROUP_ADVANTAGE_MODE must be group_whiten or mean_then_batch_whiten" >&2
    exit 1
fi
if [[ "$DATA_TRUNCATION" != "error" && "$DATA_TRUNCATION" != "left" ]]; then
    echo "ERROR: DATA_TRUNCATION must be error or left" >&2
    exit 1
fi
if [[ ! "$MIN_EFFECTIVE_STATE_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MIN_EFFECTIVE_STATE_GROUPS must be a positive integer" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import math, sys; value=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value >= 0 else 1)' "$FREQUENCY_BONUS_SCALE"; then
    echo "ERROR: FREQUENCY_BONUS_SCALE must be finite and non-negative" >&2
    exit 1
fi

: "${MODEL_PATH:?Set MODEL_PATH to the student model directory}"
: "${TAU_USER_MODEL:?Set TAU_USER_MODEL to the served user-simulator model name}"
: "${TAU_USER_API_BASE:?Set TAU_USER_API_BASE to the OpenAI-compatible user endpoint}"
if [[ "$METHOD" == "agentic_opd" ]]; then
    : "${TAU_TEACHER_MODEL:?Set TAU_TEACHER_MODEL to the served teacher model name}"
    : "${TAU_TEACHER_API_BASE:?Set TAU_TEACHER_API_BASE to the OpenAI-compatible teacher endpoint}"
fi
export TAU_USER_API_KEY="${TAU_USER_API_KEY:-EMPTY}"
export TAU_TEACHER_API_KEY="${TAU_TEACHER_API_KEY:-EMPTY}"
TAU_USER_API_HOST="${TAU_USER_API_BASE#*://}"
TAU_USER_API_HOST="${TAU_USER_API_HOST%%[:/]*}"
TAU_NO_PROXY_HOSTS="$TAU_USER_API_HOST"
if [[ "$METHOD" == "agentic_opd" ]]; then
    TAU_TEACHER_API_HOST="${TAU_TEACHER_API_BASE#*://}"
    TAU_TEACHER_API_HOST="${TAU_TEACHER_API_HOST%%[:/]*}"
    TAU_NO_PROXY_HOSTS="$TAU_NO_PROXY_HOSTS,$TAU_TEACHER_API_HOST"
fi
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$TAU_NO_PROXY_HOSTS"
export no_proxy="$NO_PROXY"
if [[ "$TAU_USER_API_KEY_ENV" != "TAU_USER_API_KEY" ]]; then
    [[ -n "${!TAU_USER_API_KEY_ENV:-}" ]] || {
        echo "ERROR: missing $TAU_USER_API_KEY_ENV" >&2
        exit 1
    }
fi
if [[ "$METHOD" == "agentic_opd" && "$TAU_TEACHER_API_KEY_ENV" != "TAU_TEACHER_API_KEY" ]]; then
    [[ -n "${!TAU_TEACHER_API_KEY_ENV:-}" ]] || {
        echo "ERROR: missing $TAU_TEACHER_API_KEY_ENV" >&2
        exit 1
    }
fi
for path in "$MODEL_PATH" "$TAU2_ROOT" "$TAU2_DATA_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done
if ! TAU2_DATA_DIR="$TAU2_DATA_DIR" "$PYTHON" -c 'import tau2; import rank_bm25' >/dev/null 2>&1; then
    echo "ERROR: Tau dependencies are incomplete; run PYTHON=$PYTHON bash examples/tau_bench/install_tau2.sh" >&2
    exit 1
fi
check_openai_endpoint() {
    local label="$1"
    local api_base="$2"
    local api_key_env="$3"
    local api_key="${!api_key_env}"
    if ! curl --noproxy '*' -fsS --max-time 15 \
        -H "Authorization: Bearer $api_key" \
        "${api_base%/}/models" >/dev/null; then
        echo "ERROR: $label endpoint is unavailable: $api_base" >&2
        exit 1
    fi
}
check_openai_endpoint "Tau user simulator" "$TAU_USER_API_BASE" "$TAU_USER_API_KEY_ENV"
if [[ "$METHOD" == "agentic_opd" ]]; then
    check_openai_endpoint "Tau teacher" "$TAU_TEACHER_API_BASE" "$TAU_TEACHER_API_KEY_ENV"
fi
if [[ "$METHOD" == "agentic_opd" && "$TAU_MASK_MATCHER_REQUIRED_GROUPS" == false ]]; then
    [[ -n "${!TAU_MATCHER_API_KEY_ENV:-}" ]] || {
        echo "ERROR: missing $TAU_MATCHER_API_KEY_ENV for Tau matcher" >&2
        exit 1
    }
    if [[ "$TAU_MATCHER_PROVIDER" != "openai-compatible" && "$TAU_MATCHER_PROVIDER" != "deepseek" ]]; then
        echo "ERROR: TAU_MATCHER_PROVIDER must be openai-compatible or deepseek" >&2
        exit 1
    fi
    if [[ "$TAU_MATCHER_API_BASE" != "$TAU_TEACHER_API_BASE" || "$TAU_MATCHER_API_KEY_ENV" != "$TAU_TEACHER_API_KEY_ENV" ]]; then
        check_openai_endpoint "Tau matcher" "$TAU_MATCHER_API_BASE" "$TAU_MATCHER_API_KEY_ENV"
    fi
fi

if [[ "$SMOKE" != "1" ]] && (( AIRLINE_TRAJ + RETAIL_TRAJ != 16 )); then
    echo "ERROR: formal training requires AIRLINE_TRAJ + RETAIL_TRAJ = 16" >&2
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
    AIRLINE_TRAJ="${SMOKE_AIRLINE_TRAJ:-1}"
    RETAIL_TRAJ="${SMOKE_RETAIL_TRAJ:-1}"
    TRAIN_STEPS=1
    PPO_MINI_BATCH="${SMOKE_PPO_MINI_BATCH:-8}"
    TRAIN_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
    EVAL_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
    SAVE_FREQ=-1
    TEST_FREQ=-1
    VAL_BATCH=2
    VALIDATION_NUM_TASKS=2
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
mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/cache" "$RUN_DIR/tensorboard" "$DATA_DIR"
VALIDATION_ARGS=(
    --validation-domains "$VALIDATION_DOMAINS"
    --validation-split "$VALIDATION_SPLIT"
    --validation-trials "$VALIDATION_TRIALS"
    --validation-batch-size "$VAL_BATCH"
)
if [[ -n "$VALIDATION_NUM_TASKS" ]]; then
    VALIDATION_ARGS+=(--validation-num-tasks "$VALIDATION_NUM_TASKS")
fi
"$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
    --output-dir "$DATA_DIR" \
    --source-root "$TAU2_ROOT" \
    --train-steps "$TRAIN_STEPS" \
    --airline "$AIRLINE_TRAJ" \
    --retail "$RETAIL_TRAJ" \
    "${VALIDATION_ARGS[@]}"

read -r TAU_VAL_AIRLINE TAU_VAL_RETAIL < <("$PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1]))[\"validation_plan\"][\"counts\"]; print(c[\"airline\"], c[\"retail\"])" "$DATA_DIR/manifest.json")

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
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

ORACLE_OVERRIDES=()
if [[ "$METHOD" == "agentic_opd" ]]; then
    ORACLE_OVERRIDES=(
        "env.tau.oracle.cache_path=$ORACLE_CACHE"
        "env.tau.oracle.matcher_cache_path=$ORACLE_MATCHER_CACHE"
        "env.tau.oracle.matcher_provider=$TAU_MATCHER_PROVIDER"
        "env.tau.oracle.matcher_model=$TAU_MATCHER_MODEL"
        "env.tau.oracle.matcher_api_base=$TAU_MATCHER_API_BASE"
        "env.tau.oracle.matcher_api_key_env=$TAU_MATCHER_API_KEY_ENV"
        "env.tau.oracle.matcher_enable_thinking=$TAU_MATCHER_ENABLE_THINKING"
        "env.tau.oracle.matcher_max_tokens=$TAU_MATCHER_MAX_TOKENS"
        "env.tau.oracle.matcher_reasoning_effort=$TAU_MATCHER_REASONING_EFFORT"
        "env.tau.oracle.matcher_max_concurrent_requests=$TAU_MATCHER_MAX_CONCURRENT_REQUESTS"
        "env.tau.transfer_reward_guard_enabled=$TAU_TRANSFER_REWARD_GUARD"
        "env.tau.mask_matcher_required_groups=$TAU_MASK_MATCHER_REQUIRED_GROUPS"
        "env.tau.oracle.model=$TAU_TEACHER_MODEL"
        "env.tau.oracle.api_base=$TAU_TEACHER_API_BASE"
        "env.tau.oracle.api_key_env=$TAU_TEACHER_API_KEY_ENV"
        "env.tau.oracle.temperature=$TAU_TEACHER_TEMPERATURE"
        "env.tau.oracle.top_p=$TAU_TEACHER_TOP_P"
        "env.tau.oracle.top_k=$TAU_TEACHER_TOP_K"
        "env.tau.oracle.min_p=$TAU_TEACHER_MIN_P"
        "env.tau.oracle.max_tokens=$TAU_TEACHER_MAX_TOKENS"
        "env.tau.oracle.teacher_validity_max_retries=$TAU_TEACHER_VALIDITY_MAX_RETRIES"
        "env.tau.oracle.use_privileged_context=$TAU_USE_PRIVILEGED_TEACHER_CONTEXT"
    )
fi

METHOD_OVERRIDES=(
    "reward_model.reward_manager=turn"
    "reward_model.overlong_buffer.enable=False"
    "actor_rollout_ref.actor.optim.weight_decay=0.01"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.log_entropy_metrics=False"
    "actor_rollout_ref.actor.log_sampled_entropy_metrics=True"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.2"
    "actor_rollout_ref.actor.clip_ratio_c=3.0"
)
if [[ "$METHOD" == "outcome" ]]; then
    METHOD_OVERRIDES+=("algorithm.adv_estimator=grpo")
else
    METHOD_OVERRIDES+=("algorithm.adv_estimator=dapo")
fi

echo "Tau $METHOD run: $RUN_DIR"
echo "Committed task groups: Airline=$AIRLINE_TRAJ Retail=$RETAIL_TRAJ; group size=$ROLLOUT_N"
echo "Mask matcher-required groups: $TAU_MASK_MATCHER_REQUIRED_GROUPS (whole-group loss mask; uniform valid commit)"
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
    data.truncation="$DATA_TRUNCATION" \
    data.return_raw_chat=True \
    data.shuffle=False \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    "${METHOD_OVERRIDES[@]}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.state_group.advantage_mode="$STATE_GROUP_ADVANTAGE_MODE" \
    algorithm.state_group.min_effective_groups="$MIN_EFFECTIVE_STATE_GROUPS" \
    algorithm.state_group.compact_policy_rows="$COMPACT_STATE_GROUP_ROWS" \
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
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    env.seed=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.teacher_reward.mode="$TEACHER_REWARD_MODE" \
    env.teacher_reward.frequency_bonus_scale="$FREQUENCY_BONUS_SCALE" \
    env.tau.source_root="$TAU2_ROOT" \
    env.tau.user_llm="$TAU_USER_MODEL" \
    env.tau.user_api_base="$TAU_USER_API_BASE" \
    env.tau.user_api_key_env="$TAU_USER_API_KEY_ENV" \
    env.tau.native_log_level="$TAU_NATIVE_LOG_LEVEL" \
    env.tau.train_max_steps="$TRAIN_MAX_STEPS" \
    env.tau.eval_max_steps="$EVAL_MAX_STEPS" \
    env.tau.trajectory_counts.airline="$AIRLINE_TRAJ" \
    env.tau.trajectory_counts.retail="$RETAIL_TRAJ" \
    env.tau.validation_counts.airline="$TAU_VAL_AIRLINE" \
    env.tau.validation_counts.retail="$TAU_VAL_RETAIL" \
    "${ORACLE_OVERRIDES[@]}" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
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
    +ray_init.num_cpus="$RAY_CPUS" \
    "$@" 2>&1 | tee "$LOG_FILE"
