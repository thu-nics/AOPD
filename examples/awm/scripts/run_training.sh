#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/paths.sh"
VARIANT="${VARIANT:?Set VARIANT to semantic or outcome}"
if [[ "$VARIANT" != "semantic" && "$VARIANT" != "outcome" ]]; then
    echo "ERROR: VARIANT must be semantic or outcome" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-/mnt/public2/yuanhuining/models/Qwen3-4B}"
AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/awm}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_STAMP}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"
TRAIN_SPLIT="${TRAIN_SPLIT:-all}"
VAL_SPLIT="${VAL_SPLIT:-smoke}"
TRAIN_DATA="${TRAIN_DATA:-}"
TRAIN_SELECTION_MANIFEST="${TRAIN_SELECTION_MANIFEST:-}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
TRAIN_TASK_COUNT="${TRAIN_TASK_COUNT:-}"
TRAIN_TASK_FRACTION="${TRAIN_TASK_FRACTION:-}"
DETERMINISTIC_FILTER_DIR="${DETERMINISTIC_FILTER_DIR:-$REPO_ROOT/runs/awm_deterministic_filter}"
USE_RAW_SPLIT="${USE_RAW_SPLIT:-0}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
VAL_BATCH="${VAL_BATCH:-8}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO="${PPO_MICRO:-1}"
LOGPROB_MICRO="${LOGPROB_MICRO:-1}"
PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-16384}"
LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-32768}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"
N_GPUS="${N_GPUS:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-25}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
SMOKE="${SMOKE:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"
SHUFFLE="${SHUFFLE:-true}"

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ "$VARIANT" == "semantic" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required for semantic training" >&2
    echo "Launch this script from the configured tmux session deepseek_api." >&2
    exit 1
fi
if (( N_GPUS % TP_SIZE != 0 || N_GPUS % SP_SIZE != 0 )); then
    echo "ERROR: N_GPUS must be divisible by TP_SIZE and SP_SIZE" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import agent_world_model_env, openenv' >/dev/null 2>&1; then
    echo "ERROR: AWM dependencies are missing; run examples/awm/scripts/install_awm.sh" >&2
    exit 1
fi
if ! "$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
    --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
    >/dev/null 2>&1; then
    echo "ERROR: AWM server is not healthy at $AWM_BASE_URL" >&2
    echo "Start it with examples/awm/scripts/start_server.sh to enable pinned logical time." >&2
    exit 1
fi

mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/cache" "$DATA_DIR"
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    "$PYTHON" "$SCRIPT_DIR/../cli/prepare_data.py" \
        --data-dir "$AWM_DATA_DIR" --output-dir "$DATA_DIR" --local-files-only
fi
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_STEPS=1
    TRAIN_SPLIT=smoke
    VAL_SPLIT=smoke
    TRAIN_BATCH=4
    VAL_BATCH=4
    PPO_MINI_BATCH=4
    SAVE_FREQ=-1
    TEST_FREQ=-1
fi

if [[ "$USE_RAW_SPLIT" != "0" && "$USE_RAW_SPLIT" != "1" ]]; then
    echo "ERROR: USE_RAW_SPLIT must be 0 or 1" >&2
    exit 1
fi
if [[ "$VAL_BEFORE_TRAIN" != "true" && "$VAL_BEFORE_TRAIN" != "false" ]]; then
    echo "ERROR: VAL_BEFORE_TRAIN must be true or false" >&2
    exit 1
fi
if [[ -n "$TRAIN_TASK_COUNT" && -n "$TRAIN_TASK_FRACTION" ]]; then
    echo "ERROR: set only one of TRAIN_TASK_COUNT or TRAIN_TASK_FRACTION" >&2
    exit 1
fi
if [[ "$SMOKE" != "1" && -z "$TRAIN_DATA" && "$USE_RAW_SPLIT" == "0" ]]; then
    TRAIN_DATA="$DETERMINISTIC_FILTER_DIR/awm_training_pool.parquet"
    TRAIN_SELECTION_MANIFEST="$DETERMINISTIC_FILTER_DIR/integrity_manifest.json"
fi

"$PYTHON" "$SCRIPT_DIR/../cli/prepare_data.py" \
    --data-dir "$AWM_DATA_DIR" \
    --output-dir "$DATA_DIR" \
    --local-files-only \
    --verify-only
VAL_FILE="$DATA_DIR/awm_${VAL_SPLIT}.parquet"
if [[ -n "$TRAIN_DATA" ]]; then
    if [[ -z "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: TRAIN_SELECTION_MANIFEST is required with TRAIN_DATA" >&2
        exit 1
    fi
    if [[ ! -f "$TRAIN_DATA" || ! -f "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: deterministic training pool is missing; run run_selection.sh and run_integrity_audit.sh first" >&2
        exit 1
    fi
    "$PYTHON" "$SCRIPT_DIR/../cli/verify_training_pool.py" \
        --data "$TRAIN_DATA" \
        --manifest "$TRAIN_SELECTION_MANIFEST"
    TRAIN_FILE="$TRAIN_DATA"
    if [[ -n "$TRAIN_TASK_COUNT" || -n "$TRAIN_TASK_FRACTION" ]]; then
        SLICE_DATA="$RUN_DIR/data/awm_training_pool_slice.parquet"
        SLICE_MANIFEST="$RUN_DIR/data/training_slice_manifest.json"
        slice_args=()
        if [[ -n "$TRAIN_TASK_COUNT" ]]; then
            slice_args+=(--task-count "$TRAIN_TASK_COUNT")
        else
            slice_args+=(--fraction "$TRAIN_TASK_FRACTION")
        fi
        "$PYTHON" "$SCRIPT_DIR/../cli/slice_training_pool.py" \
            --data "$TRAIN_DATA" \
            --manifest "$TRAIN_SELECTION_MANIFEST" \
            --output-data "$SLICE_DATA" \
            --output-manifest "$SLICE_MANIFEST" \
            "${slice_args[@]}"
        TRAIN_FILE="$SLICE_DATA"
        TRAIN_SELECTION_MANIFEST="$SLICE_MANIFEST"
    fi
else
    if [[ -n "$TRAIN_TASK_COUNT" || -n "$TRAIN_TASK_FRACTION" ]]; then
        echo "ERROR: training-task slicing requires a verified deterministic pool" >&2
        exit 1
    fi
    TRAIN_FILE="$DATA_DIR/awm_${TRAIN_SPLIT}.parquet"
fi
for path in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: prepared AWM split does not exist: $path" >&2
        exit 1
    fi
done
TASK_COUNT="$("$PYTHON" -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" )"
if (( TASK_COUNT % TRAIN_BATCH != 0 )); then
    echo "WARNING: each shuffled epoch drops $((TASK_COUNT % TRAIN_BATCH)) tail task(s) to keep full batches" >&2
fi
STEPS_PER_EPOCH=$((TASK_COUNT / TRAIN_BATCH))
if (( STEPS_PER_EPOCH <= 0 )); then
    echo "ERROR: training pool must contain at least TRAIN_BATCH=$TRAIN_BATCH tasks" >&2
    exit 1
fi
if [[ -z "$TRAIN_STEPS" ]]; then
    TRAIN_STEPS="$STEPS_PER_EPOCH"
fi
if (( TRAIN_STEPS <= 0 )); then
    echo "ERROR: TRAIN_STEPS must be positive" >&2
    exit 1
fi
TRAIN_EPOCHS=$(((TRAIN_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH))

export AWM_DATA_DIR TENSORBOARD_DIR TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CONFIG_NAME="awm_${VARIANT}"
LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then LOGGER='["console"]'; fi

echo "AWM $VARIANT run: $RUN_DIR"
echo "Training split tasks=$TASK_COUNT batch=$TRAIN_BATCH steps=$TRAIN_STEPS epochs=$TRAIN_EPOCHS"
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length=29952 \
    data.max_response_length=2048 \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle="$SHUFFLE" \
    +data.dataloader_num_workers=0 \
    data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len=32000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    env.awm.base_url="$AWM_BASE_URL" \
    env.awm.oracle.cache_path="$RUN_DIR/cache/teacher.jsonl" \
    env.awm.oracle.matcher_cache_path="$RUN_DIR/cache/matcher.jsonl" \
    env.awm.runtime_quarantine.path="$RUN_DIR/runtime_quarantine.jsonl" \
    env.rollout.n=4 \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_EPOCHS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=awm \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.logger="$LOGGER" \
    trainer.resume_mode="$RESUME_MODE" \
    hydra.run.dir="$RUN_DIR/hydra" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
