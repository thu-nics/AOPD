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
AWM_BASE_URL="${AWM_BASE_URL:-}"
AWM_HOST="${AWM_HOST:-127.0.0.1}"
AWM_PORT="${AWM_PORT:-}"
MANAGE_AWM_SERVER="${MANAGE_AWM_SERVER:-1}"
AWM_SERVER_START_TIMEOUT="${AWM_SERVER_START_TIMEOUT:-120}"
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

MAX_CKPTS="${MAX_CKPTS:-null}"
SAVE_BEFORE_VALIDATION="${SAVE_BEFORE_VALIDATION:-false}"
EXPERT_CACHE_DIR="${EXPERT_CACHE_DIR:-$RUN_DIR/cache}"
TAU2_ROOT="${TAU2_ROOT:-/mnt/public2/yuanhuining/repos/tau2-bench}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"
TAU_USER_LLM="${TAU_USER_LLM:-openrouter/qwen/qwen3.6-27b}"
TAU_VAL_DOMAINS="${TAU_VAL_DOMAINS:-airline}"
TAU_VAL_TRIALS="${TAU_VAL_TRIALS:-1}"
TAU_VAL_NUM_TASKS="${TAU_VAL_NUM_TASKS:-}"
AWM_SERVER_PID=""

stop_managed_awm_server() {
    if [[ -z "$AWM_SERVER_PID" ]]; then
        return
    fi
    if kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$AWM_SERVER_PID" 2>/dev/null || kill -TERM "$AWM_SERVER_PID" 2>/dev/null || true
        for _ in {1..40}; do
            if ! kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.25
        done
        if kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
            kill -KILL -- "-$AWM_SERVER_PID" 2>/dev/null || kill -KILL "$AWM_SERVER_PID" 2>/dev/null || true
        fi
    fi
    wait "$AWM_SERVER_PID" 2>/dev/null || true
    AWM_SERVER_PID=""
}

trap stop_managed_awm_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ "$VARIANT" == "semantic" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required for semantic training" >&2
    echo "Launch this script from the configured tmux session deepseek_api." >&2
    exit 1
fi
if [[ "$VARIANT" == "semantic" && "$TAU_USER_LLM" == openrouter/* && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is required for TAU_USER_LLM=$TAU_USER_LLM" >&2
    exit 1
fi
if (( N_GPUS % TP_SIZE != 0 || N_GPUS % SP_SIZE != 0 )); then
    echo "ERROR: N_GPUS must be divisible by TP_SIZE and SP_SIZE" >&2
    exit 1
fi
if [[ "$MANAGE_AWM_SERVER" != "0" && "$MANAGE_AWM_SERVER" != "1" ]]; then
    echo "ERROR: MANAGE_AWM_SERVER must be 0 or 1" >&2
    exit 1
fi
if [[ ! "$AWM_SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AWM_SERVER_START_TIMEOUT must be a positive integer" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import agent_world_model_env, openenv' >/dev/null 2>&1; then
    echo "ERROR: AWM dependencies are missing; run examples/awm/scripts/install_awm.sh" >&2
    exit 1
fi
if [[ "$VARIANT" == "semantic" ]] && ! TAU2_DATA_DIR="$TAU2_DATA_DIR" \
    "$PYTHON" -c 'import tau2; import rank_bm25' >/dev/null 2>&1; then
    echo "ERROR: Tau dependencies are missing; run examples/tau_bench/install_tau2.sh" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/ckpt" "$EXPERT_CACHE_DIR" "$DATA_DIR"
if [[ "$MANAGE_AWM_SERVER" == "1" ]]; then
    if [[ -n "$AWM_BASE_URL" ]]; then
        echo "ERROR: AWM_BASE_URL cannot be set when MANAGE_AWM_SERVER=1; set AWM_HOST/AWM_PORT or use MANAGE_AWM_SERVER=0" >&2
        exit 1
    fi
    AWM_PORT="$("$PYTHON" - "$AWM_HOST" "$AWM_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
requested = int(sys.argv[2]) if sys.argv[2] else 0
if requested < 0 or requested > 65535:
    raise SystemExit("AWM_PORT must be between 1 and 65535")
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.bind((host, requested))
    print(sock.getsockname()[1])
PY
)"
    AWM_BASE_URL="http://$AWM_HOST:$AWM_PORT"
    AWM_SERVER_RUN_ID="$(basename "$RUN_DIR")-$$"
    AWM_SERVER_LOG="$RUN_DIR/awm_server.log"
    printf '\n=== managed AWM server run_id=%s base_url=%s ===\n' \
        "$AWM_SERVER_RUN_ID" "$AWM_BASE_URL" >>"$AWM_SERVER_LOG"
    setsid env \
        AWM_HOST="$AWM_HOST" \
        AWM_PORT="$AWM_PORT" \
        AWM_SERVER_RUN_ID="$AWM_SERVER_RUN_ID" \
        bash "$SCRIPT_DIR/start_server.sh" \
        >>"$AWM_SERVER_LOG" 2>&1 &
    AWM_SERVER_PID=$!
    AWM_SERVER_PROTOCOL_JSON=""
    for ((attempt = 1; attempt <= AWM_SERVER_START_TIMEOUT; attempt++)); do
        if ! kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
            echo "ERROR: managed AWM server exited during startup; see $AWM_SERVER_LOG" >&2
            tail -n 80 "$AWM_SERVER_LOG" >&2 || true
            exit 1
        fi
        if AWM_SERVER_PROTOCOL_JSON="$("$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
            --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
            --expected-run-id "$AWM_SERVER_RUN_ID" --timeout 1 2>/dev/null)"; then
            break
        fi
        AWM_SERVER_PROTOCOL_JSON=""
        sleep 1
    done
    if [[ -z "$AWM_SERVER_PROTOCOL_JSON" ]]; then
        echo "ERROR: managed AWM server did not become healthy within ${AWM_SERVER_START_TIMEOUT}s; see $AWM_SERVER_LOG" >&2
        tail -n 80 "$AWM_SERVER_LOG" >&2 || true
        exit 1
    fi
    AWM_SERVER_PROTOCOL_JSON="$AWM_SERVER_PROTOCOL_JSON" "$PYTHON" - \
        "$RUN_DIR/awm_server_manifest.json" "$AWM_BASE_URL" "$AWM_SERVER_RUN_ID" <<'PY'
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
manifest = {
    "kind": "awm_managed_run_server",
    "protocol_version": 1,
    "managed": True,
    "base_url": sys.argv[2],
    "run_id": sys.argv[3],
    "log_path": "awm_server.log",
    "logical_time": json.loads(os.environ["AWM_SERVER_PROTOCOL_JSON"]),
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
    AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
    if ! "$PYTHON" "$SCRIPT_DIR/../cli/check_server.py" \
        --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
        >/dev/null 2>&1; then
        echo "ERROR: external AWM server is not healthy at $AWM_BASE_URL" >&2
        exit 1
    fi
fi
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    "$PYTHON" "$SCRIPT_DIR/../cli/prepare_data.py" \
        --data-dir "$AWM_DATA_DIR" --output-dir "$DATA_DIR" --local-files-only
fi
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_STEPS=1
    TRAIN_SPLIT=smoke
    VAL_SPLIT=smoke
    TRAIN_BATCH=4
    VAL_BATCH="${SMOKE_VAL_BATCH:-2}"
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
if [[ "$VARIANT" == "semantic" ]]; then
    TAU_VAL_DIR="$RUN_DIR/data/tau_validation"
    tau_val_args=(
        --output-dir "$TAU_VAL_DIR"
        --source-root "$TAU2_ROOT"
        --train-steps 1
        --airline 4
        --retail 4
        --validation-domains "$TAU_VAL_DOMAINS"
        --validation-trials "$TAU_VAL_TRIALS"
        --validation-batch-size "$VAL_BATCH"
    )
    if [[ -n "$TAU_VAL_NUM_TASKS" ]]; then
        tau_val_args+=(--validation-num-tasks "$TAU_VAL_NUM_TASKS")
    fi
    TAU2_DATA_DIR="$TAU2_DATA_DIR" "$PYTHON" \
        "$REPO_ROOT/examples/tau_bench/prepare_tau_training.py" "${tau_val_args[@]}"
    read -r TAU_VAL_AIRLINE TAU_VAL_RETAIL < <("$PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1]))[\"validation_plan\"][\"counts\"]; print(c[\"airline\"], c[\"retail\"])" "$TAU_VAL_DIR/manifest.json")
    VAL_FILE="$TAU_VAL_DIR/validation.parquet"
else
    VAL_FILE="$DATA_DIR/awm_${VAL_SPLIT}.parquet"
fi
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
        echo "ERROR: prepared training/validation data does not exist: $path" >&2
        exit 1
    fi
done
SOURCE_TASK_COUNT="$("$PYTHON" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "$TRAIN_FILE")"
SOURCE_STEPS_PER_EPOCH=$((SOURCE_TASK_COUNT / TRAIN_BATCH))
if (( SOURCE_STEPS_PER_EPOCH <= 0 )); then
    echo "ERROR: training pool must contain at least TRAIN_BATCH=$TRAIN_BATCH tasks" >&2
    exit 1
fi
if [[ -z "$TRAIN_STEPS" ]]; then
    TRAIN_STEPS="$SOURCE_STEPS_PER_EPOCH"
fi
if (( TRAIN_STEPS <= 0 )); then
    echo "ERROR: TRAIN_STEPS must be positive" >&2
    exit 1
fi
if [[ "$VARIANT" == "semantic" && "$SMOKE" != "1" ]]; then
    if [[ -z "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: formal semantic training requires a verified deterministic pool" >&2
        exit 1
    fi
    SCHEDULE_DATA="$RUN_DIR/data/awm_training_schedule.parquet"
    SCHEDULE_MANIFEST="$RUN_DIR/data/training_schedule_manifest.json"
    "$PYTHON" "$SCRIPT_DIR/../cli/materialize_training_schedule.py" \
        --data "$TRAIN_FILE" \
        --manifest "$TRAIN_SELECTION_MANIFEST" \
        --output-data "$SCHEDULE_DATA" \
        --output-manifest "$SCHEDULE_MANIFEST" \
        --train-steps "$TRAIN_STEPS" \
        --train-batch-size "$TRAIN_BATCH"
    TRAIN_FILE="$SCHEDULE_DATA"
fi
TASK_COUNT="$("$PYTHON" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "$TRAIN_FILE")"
if (( TASK_COUNT % TRAIN_BATCH != 0 )); then
    echo "WARNING: each epoch drops $((TASK_COUNT % TRAIN_BATCH)) tail task(s)" >&2
fi
STEPS_PER_EPOCH=$((TASK_COUNT / TRAIN_BATCH))
TRAIN_EPOCHS=$(((TRAIN_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH))

export AWM_DATA_DIR TAU2_DATA_DIR TENSORBOARD_DIR TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CONFIG_NAME="awm_${VARIANT}"
LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then LOGGER='["console"]'; fi
VALIDATION_OVERRIDES=()
if [[ "$VARIANT" == "semantic" ]]; then
    VALIDATION_OVERRIDES=(
        "env.validation.env_name=tau"
        "env.tau.source_root=$TAU2_ROOT"
        "env.tau.user_llm=$TAU_USER_LLM"
        "env.tau.validation_domains=[$TAU_VAL_DOMAINS]"
        "env.tau.validation_trials=$TAU_VAL_TRIALS"
        "env.tau.validation_counts.airline=$TAU_VAL_AIRLINE"
        "env.tau.validation_counts.retail=$TAU_VAL_RETAIL"
    )
fi


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
    env.awm.oracle.cache_path="$EXPERT_CACHE_DIR/teacher.jsonl" \
    env.awm.oracle.matcher_cache_path="$EXPERT_CACHE_DIR/matcher.jsonl" \
    env.awm.runtime_quarantine.path="$RUN_DIR/runtime_quarantine.jsonl" \
    env.rollout.n=4 \
    "${VALIDATION_OVERRIDES[@]}" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_EPOCHS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.save_before_validation="$SAVE_BEFORE_VALIDATION" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=awm \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep="$MAX_CKPTS" \
    trainer.max_critic_ckpt_to_keep="$MAX_CKPTS" \
    trainer.logger="$LOGGER" \
    trainer.resume_mode="$RESUME_MODE" \
    hydra.run.dir="$RUN_DIR/hydra" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
