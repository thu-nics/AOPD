#!/usr/bin/env bash
# Evaluate arbitrary local/HF or VERL FSDP models with the native tau2 runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)"

PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
VLLM_BIN="${VLLM_BIN:-/opt/venv/verl-agent/bin/vllm}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/.cache/tau2-bench-17e07b1}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/tau_native_eval_$RUN_ID}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$REPO_ROOT/runs/tau_native_model_cache}"

DOMAINS="${DOMAINS:-airline retail telecom}"
NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_TASKS="${NUM_TASKS:-}"
TASKS_PER_SHARD="${TASKS_PER_SHARD:-100}"
SEED="${SEED:-300}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
SIMULATION_TIMEOUT="${SIMULATION_TIMEOUT:-1800}"
TASK_RETRIES="${TASK_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"
LLM_RETRIES="${LLM_RETRIES:-6}"

AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
AGENT_TOP_P="${AGENT_TOP_P:-0.95}"
AGENT_TOP_K="${AGENT_TOP_K:-20}"
AGENT_MIN_P="${AGENT_MIN_P:-0.0}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4096}"
AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING:-true}"
AGENT_PROTOCOL="${AGENT_PROTOCOL:-strict_native}"
TRAINING_DECISION_LIMIT="${TRAINING_DECISION_LIMIT:-200}"
TRAINING_INVALID_ACTION_LIMIT="${TRAINING_INVALID_ACTION_LIMIT:-10}"
USER_MODEL="${USER_MODEL:-openrouter/qwen/qwen3.6-27b}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
GPU_BUSY_MEMORY_MIB="${GPU_BUSY_MEMORY_MIB:-2048}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
LOCAL_API_KEY="${LOCAL_API_KEY:-local-tau-eval}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE="${ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE:-0}"

SERVER_PID=""
TEMP_MODEL_DIR=""
PREPARED_MODEL_PATH=""
PREPARED_SOURCE_IDENTITY=""

usage() {
    cat <<'EOF'
Usage:
  MODEL_SPECS_FILE=/path/to/models.tsv \
    bash examples/tau_bench/run_tau_native_eval.sh

Registry columns (tab-separated):
  model_id    model_path    optional_global_step_checkpoint

Use "-" in the third column for an ordinary Hugging Face model. A checkpoint
must be a VERL global_step_* directory containing actor FSDP shards.

Defaults:
  Domains: Airline base (50), Retail base (114), Telecom base (114).
  Trials: 3 independent trials per task, seed base 300.
  Scoring: native tau2 ENV/ACTION/COMMUNICATE criteria only; no NL judge.
  Agent: local vLLM, temperature=0.6, top_p=0.95, top_k=20, min_p=0,
         thinking enabled, 4096 output tokens per decision.
  Protocol: strict_native by default; set AGENT_PROTOCOL=training_compatible
            to use the training prompt and raw action parser.
  User: openrouter/qwen/qwen3.6-27b, temperature=0, reasoning disabled.
  Serving: 8 GPUs, TP=1, DP=8, 32 concurrent simulations.

The evaluator never stops another process. It waits for the selected GPUs by
default. Native result files are split into 100-task shards to avoid quadratic
checkpoint I/O on Telecom. Re-running the same command and RUN_DIR resumes all
completed trials.

Useful overrides:
  RUN_DIR, MODEL_FILTER, DOMAINS, NUM_TASKS, TASKS_PER_SHARD,
  MAX_CONCURRENCY, CUDA_VISIBLE_DEVICES, N_GPUS, TP_SIZE, DP_SIZE,
  WAIT_FOR_FREE_GPUS, AGENT_PROTOCOL, TRAINING_DECISION_LIMIT,
  TRAINING_INVALID_ACTION_LIMIT, AGENT_MAX_TOKENS, MAX_MODEL_LEN, DRY_RUN,
  ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE (one-time migration of a compatible v2 run).
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

contains_word() {
    local haystack="$1"
    local needle="$2"
    [[ -z "$haystack" || " $haystack " == *" $needle "* ]]
}

abspath() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$REPO_ROOT/$path"
    fi
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping vLLM server pid=$SERVER_PID"
        kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
    if [[ -n "$TEMP_MODEL_DIR" && -d "$TEMP_MODEL_DIR" ]]; then
        rm -rf "$TEMP_MODEL_DIR"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] || {
    usage
    exit 0
}
[[ -n "$MODEL_SPECS_FILE" ]] || {
    usage >&2
    die "MODEL_SPECS_FILE is required"
}

MODEL_SPECS_FILE="$(abspath "$MODEL_SPECS_FILE")"
RUN_DIR="$(abspath "$RUN_DIR")"
MODEL_CACHE_ROOT="$(abspath "$MODEL_CACHE_ROOT")"
TAU2_ROOT="$(abspath "$TAU2_ROOT")"
TAU2_DATA_DIR="$(abspath "$TAU2_DATA_DIR")"

[[ -f "$MODEL_SPECS_FILE" ]] || die "model registry not found: $MODEL_SPECS_FILE"
[[ -x "$PYTHON" ]] || die "Python not found: $PYTHON"
[[ -x "$VLLM_BIN" ]] || die "vLLM not found: $VLLM_BIN"
[[ -d "$TAU2_ROOT/src/tau2" ]] || die "tau2 source not found: $TAU2_ROOT"
[[ -d "$TAU2_DATA_DIR" ]] || die "tau2 data not found: $TAU2_DATA_DIR"
if [[ "$DRY_RUN" != 1 ]]; then
    : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required for the user simulator}"
fi

for value_name in NUM_TRIALS TASKS_PER_SHARD MAX_STEPS MAX_ERRORS MAX_CONCURRENCY \
    TRAINING_DECISION_LIMIT TRAINING_INVALID_ACTION_LIMIT N_GPUS TP_SIZE DP_SIZE \
    MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$value_name must be positive"
done
if [[ -n "$NUM_TASKS" ]]; then
    [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ ]] || die "NUM_TASKS must be positive"
fi
(( TP_SIZE * DP_SIZE == N_GPUS )) ||
    die "TP_SIZE * DP_SIZE must equal N_GPUS"
case "${AGENT_ENABLE_THINKING,,}" in
    true | 1 | yes) AGENT_ENABLE_THINKING=true ;;
    false | 0 | no) AGENT_ENABLE_THINKING=false ;;
    *) die "AGENT_ENABLE_THINKING must be true or false" ;;
esac
case "$ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE" in
    0 | 1) ;;
    *) die "ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE must be 0 or 1" ;;
esac
case "$AGENT_PROTOCOL" in
    strict_native | training_compatible) ;;
    *) die "AGENT_PROTOCOL must be strict_native or training_compatible" ;;
esac

read -r -a DOMAIN_ARGS <<< "$DOMAINS"
(( ${#DOMAIN_ARGS[@]} > 0 )) || die "DOMAINS must not be empty"
for domain in "${DOMAIN_ARGS[@]}"; do
    case "$domain" in
        airline | retail | telecom) ;;
        *) die "unsupported domain: $domain" ;;
    esac
done

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_IDS[@]} == N_GPUS )) ||
    die "CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} devices, expected $N_GPUS"
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] ||
        die "GPU waiting requires numeric CUDA device IDs: $gpu_id"
done

declare -a MODEL_IDS=()
declare -a MODEL_PATHS=()
declare -a CHECKPOINT_PATHS=()
while IFS=$'\t' read -r model_id model_path checkpoint_path extra; do
    [[ -n "$model_id" && "${model_id:0:1}" != "#" ]] || continue
    [[ -z "${extra:-}" ]] ||
        die "registry row has more than 3 columns: $model_id"
    [[ "$model_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
        die "unsafe model id: $model_id"
    [[ -n "$model_path" ]] || die "missing model path for $model_id"
    checkpoint_path="${checkpoint_path:--}"
    if contains_word "$MODEL_FILTER" "$model_id"; then
        MODEL_IDS+=("$model_id")
        MODEL_PATHS+=("$(abspath "$model_path")")
        if [[ "$checkpoint_path" == "-" ]]; then
            CHECKPOINT_PATHS+=("-")
        else
            CHECKPOINT_PATHS+=("$(abspath "$checkpoint_path")")
        fi
    fi
done < "$MODEL_SPECS_FILE"
(( ${#MODEL_IDS[@]} > 0 )) || die "no models selected from $MODEL_SPECS_FILE"

validate_model_source() {
    local model_id="$1"
    local model_path="$2"
    local checkpoint_path="$3"
    [[ -d "$model_path" ]] || die "base model path not found for $model_id: $model_path"
    [[ -f "$model_path/config.json" ]] ||
        die "base model config missing for $model_id: $model_path"
    if [[ "$checkpoint_path" == "-" ]]; then
        compgen -G "$model_path/*.safetensors" >/dev/null ||
            compgen -G "$model_path/*.bin" >/dev/null ||
            die "HF weights missing for $model_id: $model_path"
        return
    fi

    local actor_dir="$checkpoint_path/actor"
    [[ -d "$actor_dir" ]] ||
        die "checkpoint actor directory missing for $model_id: $actor_dir"
    local rank_zero world_size shard_count
    rank_zero="$(find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' -print -quit)"
    [[ -n "$rank_zero" ]] || die "no FSDP rank-zero shard for $model_id"
    world_size="$(basename "$rank_zero" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
    [[ "$world_size" =~ ^[1-9][0-9]*$ ]] ||
        die "cannot parse FSDP world size for $model_id"
    shard_count="$(find "$actor_dir" -maxdepth 1 -type f -name "model_world_size_${world_size}_rank_*.pt" | wc -l)"
    [[ "$shard_count" -eq "$world_size" ]] ||
        die "checkpoint has $shard_count/$world_size FSDP shards for $model_id"
}

source_identity() {
    local model_path="$1"
    local checkpoint_path="$2"
    local scan_path="$model_path"
    {
        printf 'base_model=%s\n' "$(realpath "$model_path")"
        printf 'checkpoint=%s\n' "$checkpoint_path"
        if [[ "$checkpoint_path" != "-" ]]; then
            scan_path="$checkpoint_path/actor"
            printf 'checkpoint_realpath=%s\n' "$(realpath "$checkpoint_path")"
            printf 'merger_sha256=%s\n' "$(sha256sum "$REPO_ROOT/scripts/model_merger.py" | awk '{print $1}')"
        fi
        find "$scan_path" -maxdepth 1 -type f -regextype posix-extended \
            -regex '.*/(model.*[.]safetensors.*|.*[.]bin|model_world_size_[^/]+|config[.]json|generation_config[.]json|tokenizer[^/]*|special_tokens_map[.]json|chat_template[^/]*|merges[.]txt|vocab[.]json|added_tokens[.]json)' \
            -print0 | sort -z | xargs -0 -r stat -c '%n|%s|%y'
    } | sha256sum | awk '{print $1}'
}

prepare_model() {
    local model_id="$1"
    local model_path="$2"
    local checkpoint_path="$3"
    PREPARED_SOURCE_IDENTITY="$(source_identity "$model_path" "$checkpoint_path")"
    if [[ "$checkpoint_path" == "-" ]]; then
        PREPARED_MODEL_PATH="$model_path"
        return
    fi

    local cache_dir="$MODEL_CACHE_ROOT/$model_id/$PREPARED_SOURCE_IDENTITY"
    local marker="$cache_dir/.merge_complete"
    if [[ -f "$marker" && -f "$cache_dir/config.json" ]] &&
        compgen -G "$cache_dir/*.safetensors" >/dev/null; then
        PREPARED_MODEL_PATH="$cache_dir"
        return
    fi
    [[ ! -e "$cache_dir" ]] || die "incomplete merged-model cache: $cache_dir"

    TEMP_MODEL_DIR="$cache_dir.tmp.$$"
    mkdir -p "$(dirname "$cache_dir")" "$TEMP_MODEL_DIR"
    log "Merging $model_id into $cache_dir"
    "$PYTHON" "$REPO_ROOT/scripts/model_merger.py" merge \
        --backend fsdp \
        --local_dir "$checkpoint_path/actor" \
        --target_dir "$TEMP_MODEL_DIR"
    [[ -f "$TEMP_MODEL_DIR/config.json" ]] ||
        die "merge produced no config.json for $model_id"
    compgen -G "$TEMP_MODEL_DIR/*.safetensors" >/dev/null ||
        die "merge produced no safetensors for $model_id"
    {
        printf 'model_id=%s\n' "$model_id"
        printf 'source=%s\n' "$(realpath "$checkpoint_path")"
        printf 'source_identity=%s\n' "$PREPARED_SOURCE_IDENTITY"
        printf 'merged_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$TEMP_MODEL_DIR/.merge_complete"
    mv "$TEMP_MODEL_DIR" "$cache_dir"
    TEMP_MODEL_DIR=""
    PREPARED_MODEL_PATH="$cache_dir"
}

selected_gpus_are_free() {
    local gpu_id used pids
    for gpu_id in "${GPU_IDS[@]}"; do
        pids="$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        [[ ! "$pids" =~ [0-9] ]] || return 1
        used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$used" =~ ^[0-9]+$ ]] ||
            die "cannot query memory for GPU $gpu_id"
        (( used < GPU_BUSY_MEMORY_MIB )) || return 1
    done
}

wait_for_selected_gpus() {
    [[ "$DRY_RUN" == 1 || "$WAIT_FOR_FREE_GPUS" == 0 ]] && return
    command -v nvidia-smi >/dev/null ||
        die "nvidia-smi is required for GPU waiting"
    while ! selected_gpus_are_free; do
        log "Selected GPUs are busy; waiting $GPU_POLL_SECONDS seconds"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true
        sleep "$GPU_POLL_SECONDS"
    done
    log "Selected GPUs are free: $CUDA_VISIBLE_DEVICES"
}

protocol_without_nl_fix_fields() {
    grep -vE '^(PROTOCOL_VERSION|EVALUATION_PROTOCOL|EVALUATOR_SHA256|DRIVER_SHA256|LAUNCHER_SHA256)=' "$1"
}

write_protocol() {
    mkdir -p "$RUN_DIR"
    local candidate="$RUN_DIR/protocol.env.new"
    local tau_revision
    tau_revision="$(git -C "$TAU2_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
    {
        echo "PROTOCOL_VERSION=3"
        printf 'EVALUATION_PROTOCOL=%s\n' "tau_all_without_nl_assertions_v1"
        printf 'AGENT_PROTOCOL=%s\n' "$AGENT_PROTOCOL"
        printf 'TRAINING_DECISION_LIMIT=%s\n' "$TRAINING_DECISION_LIMIT"
        printf 'TRAINING_INVALID_ACTION_LIMIT=%s\n' "$TRAINING_INVALID_ACTION_LIMIT"
        printf 'MODEL_SPECS_SHA256=%s\n' "$(sha256sum "$MODEL_SPECS_FILE" | awk '{print $1}')"
        printf 'MODEL_FILTER=%s\n' "${MODEL_FILTER:-all}"
        printf 'DOMAINS=%s\n' "$DOMAINS"
        printf 'TASK_SPLITS=airline:base,retail:base,telecom:base\n'
        printf 'NUM_TRIALS=%s\n' "$NUM_TRIALS"
        printf 'NUM_TASKS=%s\n' "${NUM_TASKS:-all}"
        printf 'TASKS_PER_SHARD=%s\n' "$TASKS_PER_SHARD"
        printf 'SEED=%s\n' "$SEED"
        printf 'MAX_STEPS=%s\n' "$MAX_STEPS"
        printf 'MAX_ERRORS=%s\n' "$MAX_ERRORS"
        printf 'MAX_CONCURRENCY=%s\n' "$MAX_CONCURRENCY"
        printf 'SIMULATION_TIMEOUT=%s\n' "$SIMULATION_TIMEOUT"
        printf 'EVALUATION_TYPE=all_no_nl_assertions\n'
        printf 'AGENT_SAMPLING=temperature:%s,top_p:%s,top_k:%s,min_p:%s,max_tokens:%s,thinking:%s\n' \
            "$AGENT_TEMPERATURE" "$AGENT_TOP_P" "$AGENT_TOP_K" \
            "$AGENT_MIN_P" "$AGENT_MAX_TOKENS" "$AGENT_ENABLE_THINKING"
        printf 'USER_SIMULATOR=model:%s,temperature:0,reasoning:false\n' "$USER_MODEL"
        printf 'VLLM=cuda:%s,n_gpus:%s,tp:%s,dp:%s,memory_util:%s,max_model_len:%s,max_batched_tokens:%s,max_num_seqs:%s\n' \
            "$CUDA_VISIBLE_DEVICES" "$N_GPUS" "$TP_SIZE" "$DP_SIZE" \
            "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" \
            "$MAX_NUM_SEQS"
        printf 'TAU2_REVISION=%s\n' "$tau_revision"
        printf 'EVALUATOR_SHA256=%s\n' "$(sha256sum "$SCRIPT_DIR/deterministic_evaluator.py" | awk '{print $1}')"
        printf 'DRIVER_SHA256=%s\n' "$(sha256sum "$SCRIPT_DIR/native_tau_eval.py" | awk '{print $1}')"
        printf 'AGENT_ADAPTER_SHA256=%s\n' "$(sha256sum "$SCRIPT_DIR/training_compatible_agent.py" | awk '{print $1}')"
        printf 'LAUNCHER_SHA256=%s\n' "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
    } > "$candidate"

    local existing="$RUN_DIR/protocol.env"
    local backup="$RUN_DIR/protocol.env.pre_nl_fix_v2"
    if [[ -f "$existing" ]]; then
        if cmp -s "$existing" "$candidate"; then
            rm -f "$candidate"
            return
        fi
        if [[ "$ALLOW_NL_ASSERTION_PROTOCOL_UPGRADE" == 1 ]] \
            && grep -qx 'PROTOCOL_VERSION=2' "$existing" \
            && grep -qx 'EVALUATION_TYPE=all_no_nl_assertions' "$existing" \
            && cmp -s <(protocol_without_nl_fix_fields "$existing") \
                <(protocol_without_nl_fix_fields "$candidate"); then
            if [[ -f "$backup" ]]; then
                cmp -s "$backup" "$existing" || {
                    rm -f "$candidate"
                    die "existing protocol migration backup does not match"
                }
            else
                cp "$existing" "$backup"
            fi
            mv "$candidate" "$existing"
            log "Migrated compatible evaluation protocol v2 to v3; backup: $backup"
            return
        fi
        diff -u "$existing" "$candidate" >&2 || true
        rm -f "$candidate"
        die "protocol differs from existing RUN_DIR; use a new directory"
    fi
    mv "$candidate" "$existing"
}

record_source_identity() {
    local model_id="$1"
    local marker="$RUN_DIR/results/$model_id/source_identity"
    mkdir -p "$(dirname "$marker")"
    if [[ -f "$marker" ]]; then
        [[ "$(cat "$marker")" == "$PREPARED_SOURCE_IDENTITY" ]] ||
            die "model weights changed for existing results: $model_id"
    else
        printf '%s\n' "$PREPARED_SOURCE_IDENTITY" > "$marker"
    fi
}

start_server() {
    local model_id="$1"
    local server_log="$RUN_DIR/results/$model_id/vllm_server.log"
    mkdir -p "$(dirname "$server_log")"
    local -a command=(
        "$VLLM_BIN" serve "$PREPARED_MODEL_PATH"
        --host "$VLLM_HOST"
        --port "$VLLM_PORT"
        --served-model-name "$model_id"
        --api-key "$LOCAL_API_KEY"
        --tensor-parallel-size "$TP_SIZE"
        --data-parallel-size "$DP_SIZE"
        --data-parallel-size-local "$DP_SIZE"
        --data-parallel-backend mp
        --dtype bfloat16
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --enable-chunked-prefill
        --enable-prefix-caching
        --generation-config vllm
        --disable-log-requests
        --uvicorn-log-level warning
    )
    if [[ "$AGENT_PROTOCOL" == strict_native ]]; then
        command+=(
            --enable-auto-tool-choice
            --tool-call-parser qwen3_xml
            --reasoning-parser qwen3
        )
    fi
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    log "Starting vLLM for $model_id (TP=$TP_SIZE DP=$DP_SIZE)"
    setsid env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        TOKENIZERS_PARALLELISM=false \
        "${command[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    local deadline=$((SECONDS + VLLM_START_TIMEOUT))
    until curl -fsS \
        -H "Authorization: Bearer $LOCAL_API_KEY" \
        "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 100 "$server_log" >&2 || true
            die "vLLM exited before becoming ready for $model_id"
        fi
        if (( SECONDS >= deadline )); then
            tail -n 100 "$server_log" >&2 || true
            die "timed out waiting for vLLM for $model_id"
        fi
        sleep 5
    done
    log "vLLM is ready for $model_id"
}

run_domain() {
    local model_id="$1"
    local domain="$2"
    local log_file="$RUN_DIR/results/$model_id/$domain/eval.log"
    local task_split=base
    mkdir -p "$(dirname "$log_file")"
    local thinking_flag="--agent-enable-thinking"
    [[ "$AGENT_ENABLE_THINKING" == true ]] ||
        thinking_flag="--no-agent-enable-thinking"
    local -a command=(
        "$PYTHON" "$SCRIPT_DIR/native_tau_eval.py" run-domain
        --run-dir "$RUN_DIR"
        --model-id "$model_id"
        --domain "$domain"
        --task-split "$task_split"
        --num-trials "$NUM_TRIALS"
        --tasks-per-shard "$TASKS_PER_SHARD"
        --seed "$SEED"
        --max-steps "$MAX_STEPS"
        --max-errors "$MAX_ERRORS"
        --max-concurrency "$MAX_CONCURRENCY"
        --simulation-timeout "$SIMULATION_TIMEOUT"
        --task-retries "$TASK_RETRIES"
        --retry-delay "$RETRY_DELAY"
        --llm-retries "$LLM_RETRIES"
        --agent-base-url "http://$VLLM_HOST:$VLLM_PORT/v1"
        --agent-protocol "$AGENT_PROTOCOL"
        --training-decision-limit "$TRAINING_DECISION_LIMIT"
        --training-invalid-action-limit "$TRAINING_INVALID_ACTION_LIMIT"
        --agent-api-key "$LOCAL_API_KEY"
        --agent-temperature "$AGENT_TEMPERATURE"
        --agent-top-p "$AGENT_TOP_P"
        --agent-top-k "$AGENT_TOP_K"
        --agent-min-p "$AGENT_MIN_P"
        --agent-max-tokens "$AGENT_MAX_TOKENS"
        "$thinking_flag"
        --user-model "$USER_MODEL"
    )
    if [[ -n "$NUM_TASKS" ]]; then
        command+=(--num-tasks "$NUM_TASKS")
    fi
    log "Evaluating $model_id/$domain with native tau2 ($AGENT_PROTOCOL)"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'TAU2_DATA_DIR=%q PYTHONPATH=%q ' "$TAU2_DATA_DIR" "$REPO_ROOT:$TAU2_ROOT/src"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    env \
        TAU2_DATA_DIR="$TAU2_DATA_DIR" \
        PYTHONPATH="$REPO_ROOT:$TAU2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        TOKENIZERS_PARALLELISM=false \
        "${command[@]}" 2>&1 | tee "$log_file"
}

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.eval.lock"
flock -n 9 || die "another evaluator is using RUN_DIR=$RUN_DIR"
write_protocol
wait_for_selected_gpus

log "Run directory: $RUN_DIR"
log "Selected models: ${#MODEL_IDS[@]}"
log "Domains: $DOMAINS; trials per task: $NUM_TRIALS"
log "Only the user simulator will call OpenRouter"

for i in "${!MODEL_IDS[@]}"; do
    model_id="${MODEL_IDS[$i]}"
    model_path="${MODEL_PATHS[$i]}"
    checkpoint_path="${CHECKPOINT_PATHS[$i]}"
    validate_model_source "$model_id" "$model_path" "$checkpoint_path"
    if [[ "$DRY_RUN" == 1 ]]; then
        PREPARED_MODEL_PATH="$model_path"
        PREPARED_SOURCE_IDENTITY="dry_run"
    else
        prepare_model "$model_id" "$model_path" "$checkpoint_path"
        record_source_identity "$model_id"
    fi
    start_server "$model_id"
    for domain in "${DOMAIN_ARGS[@]}"; do
        run_domain "$model_id" "$domain"
    done
    stop_server
done

if [[ "$DRY_RUN" != 1 ]]; then
    env \
        TAU2_DATA_DIR="$TAU2_DATA_DIR" \
        PYTHONPATH="$REPO_ROOT:$TAU2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$SCRIPT_DIR/native_tau_eval.py" summarize --run-dir "$RUN_DIR"
fi
log "All selected native Tau evaluations completed: $RUN_DIR"
