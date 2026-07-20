#!/usr/bin/env bash
# Evaluate an arbitrary model manifest on ALFWorld OOD and WebShop test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)"

MODEL_MANIFEST="${MODEL_MANIFEST:-}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/eval_agentic_ood_$RUN_ID}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$REPO_ROOT/runs/eval_model_cache}"
PYTHON="${PYTHON:-${PYTHON_BIN:-/opt/venv/verl-agent/bin/python}}"
JAVA_HOME="${JAVA_HOME:-$REPO_ROOT/data/agentic_eval/java11}"
JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
GPU_BUSY_MEMORY_MIB="${GPU_BUSY_MEMORY_MIB:-2048}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
RAY_CPUS="${RAY_CPUS:-64}"
RAY_TEMP_ROOT="/tmp/vpr_agentic_ood_ray_$$"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
ENV_SEED="${ENV_SEED:-0}"

ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/agentic_eval/alfworld}"
ALFWORLD_EPISODES="${ALFWORLD_EPISODES:-134}"
ALFWORLD_BATCH_SIZE="${ALFWORLD_BATCH_SIZE:-$ALFWORLD_EPISODES}"
ALFWORLD_SEEDS="${ALFWORLD_SEEDS:-0 1 2 3 4}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"

WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$REPO_ROOT/agent_system/environments/env_package/webshop/webshop/data}"
WEBSHOP_EPISODES="${WEBSHOP_EPISODES:-500}"
WEBSHOP_BATCH_SIZE="${WEBSHOP_BATCH_SIZE:-$WEBSHOP_EPISODES}"
WEBSHOP_SEEDS="${WEBSHOP_SEEDS:-0 1 2}"
WEBSHOP_MAX_STEPS="${WEBSHOP_MAX_STEPS:-15}"
WEBSHOP_HISTORY_LENGTH="${WEBSHOP_HISTORY_LENGTH:-2}"

TASK_FILTER="${TASK_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-0}"

TEMP_MODEL_DIR=""
PREPARED_MODEL_PATH=""
PREPARED_SOURCE_IDENTITY=""

usage() {
    cat <<'EOF'
Usage:
  MODEL_MANIFEST=/path/to/models.tsv \
    bash examples/vpr_games/eval/eval_agentic_ood_all.sh

Manifest columns (tab-separated):
  model_id    source_type    source_path

source_type:
  hf          Hugging Face model directory.
  verl_fsdp   VERL global_step_* directory containing actor FSDP shards.

Defaults:
  ALFWorld: valid_unseen, all 134 tasks, 5 sampling seeds, max 50 steps.
  WebShop:  full 500-task test split, 3 sampling seeds, max 15 steps.
  Prompt:    ChatML with a final `Action: ACTION` line.
  Sampling:  16K response / 32K context, no format stop,
             temperature=0.6, top_p=0.95, top_k=20.

The script never stops existing training processes. By default it waits until
all selected GPUs use less than GPU_BUSY_MEMORY_MIB memory. Select idle devices
with CUDA_VISIBLE_DEVICES, or run on another machine sharing the repository.

Useful overrides:
  RUN_DIR, TASK_FILTER, MODEL_FILTER, CUDA_VISIBLE_DEVICES, N_GPUS, TP_SIZE,
  WAIT_FOR_FREE_GPUS, FORCE, DRY_RUN, ALFWORLD_DATA, WEBSHOP_DATA_DIR.
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
RUN_DIR="$(abspath "$RUN_DIR")"
MODEL_CACHE_ROOT="$(abspath "$MODEL_CACHE_ROOT")"
COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-$RUN_DIR/compile_cache/n${N_GPUS}_tp${TP_SIZE}}"
COMPILE_CACHE_ROOT="$(abspath "$COMPILE_CACHE_ROOT")"
ALFWORLD_DATA="$(abspath "$ALFWORLD_DATA")"
WEBSHOP_DATA_DIR="$(abspath "$WEBSHOP_DATA_DIR")"
JAVA_HOME="$(abspath "$JAVA_HOME")"
JVM_PATH="$(abspath "$JVM_PATH")"


cleanup() {
    rm -rf "$RAY_TEMP_ROOT"
    if [[ -n "$TEMP_MODEL_DIR" && -d "$TEMP_MODEL_DIR" ]]; then
        rm -rf "$TEMP_MODEL_DIR"
    fi
}
trap cleanup EXIT

[[ -n "$MODEL_MANIFEST" ]] || {
    usage >&2
    die "MODEL_MANIFEST is required"
}
MODEL_MANIFEST="$(abspath "$MODEL_MANIFEST")"
[[ -f "$MODEL_MANIFEST" ]] || die "model manifest not found: $MODEL_MANIFEST"
[[ -x "$PYTHON" ]] || die "Python not found: $PYTHON"
[[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]] || die "N_GPUS must be positive"
[[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be positive"
[[ "$MAX_PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]] || die "MAX_PROMPT_LENGTH must be positive"
[[ "$MAX_RESPONSE_LENGTH" =~ ^[1-9][0-9]*$ ]] || die "MAX_RESPONSE_LENGTH must be positive"
[[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || die "MAX_MODEL_LEN must be positive"
(( N_GPUS % TP_SIZE == 0 )) || die "N_GPUS must be divisible by TP_SIZE"
(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH <= MAX_MODEL_LEN )) ||
    die "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LEN"
(( ALFWORLD_EPISODES % ALFWORLD_BATCH_SIZE == 0 )) || die "ALFWORLD_EPISODES must be divisible by ALFWORLD_BATCH_SIZE"
(( WEBSHOP_EPISODES % WEBSHOP_BATCH_SIZE == 0 )) || die "WEBSHOP_EPISODES must be divisible by WEBSHOP_BATCH_SIZE"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_IDS[@]} == N_GPUS )) || die "CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} devices, expected N_GPUS=$N_GPUS"
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "GPU waiting supports numeric CUDA_VISIBLE_DEVICES entries only: $gpu_id"
done

declare -a MODEL_IDS=()
declare -a SOURCE_TYPES=()
declare -a SOURCE_PATHS=()
while IFS=$'\t' read -r model_id source_type source_path extra; do
    [[ -n "$model_id" && "${model_id:0:1}" != "#" ]] || continue
    [[ -z "${extra:-}" ]] || die "manifest row has more than 3 columns: $model_id"
    [[ "$model_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid model_id: $model_id"
    [[ "$source_type" == hf || "$source_type" == verl_fsdp ]] || die "unsupported source_type '$source_type' for $model_id"
    [[ -n "$source_path" ]] || die "missing source_path for $model_id"
    if contains_word "$MODEL_FILTER" "$model_id"; then
        MODEL_IDS+=("$model_id")
        SOURCE_TYPES+=("$source_type")
        SOURCE_PATHS+=("$(abspath "$source_path")")
    fi
done < "$MODEL_MANIFEST"
(( ${#MODEL_IDS[@]} > 0 )) || die "no models selected from $MODEL_MANIFEST"

validate_model_source() {
    local model_id="$1"
    local source_type="$2"
    local source_path="$3"
    [[ -d "$source_path" ]] || die "model '$model_id' source not found: $source_path"
    if [[ "$source_type" == hf ]]; then
        [[ -f "$source_path/config.json" ]] || die "model '$model_id' has no config.json: $source_path"
        compgen -G "$source_path/*.safetensors" >/dev/null || compgen -G "$source_path/*.bin" >/dev/null || die "model '$model_id' has no HF weights: $source_path"
        return
    fi

    local actor_dir="$source_path/actor"
    [[ -f "$actor_dir/config.json" ]] || die "model '$model_id' actor config missing: $actor_dir"
    local rank_zero world_size shard_count
    rank_zero="$(find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' -print -quit)"
    [[ -n "$rank_zero" ]] || die "model '$model_id' has no FSDP rank-zero shard"
    world_size="$(basename "$rank_zero" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
    [[ "$world_size" =~ ^[0-9]+$ ]] || die "cannot parse FSDP world size for '$model_id'"
    shard_count="$(find "$actor_dir" -maxdepth 1 -type f -name "model_world_size_${world_size}_rank_*.pt" | wc -l)"
    [[ "$shard_count" -eq "$world_size" ]] || die "model '$model_id' has $shard_count/$world_size FSDP shards"
}

source_identity() {
    local source_type="$1"
    local source_path="$2"
    local scan_path="$source_path"
    {
        printf 'source_type=%s\n' "$source_type"
        printf 'source_path=%s\n' "$(realpath "$source_path")"
        if [[ "$source_type" == verl_fsdp ]]; then
            printf 'model_merger_sha256=%s\n' "$(sha256sum "$REPO_ROOT/scripts/model_merger.py" | awk '{print $1}')"
            scan_path="$source_path/actor"
            find "$scan_path" -maxdepth 1 -type f \
                \( -name 'model_world_size_*_rank_*.pt' -o -name 'config.json' -o -name 'generation_config.json' \) \
                -printf '%f|%s|%T@\n' | sort
        else
            find "$scan_path" -maxdepth 1 -type f \
                \( -name 'model*.safetensors*' -o -name '*.bin' -o -name 'config.json' -o -name 'generation_config.json' \) \
                -printf '%f|%s|%T@\n' | sort
        fi
    } | sha256sum | awk '{print $1}'
}

prepare_model() {
    local model_id="$1"
    local source_type="$2"
    local source_path="$3"
    PREPARED_SOURCE_IDENTITY="$(source_identity "$source_type" "$source_path")"
    if [[ "$source_type" == hf ]]; then
        PREPARED_MODEL_PATH="$source_path"
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
    "$PYTHON" "$REPO_ROOT/scripts/model_merger.py" merge --backend fsdp --local_dir "$source_path/actor" --target_dir "$TEMP_MODEL_DIR"
    [[ -f "$TEMP_MODEL_DIR/config.json" ]] || die "merge produced no config.json for '$model_id'"
    compgen -G "$TEMP_MODEL_DIR/*.safetensors" >/dev/null || die "merge produced no safetensors for '$model_id'"
    {
        printf 'model_id=%s\n' "$model_id"
        printf 'source=%s\n' "$(realpath "$source_path")"
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
        if [[ "$pids" =~ [0-9] ]]; then
            return 1
        fi
        used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$used" =~ ^[0-9]+$ ]] || die "cannot query memory for GPU $gpu_id"
        if (( used >= GPU_BUSY_MEMORY_MIB )); then
            return 1
        fi
    done
}

wait_for_selected_gpus() {
    [[ "$DRY_RUN" == 1 || "$WAIT_FOR_FREE_GPUS" == 0 ]] && return
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required for GPU waiting"
    while ! selected_gpus_are_free; do
        log "Selected GPUs are busy; waiting $GPU_POLL_SECONDS seconds (no process will be stopped)"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
        sleep "$GPU_POLL_SECONDS"
    done
    log "Selected GPUs are free: $CUDA_VISIBLE_DEVICES"
}

preflight_data() {
    [[ "$SKIP_DATA_CHECK" == 1 ]] && return
    if contains_word "$TASK_FILTER" alfworld; then
        [[ -n "$ALFWORLD_DATA" ]] || die "ALFWORLD_DATA is unset; point it at the extracted ALFWorld data root"
        [[ -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ]] || die "ALFWorld valid_unseen not found under $ALFWORLD_DATA"
        [[ -f "$ALFWORLD_DATA/logic/alfred.pddl" ]] || die "ALFWorld logic files not found under $ALFWORLD_DATA"
    fi
    if contains_word "$TASK_FILTER" webshop; then
        [[ -f "$WEBSHOP_DATA_DIR/items_shuffle.json" ]] || die "full WebShop products missing: $WEBSHOP_DATA_DIR/items_shuffle.json"
        [[ -f "$WEBSHOP_DATA_DIR/items_ins_v2.json" ]] || die "full WebShop attributes missing: $WEBSHOP_DATA_DIR/items_ins_v2.json"
        [[ -x "$JAVA_HOME/bin/java" ]] || die "Java runtime missing: $JAVA_HOME/bin/java"
        [[ -f "$JVM_PATH" ]] || die "JVM library missing: $JVM_PATH"
        local index_dir="$REPO_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine/indexes"
        [[ -d "$index_dir" ]] || die "full WebShop Lucene index missing: $index_dir (run webshop/setup.sh -d all)"
    fi
}

write_protocol() {
    mkdir -p "$RUN_DIR"
    local candidate="$RUN_DIR/protocol.env.new"
    {
        printf 'PROTOCOL_VERSION=2\n'
        printf 'MODEL_MANIFEST=%s\n' "$MODEL_MANIFEST"
        printf 'TASK_FILTER=%s\n' "${TASK_FILTER:-all}"
        printf 'ENV_SEED=%s\n' "$ENV_SEED"
        printf 'GPU=cuda:%s,n_gpus:%s,tp:%s,memory_util:%s\n' "$CUDA_VISIBLE_DEVICES" "$N_GPUS" "$TP_SIZE" "$GPU_MEM_UTIL"
        printf 'COMPILE_CACHE_SCOPE=n%s_tp%s\n' "$N_GPUS" "$TP_SIZE"
        printf 'BATCHING=max_batched_tokens:%s,max_num_seqs:%s\n' "$MAX_NUM_BATCHED_TOKENS" "$MAX_NUM_SEQS"
        printf 'ALFWORLD_DATA=%s\n' "${ALFWORLD_DATA:-unset}"
        printf 'WEBSHOP_DATA_DIR=%s\n' "$WEBSHOP_DATA_DIR"
        printf 'JAVA_HOME=%s\n' "$JAVA_HOME"
        printf 'JVM_PATH=%s\n' "$JVM_PATH"
        printf 'GIT_REVISION=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'EVALUATOR_SHA256=%s\n' "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
        printf 'SAMPLING=temperature:%s,top_p:%s,top_k:%s,min_p:%s\n' "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MIN_P"
        printf 'PROMPT_RENDERING=chatml\n'
        printf 'ACTION_PROTOCOL=native_action_v2,prompt:Action,strict_admissible:true,format_stop:none\n'
        printf 'MODEL_LIMITS=prompt:%s,response_per_turn:%s,model:%s\n' "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_MODEL_LEN"
        printf 'ALFWORLD=episodes:%s,batch:%s,seeds:%s,max_steps:%s,split:valid_unseen\n' "$ALFWORLD_EPISODES" "$ALFWORLD_BATCH_SIZE" "$ALFWORLD_SEEDS" "$ALFWORLD_MAX_STEPS"
        printf 'WEBSHOP=episodes:%s,batch:%s,seeds:%s,max_steps:%s,split:test\n' "$WEBSHOP_EPISODES" "$WEBSHOP_BATCH_SIZE" "$WEBSHOP_SEEDS" "$WEBSHOP_MAX_STEPS"
        for i in "${!MODEL_IDS[@]}"; do
            printf 'MODEL=%s|%s|%s\n' "${MODEL_IDS[$i]}" "${SOURCE_TYPES[$i]}" "${SOURCE_PATHS[$i]}"
        done
    } > "$candidate"

    if [[ -f "$RUN_DIR/protocol.env" ]]; then
        if ! cmp -s "$RUN_DIR/protocol.env" "$candidate"; then
            diff -u "$RUN_DIR/protocol.env" "$candidate" >&2 || true
            rm -f "$candidate"
            die "protocol differs from existing RUN_DIR; use a new directory"
        fi
        rm -f "$candidate"
    else
        mv "$candidate" "$RUN_DIR/protocol.env"
    fi
    PROTOCOL_SHA256="$(sha256sum "$RUN_DIR/protocol.env" | awk '{print $1}')"
    printf '%s  protocol.env\n' "$PROTOCOL_SHA256" > "$RUN_DIR/protocol.sha256"
    export PROTOCOL_SHA256
}

write_source_metadata() {
    local metadata="$RUN_DIR/source_$RUN_ID.env"
    local dirty=false
    if ! git -C "$REPO_ROOT" diff --quiet --ignore-submodules HEAD -- 2>/dev/null ||
        [[ -n "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null)" ]]; then
        dirty=true
    fi
    {
        printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_revision=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'git_dirty=%s\n' "$dirty"
        printf 'command=%q ' "$0"
        printf '%q ' "$@"
        printf '\n'
    } > "$metadata"
}

prepare_eval_data() {
    local task="$1"
    local count="$2"
    local data_dir="$RUN_DIR/data/$task"
    mkdir -p "$data_dir"
    "$PYTHON" "$REPO_ROOT/examples/vpr_games/prepare_data.py" --env-name "$task" --train-size "$N_GPUS" --val-size "$count" --output-dir "$data_dir"
}

append_task_overrides() {
    local task="$1"
    local -n args_ref="$2"
    if [[ "$task" == alfworld ]]; then
        args_ref+=(
            "env.env_name=alfworld/AlfredTWEnv"
            "env.max_steps=$ALFWORLD_MAX_STEPS"
            "env.history_length=2"
            "env.alfworld.eval_dataset=eval_out_of_distribution"
            "env.alfworld.deterministic_eval=true"
        )
    else
        args_ref+=(
            "env.env_name=Webshop"
            "env.max_steps=$WEBSHOP_MAX_STEPS"
            "env.history_length=$WEBSHOP_HISTORY_LENGTH"
            "env.webshop.use_small=false"
            "env.webshop.human_goals=false"
            "env.webshop.data_dir=$WEBSHOP_DATA_DIR"
            "env.webshop.deterministic_eval=true"
            "env.webshop.shared_server=true"
        )
    fi
}

summarize_results() {
    "$PYTHON" "$SCRIPT_DIR/summarize_agentic_ood.py" --run-dir "$RUN_DIR"
}

run_one() {
    local model_id="$1"
    local task="$2"
    local sample_seed="$3"
    local val_count val_batch
    if [[ "$task" == alfworld ]]; then
        val_count="$ALFWORLD_EPISODES"
        val_batch="$ALFWORLD_BATCH_SIZE"
    else
        val_count="$WEBSHOP_EPISODES"
        val_batch="$WEBSHOP_BATCH_SIZE"
    fi

    local output_dir="$RUN_DIR/results/$model_id/$task/seed_$sample_seed"
    local raw_dir="$output_dir/raw"
    local done_file="$output_dir/.done"
    local log_file="$output_dir/eval.log"
    if [[ -f "$done_file" && "$FORCE" != 1 ]]; then
        if grep -Fxq "protocol_sha256=$PROTOCOL_SHA256" "$done_file" &&
            grep -Fxq "source_identity=$PREPARED_SOURCE_IDENTITY" "$done_file" &&
            compgen -G "$raw_dir/*.metrics.json" >/dev/null; then
            log "Skipping completed $model_id/$task/seed_$sample_seed"
            return
        fi
        die "stale completion marker: $done_file (set FORCE=1 to rerun)"
    fi
    if [[ "$FORCE" == 1 && -d "$output_dir" ]]; then
        rm -rf "$output_dir"
    fi
    mkdir -p \
        "$raw_dir" \
        "$output_dir/hydra" \
        "$output_dir/ray" \
        "$COMPILE_CACHE_ROOT/vllm" \
        "$COMPILE_CACHE_ROOT/torchinductor" \
        "$COMPILE_CACHE_ROOT/triton"

    local -a cmd=(
        "$PYTHON" -m verl.trainer.main_ppo
        "data.train_files=$RUN_DIR/data/$task/train.parquet"
        "data.val_files=$RUN_DIR/data/$task/test.parquet"
        "data.train_batch_size=$N_GPUS"
        "data.val_batch_size=$val_batch"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH"
        "data.max_response_length=$MAX_RESPONSE_LENGTH"
        "data.filter_overlong_prompts=False"
        "data.return_raw_chat=True"
        "+data.dataloader_num_workers=0"
        "actor_rollout_ref.model.path=$PREPARED_MODEL_PATH"
        "actor_rollout_ref.model.use_remove_padding=False"
        "actor_rollout_ref.model.enable_gradient_checkpointing=False"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$N_GPUS"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.use_torch_compile=False"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE"
        "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL"
        "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
        "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        "actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS"
        "actor_rollout_ref.rollout.enable_chunked_prefill=True"
        "actor_rollout_ref.rollout.enforce_eager=False"
        "actor_rollout_ref.rollout.free_cache_engine=False"
        "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
        "actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE"
        "actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P"
        "actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K"
        "actor_rollout_ref.rollout.val_kwargs.min_p=$MIN_P"
        "actor_rollout_ref.rollout.val_kwargs.seed=$sample_seed"
        "actor_rollout_ref.rollout.val_kwargs.n=1"
        "algorithm.adv_estimator=grpo"
        "algorithm.use_kl_in_reward=False"
        "algorithm.filter_groups.enable=False"
        "env.seed=$ENV_SEED"
        "env.agentic_eval.native_action_protocol=true"
        "env.rollout.n=1"
        "env.resources_per_worker.num_cpus=0.1"
        "trainer.total_training_steps=1"
        "trainer.total_epochs=1"
        "trainer.val_before_train=True"
        "trainer.val_only=True"
        "trainer.test_freq=-1"
        "trainer.save_freq=-1"
        "trainer.balance_batch=False"
        "trainer.n_gpus_per_node=$N_GPUS"
        "trainer.nnodes=1"
        "trainer.logger=[console]"
        "trainer.project_name=vpr_agentic_ood_eval"
        "trainer.experiment_name=${model_id}_${task}_seed_${sample_seed}"
        "trainer.default_local_dir=$output_dir/checkpoints"
        "trainer.validation_data_dir=$raw_dir"
        "trainer.resume_mode=disable"
        "hydra.run.dir=$output_dir/hydra"
        "ray_init.num_cpus=$RAY_CPUS"
        "+ray_init._temp_dir=$RAY_TEMP_ROOT"
    )
    append_task_overrides "$task" cmd

    log "Running $model_id/$task/seed_$sample_seed ($val_count episodes)"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ALFWORLD_DATA=%q ' "$CUDA_VISIBLE_DEVICES" "$ALFWORLD_DATA"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi

    env -u RAY_ADDRESS \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        ALFWORLD_DATA="$ALFWORLD_DATA" \
        JAVA_HOME="$JAVA_HOME" \
        JVM_PATH="$JVM_PATH" \
        PATH="$JAVA_HOME/bin:$PATH" \
        VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        VLLM_CACHE_ROOT="$COMPILE_CACHE_ROOT/vllm" \
        TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_ROOT/torchinductor" \
        TRITON_CACHE_DIR="$COMPILE_CACHE_ROOT/triton" \
        TOKENIZERS_PARALLELISM=false \
        HYDRA_FULL_ERROR=1 \
        "${cmd[@]}" 2>&1 | tee "$log_file"

    compgen -G "$raw_dir/*.metrics.json" >/dev/null || die "evaluation produced no metrics JSON: $raw_dir"
    compgen -G "$raw_dir/*.jsonl" >/dev/null || die "evaluation produced no raw generations: $raw_dir"
    {
        printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'protocol_sha256=%s\n' "$PROTOCOL_SHA256"
        printf 'source_identity=%s\n' "$PREPARED_SOURCE_IDENTITY"
        printf 'prepared_model=%s\n' "$PREPARED_MODEL_PATH"
    } > "$done_file"
    summarize_results
}

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.eval.lock"
flock -n 9 || die "another evaluator is using RUN_DIR=$RUN_DIR"

write_protocol
write_source_metadata "$@"
preflight_data
wait_for_selected_gpus

declare -a TASKS=(alfworld webshop)
for task in "${TASKS[@]}"; do
    contains_word "$TASK_FILTER" "$task" || continue
    if [[ "$task" == alfworld ]]; then
        prepare_eval_data "$task" "$ALFWORLD_EPISODES"
    else
        prepare_eval_data "$task" "$WEBSHOP_EPISODES"
    fi
done

log "Run directory: $RUN_DIR"
log "Selected models: ${#MODEL_IDS[@]}"
log "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS tp=$TP_SIZE"
log "The evaluator will not stop existing training processes"

for i in "${!MODEL_IDS[@]}"; do
    model_id="${MODEL_IDS[$i]}"
    source_type="${SOURCE_TYPES[$i]}"
    source_path="${SOURCE_PATHS[$i]}"
    validate_model_source "$model_id" "$source_type" "$source_path"
    if [[ "$DRY_RUN" == 1 ]]; then
        PREPARED_MODEL_PATH="$source_path"
        PREPARED_SOURCE_IDENTITY="dry_run"
    else
        prepare_model "$model_id" "$source_type" "$source_path"
    fi

    for task in "${TASKS[@]}"; do
        contains_word "$TASK_FILTER" "$task" || continue
        if [[ "$task" == alfworld ]]; then
            for sample_seed in $ALFWORLD_SEEDS; do
                run_one "$model_id" "$task" "$sample_seed"
            done
        else
            for sample_seed in $WEBSHOP_SEEDS; do
                run_one "$model_id" "$task" "$sample_seed"
            done
        fi
    done
done

if [[ "$DRY_RUN" != 1 ]]; then
    summarize_results
fi
log "All selected agentic evaluations completed: $RUN_DIR"
