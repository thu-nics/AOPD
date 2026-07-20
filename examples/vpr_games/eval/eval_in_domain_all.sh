#!/usr/bin/env bash
# Reproducible sequential in-domain evaluation for VPR game checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH:-}}"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)"

if [[ -n "${RUN_DIR:-}" && -n "${EVAL_ROOT:-}" && "$RUN_DIR" != "$EVAL_ROOT" ]]; then
    echo "ERROR: RUN_DIR and deprecated EVAL_ROOT point to different directories" >&2
    exit 1
fi
RUN_DIR="${RUN_DIR:-${EVAL_ROOT:-$REPO_ROOT/runs/eval_$RUN_ID}}"

VAL_GAMES="${VAL_GAMES:-100}"
ENV_SEEDS="${ENV_SEEDS:-${EVAL_SEEDS:-0 100 200 300 400}}"
TRAIN_STUB_SIZE="${TRAIN_STUB_SIZE:-32}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-}"
# Empty: engine RNG; "env": each environment seed; integer: one fixed seed.
GENERATION_SEED="${GENERATION_SEED:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
RAY_CPUS="${RAY_CPUS:-64}"

TASK_FILTER="${TASK_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

VPR_SOKOBAN_CKPT="${VPR_SOKOBAN_CKPT:-runs/20260703T043131/ckpt/global_step_100}"
GRPO_SOKOBAN_CKPT="${GRPO_SOKOBAN_CKPT:-runs/20260703T170651/ckpt/global_step_100}"
VINEPPO_SOKOBAN_CKPT="${VINEPPO_SOKOBAN_CKPT:-runs/20260713T182939/ckpt/global_step_50}"
VANILLA_SOKOBAN_CKPT="${VANILLA_SOKOBAN_CKPT:-runs/20260708T031237/ckpt/global_step_100}"
VPR_SUDOKU_CKPT="${VPR_SUDOKU_CKPT:-runs/20260706T170113/ckpt/global_step_60}"
GRPO_SUDOKU_CKPT="${GRPO_SUDOKU_CKPT:-runs/20260702T174140/ckpt/global_step_100}"
VINEPPO_SUDOKU_CKPT="${VINEPPO_SUDOKU_CKPT:-runs/20260714T061405/ckpt/global_step_50}"
VPR_MINESWEEPER_CKPT="${VPR_MINESWEEPER_CKPT:-runs/20260627T093616/ckpt/global_step_200}"
GRPO_MINESWEEPER_CKPT="${GRPO_MINESWEEPER_CKPT:-runs/20260630T171610/ckpt/global_step_200}"
VINEPPO_MINESWEEPER_CKPT="${VINEPPO_MINESWEEPER_CKPT:-runs/20260713T183016/ckpt/global_step_100}"
NOISE20_CKPT="${NOISE20_CKPT:-runs/20260709T032104/ckpt/global_step_100}"
NOISE40_CKPT="${NOISE40_CKPT:-runs/20260711T161325/ckpt/global_step_100}"

SOKOBAN_DIM_ROOM="${SOKOBAN_DIM_ROOM:-7,7}"
SOKOBAN_NUM_BOXES="${SOKOBAN_NUM_BOXES:-3}"
SOKOBAN_SEARCH_DEPTH="${SOKOBAN_SEARCH_DEPTH:-25}"
SOKOBAN_MAX_STEPS="${SOKOBAN_MAX_STEPS:-36}"
SOKOBAN_INVALID_PENALTY="${SOKOBAN_INVALID_PENALTY:--2}"
SUDOKU_N="${SUDOKU_N:-3}"
SUDOKU_CLUES="${SUDOKU_CLUES:-40}"
SUDOKU_MAX_STEPS="${SUDOKU_MAX_STEPS:-100}"
SUDOKU_INVALID_PENALTY="${SUDOKU_INVALID_PENALTY:--2}"
MINESWEEPER_ROWS="${MINESWEEPER_ROWS:-5}"
MINESWEEPER_COLS="${MINESWEEPER_COLS:-5}"
MINESWEEPER_MINES="${MINESWEEPER_MINES:-5}"
MINESWEEPER_MAX_STEPS="${MINESWEEPER_MAX_STEPS:-25}"
MINESWEEPER_INVALID_PENALTY="${MINESWEEPER_INVALID_PENALTY:--2}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found: $PYTHON" >&2
    exit 1
fi
if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: set MODEL_PATH to the base Hugging Face model directory" >&2
    exit 1
fi
if (( VAL_GAMES <= 0 )); then
    echo "ERROR: VAL_GAMES must be positive" >&2
    exit 1
fi
if (( TRAIN_STUB_SIZE <= 0 || TRAIN_STUB_SIZE % N_GPUS != 0 )); then
    echo "ERROR: TRAIN_STUB_SIZE must be positive and divisible by N_GPUS" >&2
    exit 1
fi
if [[ -n "$GENERATION_SEED" && "$GENERATION_SEED" != "env" && ! "$GENERATION_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GENERATION_SEED must be empty, 'env', or a non-negative integer" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.eval.lock"
if ! flock -n 9; then
    echo "ERROR: another evaluator is using $RUN_DIR" >&2
    exit 1
fi

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

declare -a JOBS=(
    "sokoban|qwen3_4b_base|BASE"
    "sokoban|vpr_sokoban|$VPR_SOKOBAN_CKPT"
    "sokoban|grpo_sokoban|$GRPO_SOKOBAN_CKPT"
    "sokoban|vineppo_sokoban|$VINEPPO_SOKOBAN_CKPT"
    "sokoban|vpr_sokoban_vanilla|$VANILLA_SOKOBAN_CKPT"
    "sudoku|qwen3_4b_base|BASE"
    "sudoku|vpr_sudoku|$VPR_SUDOKU_CKPT"
    "sudoku|grpo_sudoku|$GRPO_SUDOKU_CKPT"
    "sudoku|vineppo_sudoku|$VINEPPO_SUDOKU_CKPT"
    "minesweeper|qwen3_4b_base|BASE"
    "minesweeper|vpr_minesweeper|$VPR_MINESWEEPER_CKPT"
    "minesweeper|grpo_minesweeper|$GRPO_MINESWEEPER_CKPT"
    "minesweeper|vineppo_minesweeper|$VINEPPO_MINESWEEPER_CKPT"
)
if [[ -n "$NOISE20_CKPT" ]]; then
    JOBS+=("sokoban|vpr_sokoban_noise20|$NOISE20_CKPT")
else
    echo "INFO: set NOISE20_CKPT to include the 20% noise checkpoint"
fi
if [[ -n "$NOISE40_CKPT" ]]; then
    JOBS+=("sokoban|vpr_sokoban_noise40|$NOISE40_CKPT")
else
    echo "INFO: set NOISE40_CKPT to include the 40% noise checkpoint"
fi

write_protocol() {
    local candidate="$RUN_DIR/protocol.env.new"
    {
        printf 'PROTOCOL_VERSION=1\n'
        printf 'MODEL_PATH=%s\n' "$(abspath "$MODEL_PATH")"
        printf 'VAL_GAMES=%s\n' "$VAL_GAMES"
        printf 'ENV_SEEDS=%s\n' "$ENV_SEEDS"
        printf 'ENABLE_THINKING=%s\n' "$ENABLE_THINKING"
        printf 'TEMPERATURE=%s\n' "$TEMPERATURE"
        printf 'TOP_P=%s\n' "$TOP_P"
        printf 'TOP_K=%s\n' "$TOP_K"
        printf 'MIN_P=%s\n' "${MIN_P:-unset}"
        printf 'GENERATION_SEED=%s\n' "${GENERATION_SEED:-unset}"
        printf 'DO_SAMPLE=true\n'
        printf 'ROLLOUT_MODE=vanilla\n'
        printf 'ROLLOUT_N=1\n'
        printf 'REWARD_MODE=outcome\n'
        printf 'MAX_PROMPT_LENGTH=%s\n' "$MAX_PROMPT_LENGTH"
        printf 'MAX_RESPONSE_LENGTH=%s\n' "$MAX_RESPONSE_LENGTH"
        printf 'N_GPUS=%s\n' "$N_GPUS"
        printf 'TP_SIZE=%s\n' "$TP_SIZE"
        printf 'GPU_MEM_UTIL=%s\n' "$GPU_MEM_UTIL"
        printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "$MAX_NUM_BATCHED_TOKENS"
        printf 'SOKOBAN=%s,%s,%s,%s,%s\n' "$SOKOBAN_DIM_ROOM" "$SOKOBAN_NUM_BOXES" "$SOKOBAN_SEARCH_DEPTH" "$SOKOBAN_MAX_STEPS" "$SOKOBAN_INVALID_PENALTY"
        printf 'SUDOKU=%s,%s,%s,%s\n' "$SUDOKU_N" "$SUDOKU_CLUES" "$SUDOKU_MAX_STEPS" "$SUDOKU_INVALID_PENALTY"
        printf 'MINESWEEPER=%s,%s,%s,%s,%s\n' "$MINESWEEPER_ROWS" "$MINESWEEPER_COLS" "$MINESWEEPER_MINES" "$MINESWEEPER_MAX_STEPS" "$MINESWEEPER_INVALID_PENALTY"
        for job in "${JOBS[@]}"; do
            IFS='|' read -r task model_id checkpoint_spec <<< "$job"
            if [[ "$checkpoint_spec" != BASE ]]; then
                checkpoint_spec="$(abspath "$checkpoint_spec")"
            fi
            printf 'JOB=%s|%s|%s\n' "$task" "$model_id" "$checkpoint_spec"
        done
    } > "$candidate"

    if [[ -f "$RUN_DIR/protocol.env" ]]; then
        if ! cmp -s "$RUN_DIR/protocol.env" "$candidate"; then
            echo "ERROR: protocol differs from existing run: $RUN_DIR" >&2
            diff -u "$RUN_DIR/protocol.env" "$candidate" >&2 || true
            rm -f "$candidate"
            echo "Use a new RUN_DIR or restore the original protocol." >&2
            exit 1
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
    local revision dirty metadata_path="$RUN_DIR/source.env"
    revision="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
    if git -C "$REPO_ROOT" diff --quiet --ignore-submodules HEAD -- 2>/dev/null &&
        [[ -z "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null)" ]]; then
        dirty=false
    else
        dirty=true
    fi
    if [[ -e "$metadata_path" ]]; then
        metadata_path="$RUN_DIR/source_resume_${RUN_ID}_$$.env"
    fi
    {
        printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_revision=%s\n' "$revision"
        printf 'git_dirty=%s\n' "$dirty"
        printf 'command=%q ' "$0"
        printf '%q ' "$@"
        printf '\n'
    } > "$metadata_path"
}

prepare_eval_data() {
    local task="$1"
    local data_dir="$RUN_DIR/data/$task"
    mkdir -p "$data_dir"
    "$PYTHON" "$REPO_ROOT/examples/vpr_games/prepare_data.py" \
        --env-name "vpr_$task" \
        --train-size "$TRAIN_STUB_SIZE" \
        --val-size "$VAL_GAMES" \
        --output-dir "$data_dir"
}

append_task_overrides() {
    local task="$1"
    local -n args_ref="$2"
    case "$task" in
        sokoban)
            args_ref+=(
                "env.max_steps=$SOKOBAN_MAX_STEPS"
                "env.invalid_penalty=$SOKOBAN_INVALID_PENALTY"
                "env.sokoban.dim_room=[$SOKOBAN_DIM_ROOM]"
                "env.sokoban.num_boxes=$SOKOBAN_NUM_BOXES"
                "env.sokoban.search_depth=$SOKOBAN_SEARCH_DEPTH"
                "env.sokoban.reward_mode=outcome"
            )
            ;;
        sudoku)
            args_ref+=(
                "env.max_steps=$SUDOKU_MAX_STEPS"
                "env.invalid_penalty=$SUDOKU_INVALID_PENALTY"
                "env.sudoku.n=$SUDOKU_N"
                "env.sudoku.clues=$SUDOKU_CLUES"
                "env.sudoku.terminate_on_wrong_digit=false"
                "env.sudoku.reward_mode=outcome"
            )
            ;;
        minesweeper)
            args_ref+=(
                "env.max_steps=$MINESWEEPER_MAX_STEPS"
                "env.invalid_penalty=$MINESWEEPER_INVALID_PENALTY"
                "env.minesweeper.rows=$MINESWEEPER_ROWS"
                "env.minesweeper.cols=$MINESWEEPER_COLS"
                "env.minesweeper.mines=$MINESWEEPER_MINES"
                "env.minesweeper.auto_reveal_center=true"
                "env.minesweeper.reward_mode=outcome"
            )
            ;;
        *)
            echo "ERROR: unsupported task: $task" >&2
            exit 1
            ;;
    esac
}

summarize_results() {
    "$PYTHON" "$SCRIPT_DIR/summarize_in_domain_eval.py" --run-dir "$RUN_DIR"
}

run_one() {
    local task="$1"
    local model_id="$2"
    local checkpoint_spec="$3"
    local env_seed="$4"
    local data_dir="$RUN_DIR/data/$task"
    local eval_id="seed_$env_seed"
    local output_dir="$RUN_DIR/$task/$model_id/$eval_id"
    local raw_dir="$output_dir/raw"
    local log_file="$output_dir/eval.log"
    local done_file="$output_dir/.done"
    local resume_mode resume_path checkpoint_identity

    if [[ "$checkpoint_spec" == BASE ]]; then
        resume_mode=disable
        resume_path=null
        checkpoint_identity=BASE
    else
        resume_mode=resume_path
        resume_path="$(abspath "$checkpoint_spec")"
        checkpoint_identity="$resume_path"
        if [[ ! -d "$resume_path" ]]; then
            echo "ERROR: checkpoint not found: $resume_path" >&2
            exit 1
        fi
    fi

    if [[ -f "$done_file" && "$FORCE" != 1 ]]; then
        if grep -Fxq "protocol_sha256=$PROTOCOL_SHA256" "$done_file" \
            && grep -Fxq "checkpoint=$checkpoint_identity" "$done_file" \
            && compgen -G "$raw_dir/*.metrics.json" >/dev/null \
            && compgen -G "$raw_dir/*.jsonl" >/dev/null; then
            echo "SKIP: $task / $model_id / $eval_id already completed"
            return
        fi
        echo "ERROR: stale completion marker: $done_file (use FORCE=1 to rerun)" >&2
        exit 1
    fi

    mkdir -p "$raw_dir" "$output_dir/hydra"
    rm -f "$done_file" "$raw_dir"/*.jsonl "$raw_dir"/*.metrics.json

    local -a cmd=(
        "$PYTHON" -m verl.trainer.main_ppo
        --config-name "vpr_$task"
        "data.train_files=$data_dir/train.parquet"
        "data.val_files=$data_dir/test.parquet"
        "data.train_batch_size=$TRAIN_STUB_SIZE"
        "data.val_batch_size=$VAL_GAMES"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH"
        "data.max_response_length=$MAX_RESPONSE_LENGTH"
        "data.filter_overlong_prompts=False"
        "data.return_raw_chat=True"
        "+data.dataloader_num_workers=0"
        "+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING"
        "actor_rollout_ref.model.path=$MODEL_PATH"
        "actor_rollout_ref.model.use_remove_padding=False"
        "actor_rollout_ref.model.enable_gradient_checkpointing=False"
        "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_STUB_SIZE"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"
        "actor_rollout_ref.actor.use_kl_loss=False"
        "actor_rollout_ref.actor.use_torch_compile=False"
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE"
        "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL"
        "actor_rollout_ref.rollout.max_model_len=8192"
        "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        "actor_rollout_ref.rollout.enable_chunked_prefill=False"
        "actor_rollout_ref.rollout.enforce_eager=True"
        "actor_rollout_ref.rollout.free_cache_engine=True"
        "actor_rollout_ref.rollout.multi_turn.enable=true"
        "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
        "actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE"
        "actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P"
        "actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K"
        "actor_rollout_ref.rollout.val_kwargs.n=1"
        "algorithm.use_kl_in_reward=False"
        "algorithm.filter_groups.enable=False"
        "env.seed=$env_seed"
        "env.rollout.mode=vanilla"
        "env.rollout.n=1"
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
        "trainer.project_name=vpr_in_domain_eval"
        "trainer.experiment_name=${model_id}_${task}_${eval_id}"
        "trainer.default_local_dir=$output_dir/checkpoints"
        "trainer.validation_data_dir=$raw_dir"
        "trainer.resume_mode=$resume_mode"
        "trainer.resume_from_path=$resume_path"
        "hydra.run.dir=$output_dir/hydra"
        "ray_init.num_cpus=$RAY_CPUS"
    )
    if [[ -n "$MIN_P" ]]; then
        cmd+=("actor_rollout_ref.rollout.val_kwargs.min_p=$MIN_P")
    fi
    if [[ "$GENERATION_SEED" == env ]]; then
        cmd+=("actor_rollout_ref.rollout.val_kwargs.seed=$env_seed")
    elif [[ -n "$GENERATION_SEED" ]]; then
        cmd+=("actor_rollout_ref.rollout.val_kwargs.seed=$GENERATION_SEED")
    fi
    append_task_overrides "$task" cmd

    echo "RUN: $task / $model_id / $eval_id"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '  CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    TOKENIZERS_PARALLELISM=false \
    HYDRA_FULL_ERROR=1 \
        "${cmd[@]}" 2>&1 | tee "$log_file"

    if ! compgen -G "$raw_dir/*.metrics.json" >/dev/null; then
        echo "ERROR: evaluation finished without metrics JSON: $raw_dir" >&2
        exit 1
    fi
    if ! compgen -G "$raw_dir/*.jsonl" >/dev/null; then
        echo "ERROR: evaluation finished without raw JSONL: $raw_dir" >&2
        exit 1
    fi
    {
        printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'checkpoint=%s\n' "$checkpoint_identity"
        printf 'protocol_sha256=%s\n' "$PROTOCOL_SHA256"
    } > "$done_file"
    summarize_results
}

write_protocol
write_source_metadata "$@"

echo "=== VPR games sequential in-domain evaluation ==="
echo "Run directory: $RUN_DIR"
echo "Games/run:     $VAL_GAMES"
echo "Environment:   $ENV_SEEDS"
echo "Sampling:      thinking=$ENABLE_THINKING temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K min_p=${MIN_P:-unset} generation_seed=${GENERATION_SEED:-unset}"
echo "GPU:           CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS tp=$TP_SIZE"
echo "Protocol:      $PROTOCOL_SHA256"

declare -A prepared_tasks=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r task model_id checkpoint_spec <<< "$job"
    contains_word "$TASK_FILTER" "$task" || continue
    contains_word "$MODEL_FILTER" "$model_id" || continue
    if [[ -z "${prepared_tasks[$task]:-}" ]]; then
        prepare_eval_data "$task"
        prepared_tasks[$task]=1
    fi
    for env_seed in $ENV_SEEDS; do
        run_one "$task" "$model_id" "$checkpoint_spec" "$env_seed"
    done
done

if [[ "$DRY_RUN" != 1 ]]; then
    summarize_results
fi
echo "=== all selected evaluations completed: $RUN_DIR ==="
