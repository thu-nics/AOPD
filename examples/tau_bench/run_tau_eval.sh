#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:?Set MODEL_SPECS_FILE to a tab-separated model registry}"
QUALIFICATION_MANIFEST="${QUALIFICATION_MANIFEST:?Set QUALIFICATION_MANIFEST to qualification_manifest.json}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/.cache/tau2-bench-17e07b1}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/tau_eval_$(date -u +%Y%m%dT%H%M%S)}"
SEEDS="${SEEDS:-300 301 302 303}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_PROMPT="${MAX_PROMPT:-24576}"
MAX_RESPONSE="${MAX_RESPONSE:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
RAY_CPUS="${RAY_CPUS:-64}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"
FORCE="${FORCE:-0}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required for the Tau user simulator}"
for path in "$MODEL_SPECS_FILE" "$QUALIFICATION_MANIFEST" "$TAU2_DATA_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done

mkdir -p "$RUN_DIR/data" "$RUN_DIR/results"
"$PYTHON" "$SCRIPT_DIR/prepare_tau_training.py" \
    --qualification-manifest "$QUALIFICATION_MANIFEST" \
    --output-dir "$RUN_DIR/data" \
    --train-steps 1 \
    --airline 4 \
    --retail 4 \
    --val-airline 20 \
    --val-retail 40

export TAU2_DATA_DIR
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

run_one() {
    local model_id="$1"
    local model_path="$2"
    local checkpoint_path="$3"
    local seed="$4"
    local output_dir="$RUN_DIR/results/$model_id/seed_$seed"
    local metrics_dir="$output_dir/raw"
    local done_file="$output_dir/.done"

    if [[ -f "$done_file" && "$FORCE" != 1 ]]; then
        echo "Skipping completed $model_id seed=$seed"
        return
    fi
    if [[ "$FORCE" == 1 ]]; then
        rm -rf "$output_dir"
    fi
    mkdir -p "$metrics_dir" "$output_dir/hydra"

    local resume_mode=disable
    local resume_path=null
    if [[ -n "$checkpoint_path" && "$checkpoint_path" != "-" ]]; then
        if [[ ! -d "$checkpoint_path/actor" ]]; then
            echo "ERROR: checkpoint must be a global_step_* directory: $checkpoint_path" >&2
            exit 1
        fi
        resume_mode=resume_path
        resume_path="$checkpoint_path"
    fi

    echo "Evaluating $model_id seed=$seed (20 Airline + 40 Retail)"
    "$PYTHON" -m verl.trainer.main_ppo \
        --config-name tau_outcome \
        data.train_files="$RUN_DIR/data/train.parquet" \
        data.val_files="$RUN_DIR/data/validation.parquet" \
        data.train_batch_size=8 \
        data.val_batch_size=60 \
        data.max_prompt_length="$MAX_PROMPT" \
        data.max_response_length="$MAX_RESPONSE" \
        data.filter_overlong_prompts=False \
        data.truncation=left \
        data.return_raw_chat=True \
        +data.dataloader_num_workers=0 \
        +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
        reward_model.reward_manager=dapo_turn \
        algorithm.adv_estimator=dapo \
        algorithm.filter_groups.enable=False \
        algorithm.use_kl_in_reward=False \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.multi_turn.enable=true \
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
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.min_p=0.0 \
        actor_rollout_ref.rollout.val_kwargs.seed="$seed" \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        env.seed=0 \
        env.rollout.n=1 \
        env.tau.eval_seed="$seed" \
        env.tau.trajectory_counts.airline=4 \
        env.tau.trajectory_counts.retail=4 \
        env.tau.validation_counts.airline=20 \
        env.tau.validation_counts.retail=40 \
        env.tau.qualification_manifest="$QUALIFICATION_MANIFEST" \
        trainer.total_training_steps=1 \
        trainer.total_epochs=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 \
        trainer.balance_batch=False \
        trainer.n_gpus_per_node="$N_GPUS" \
        trainer.nnodes=1 \
        trainer.logger='[console]' \
        trainer.project_name=tau_eval \
        trainer.experiment_name="${model_id}_seed_${seed}" \
        trainer.default_local_dir="$output_dir/checkpoints" \
        trainer.validation_data_dir="$metrics_dir" \
        trainer.resume_mode="$resume_mode" \
        trainer.resume_from_path="$resume_path" \
        hydra.run.dir="$output_dir/hydra" \
        +ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$output_dir/eval.log"

    compgen -G "$metrics_dir/*.metrics.json" >/dev/null || {
        echo "ERROR: no metrics JSON produced for $model_id seed=$seed" >&2
        exit 1
    }
    printf 'model_id=%s\nseed=%s\ncompleted_at=%s\n' \
        "$model_id" "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$done_file"
}

while IFS=$'\t' read -r model_id model_path checkpoint_path extra; do
    [[ -z "$model_id" || "$model_id" == \#* ]] && continue
    if [[ -n "${extra:-}" ]]; then
        echo "ERROR: expected 2 or 3 tab-separated columns in $MODEL_SPECS_FILE" >&2
        exit 1
    fi
    if [[ ! "$model_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: unsafe model id: $model_id" >&2
        exit 1
    fi
    if [[ ! -d "$model_path" ]]; then
        echo "ERROR: model path does not exist: $model_path" >&2
        exit 1
    fi
    checkpoint_path="${checkpoint_path:--}"
    for seed in $SEEDS; do
        run_one "$model_id" "$model_path" "$checkpoint_path" "$seed"
    done
done < "$MODEL_SPECS_FILE"

"$PYTHON" "$SCRIPT_DIR/summarize_tau_eval.py" --results-dir "$RUN_DIR/results"
