#!/bin/bash
# GRPO smoke test for vpr_tictactoe — runs 2 training steps with Qwen3-4B.
# Asserts per-row invariants: terminal-only outcome bonus, bounded prompts,
# multi-turn reward preservation. Preserves log + evidence file.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/yuanhuining/models/Qwen3-4B}"
PYTHON="${PYTHON:-/opt/venv/verl-agent/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_tictactoe"
LOG_DIR="$SCRIPT_DIR/smoke_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/tictactoe_${TS}.log"
EVIDENCE_FILE="$LOG_DIR/tictactoe_${TS}.evidence.json"

echo "=== VPR TicTacToe GRPO Smoke Test ==="
echo "Model:    $MODEL_PATH"
echo "Log:      $LOG_FILE"
echo "Evidence: $EVIDENCE_FILE"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH" >&2
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preparing data..."
    "$PYTHON" "$VPR_GAMES_DIR/prepare_data.py" \
        --env-name vpr_tictactoe --train-size 2 --val-size 1 \
        --output-dir "$DATA_DIR"
fi

VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
VPR_SMOKE_EVIDENCE="$EVIDENCE_FILE" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_tictactoe \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    env.seed=0 \
    env.rollout.n=2 \
    algorithm.use_kl_in_reward=False \
    trainer.total_training_steps=2 \
    trainer.test_freq=2 \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.logger=["console"] \
    trainer.resume_mode=disable \
    +ray_init.num_cpus=16 2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Verifying smoke test evidence ==="
"$PYTHON" "$SCRIPT_DIR/smoke_verify.py" "$LOG_FILE" "$EVIDENCE_FILE"

echo ""
echo "Log preserved at:      $LOG_FILE"
echo "Evidence preserved at: $EVIDENCE_FILE"
echo "=== TicTacToe smoke test PASSED ==="
