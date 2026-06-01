#!/bin/bash
# GRPO smoke test for vpr_tictactoe — runs 2 training steps with Qwen3-4B.
set -euo pipefail

MODEL_PATH="/mnt/project_rlinf/yuanhuining/models/Qwen3-4B/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== VPR TicTacToe GRPO Smoke Test ==="
echo "Model: $MODEL_PATH"

python3 -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    env.env_name=vpr_tictactoe \
    env.seed=0 \
    env.history_length=0 \
    env.max_steps=9 \
    env.invalid_penalty=-1.0 \
    env.resources_per_worker.num_cpus=0.1 \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=2 \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    trainer.total_training_steps=2 \
    trainer.test_freq=2 \
    trainer.save_freq=-1 \
    algorithm.adv_estimator=vpr \
    algorithm.vpr.outcome_reward_scale=1.0 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.max_tokens=64 \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.logger=[]

echo "=== TicTacToe smoke test PASSED ==="
