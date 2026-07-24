#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME=dapo_games_non_vpr_mixed
export TRAINING_VARIANT="outcome GRPO games"
export PROJECT_NAME=dapo-games-non-vpr-mixed
export RUN_NAME="${RUN_NAME:-dapo_games_non_vpr_base_3_4_9_boxed_soko15_sudoku40_target12_mines4_outcome}"
export MATH_TRAJ="${MATH_TRAJ:-64}"
export SOKOBAN_TRAJ="${SOKOBAN_TRAJ:-3}"
export SUDOKU_TRAJ="${SUDOKU_TRAJ:-4}"
export MINESWEEPER_TRAJ="${MINESWEEPER_TRAJ:-9}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export GAME_ACTION_FORMAT="${GAME_ACTION_FORMAT:-boxed}"
export DIM_ROOM="${DIM_ROOM:-6,6}"
export NUM_BOXES="${NUM_BOXES:-2}"
export SOKOBAN_MAX_STEPS="${SOKOBAN_MAX_STEPS:-15}"
export NUM_BLANKS="${NUM_BLANKS:-40}"
export SUDOKU_MAX_STEPS="${SUDOKU_MAX_STEPS:-15}"
export SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS="${SUDOKU_OUTCOME_SUCCESS_CORRECT_FILLS:-12}"
export NUM_MINES="${NUM_MINES:-4}"
export MINESWEEPER_MAX_STEPS="${MINESWEEPER_MAX_STEPS:-15}"

exec "$SCRIPT_DIR/run_qwen3_4b_base_vpr_mixed.sh"
