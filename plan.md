# VPR Environments Integration into verl-agent

## Goal Description

Implement and register three VPR training environments — `vpr_tictactoe`, `vpr_sudoku`, and `vpr_minesweeper` — into the verl-agent framework. Each environment follows the VPR paper's Markovian per-step training paradigm: the model receives only the current game state at each step with no history context, and earns a dense oracle reward for each action. The implementation must support standard GRPO training using the local Qwen3-4B model at `/mnt/project_rlinf/yuanhuining/models/Qwen3-4B/`.

Environment sources:
- **vpr_tictactoe**: Implemented directly in this repository. Oracle verifier uses exact minimax.
- **vpr_sudoku**: Wraps the `gem` library (`https://github.com/axon-rl/gem`) for game logic; adds VPR reward computation.
- **vpr_minesweeper**: Wraps the `gem` library for game logic; adds VPR posterior-based reward computation.

Paper-default configurations: Minesweeper 5×5 with 5 mines; Sudoku 9×9 with 40 blank cells; TicTacToe 3×3.

All environments use 1-indexed coordinates (draft specification). GEM is used exclusively for game state management; the VPR layer owns all parsing, coordinate mapping, and reward computation. The VPR paper PDF at `/mnt/project_rlinf/yuanhuining/repos/VPR-v2/vpr_arxiv_v2.pdf` is available for reference; install `pypdf` with `pip` if PDF parsing is needed.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: All three environment names are recognized by `make_envs()` and dispatchable to correctly sized train and validation environment pools with correct seeding.
  - Positive Tests (expected to PASS):
    - `make_envs(config)` with `env_name="vpr_tictactoe"`, `train_batch_size=4`, `rollout.n=2` returns train envs with 8 Ray actors and val envs seeded at `seed+1000` with `group_n=1`.
    - Same test for `vpr_sudoku` and `vpr_minesweeper` produces matching actor counts.
    - `VPRBaseEnvironmentManager` initialized with `history_length=0` succeeds without error.
  - Negative Tests (expected to FAIL):
    - `make_envs(config)` with `env_name="vpr_unknown"` calls `exit(1)` with "Environment not supported".
    - `VPRBaseEnvironmentManager` initialized with `history_length=2` raises `ValueError` before any episode begins.

- AC-2: Each environment prompt is Markovian — it contains only the current game state and action format, leveraging verl-agent's built-in step-independent multi-turn rollout mechanism. VPR environments set `history_length=0` in Hydra config; `VPRBaseEnvironmentManager` validates this defensively at initialization.
  - Positive Tests (expected to PASS):
    - After 5 steps in a TicTacToe episode, `build_text_obs()` returns a prompt that contains the current board state and does not reference any past actions or prior observations.
    - Prompt character length after step 10 equals prompt character length after step 1, modulo changes in board content (not history accumulation).
    - A Hydra config with `env.history_length=0` loads and initializes the manager without error.
  - Negative Tests (expected to FAIL):
    - A prompt generated at step 5 does not contain any text fragment from the action taken at step 3.
    - Providing `history_length=1` in the Hydra config while using a VPR environment manager raises `ValueError` at manager initialization time, catching the misconfiguration before any rollout begins.

- AC-3: Dense VPR rewards are computed per step following the specified formulas for each environment; oracle reward is +1.0, legal-non-oracle is 0.0, invalid action receives the configured penalty (default −1.0).
  - Positive Tests (expected to PASS):
    - TicTacToe: at a board state with exactly one minimax-optimal move, that move receives `+1.0`; all other legal moves receive `0.0`.
    - Sudoku: placing the solution-correct digit at a blank cell returns `+1.0`; placing a wrong digit at a blank cell returns `0.0` (or triggers termination per `terminate_on_wrong_digit`).
    - Minesweeper: on a board with known mine positions, revealing the cell with minimum posterior mine probability receives `+1.0`; revealing a legal but non-minimum-probability unrevealed cell receives `0.0`; flagging a cell whose exact integer posterior equals 1 receives `+1.0`.
    - Minesweeper flag-toggle: flagging an already-flagged cell (un-flag) returns `0.0` (legal non-oracle).
    - Minesweeper first step (GEM first-click safety active): any reveal receives `+1.0`.
  - Negative Tests (expected to FAIL):
    - TicTacToe: occupying an already-played cell returns the configured penalty, not `0.0`.
    - Sudoku: placing a digit on a given (pre-filled) cell returns the configured penalty.
    - Minesweeper: unparseable action returns the configured penalty and `parse_ok=False` in info.
    - Minesweeper: flagging a cell with posterior < 1.0 does not return `+1.0`.

- AC-4: Every `step()` call returns an info dict with all required fields, JSON-serializable values, and `is_action_valid`; `success_evaluator()` uses `terminal_success` from info.
  - Positive Tests (expected to PASS):
    - Every `step()` result includes: `env_name`, `step`, `max_steps`, `raw_action`, `parsed_action`, `parse_ok`, `illegal_action`, `available_actions`, `vpr_reward`, `terminal_success`, `terminal_reason`, `is_action_valid`.
    - Minesweeper info additionally contains: `posterior_min_prob`, `posterior_prob_for_action`, `oracle_valid_actions`, `completion_rate`, `oracle_degraded`.
    - TicTacToe info additionally contains: `game_result`, `oracle_valid_actions`, `opponent_action`.
    - Sudoku info additionally contains: `num_blanks_remaining`, `completion_rate`.
    - `json.dumps(info)` succeeds without error for any returned info dict.
    - `success_evaluator()` returns `True` when `info["terminal_success"]` is `True`.
  - Negative Tests (expected to FAIL):
    - A schema validation test that removes any required field from the returned info dict causes an `AssertionError`.
    - `success_evaluator()` does not read `info["won"]` directly (the override must use `terminal_success`).

- AC-5: The `<action>...</action>` parser never crashes on any input; invalid actions are intercepted before reaching GEM via a sentinel path.
  - Positive Tests (expected to PASS):
    - `"<think>ok</think><action> 3 </action>"` parses to `action_text="3"` with `parse_ok=True`.
    - `"<action>reveal 2 3</action>"` parses with `action_text="reveal 2 3"`, `parse_ok=True`.
    - `"<action>open 1 1</action>"` (alias) expands to `"reveal 1 1"` with `parse_ok=True`.
    - `"...<action>5</action>...<action>7</action>"` (repeated tags) extracts the last: `action_text="7"`.
    - Completely empty string input returns `ParseResult(raw_action="", action_text=None, parse_ok=False, error="no_action_tag")`.
    - `"<action></action>"` (empty tag) returns `parse_ok=False`.
  - Negative Tests (expected to FAIL):
    - `parse("<action>abc</action>")` for TicTacToe returns `illegal_action=True` in info (non-numeric cell).
    - GEM is never called when `parse_ok=False` — verified by test stub confirming GEM receives no call.

- AC-6: Unit tests pass for all three environments, covering parser edge cases, seeded determinism, grouped-reset identity, reward correctness, termination, info schema, and `make_envs()` actor counts.
  - Positive Tests (expected to PASS):
    - `pytest tests/vpr_games/` exits with code 0.
    - TicTacToe: at a known board state with one blocking optimal move, `oracle_valid_actions` contains exactly that cell; reward for that cell is `+1.0`.
    - Minesweeper: on a 3×3 board with known mine placement, `compute_posteriors()` output matches brute-force enumeration cell-by-cell.
    - Sudoku: known 9×9 puzzle with verified unique solution; placing correct digit at blank cell (3,4) returns `+1.0`.
    - `env.reset(seed=42)` called twice on the same environment produces identical initial observations.
    - Within a GRPO group (`group_n=2`), both replicas receive the same initial board after `reset()`.
    - `make_envs()` for training creates exactly `train_batch_size × rollout.n` Ray actors.
  - Negative Tests (expected to FAIL):
    - Seeded reset with `seed=42` and `seed=43` produces different initial boards.
    - Manually injecting a wrong posterior probability causes the brute-force comparison assertion to fail.
    - TicTacToe: a non-optimal legal move does not receive `+1.0`.

- AC-7: GRPO smoke scripts complete 1–3 training steps with Qwen3-4B; per-step rewards are confirmed as distinct values in training tensors; prompts do not grow with rollout length.
  - Positive Tests (expected to PASS):
    - `bash examples/vpr_games/grpo_tictactoe_smoke.sh` completes without error and exits with code 0.
    - Same for `grpo_sudoku_smoke.sh` and `grpo_minesweeper_smoke.sh`.
    - Training batch reward tensor contains at least 2 distinct non-zero values across steps of a sampled episode (confirming per-step, not episode-collapsed rewards).
    - The prompt string logged at step 5 of any episode does not contain content from steps 1–4.
  - Negative Tests (expected to FAIL):
    - Running any smoke script with a nonexistent model path produces a clear file-not-found error, not a silent hang.
    - Running a smoke script with `history_length=1` in the config raises `ValueError` before training begins.

- AC-8: The GRPO training pipeline uses VPR turn-level reward normalization for advantage estimation and a standard outcome reward mechanism for terminal signals.
  - AC-8.1: VPR turn-level advantage estimation — for each turn position t, rewards r_t across all batch episodes that reached at least t steps are normalized: advantage_t = (r_t − mean_t) / (std_t + ε). When fewer than 4 same-position turns are available for a given t, fall back to normalization across all active turns in the entire batch.
    - Positive Tests (expected to PASS):
      - The training pipeline audit (`task0`) produces a document naming source files and the exact tensor path from `env.step()` through advantage estimation.
      - `task0b` implements the VPR turn-level advantage estimator; a unit test verifies that turn-3 rewards are normalized independently from turn-1 rewards within the same batch.
      - When only 2 episodes in the batch reach turn 5, the fallback activates and normalizes those turn-5 rewards across the entire batch; a unit test verifies this boundary case.
    - Negative Tests (expected to FAIL):
      - A regression guard test that applies global normalization uniformly to all turns produces different advantage values than the per-turn algorithm, confirming the guard detects regressions.
  - AC-8.2: Standard outcome reward — at episode termination, a configurable terminal outcome reward (default +1.0 for `terminal_success=True`, 0.0 otherwise) is added to the final step's reward, separate from and additive to the VPR oracle reward for that step.
    - Positive Tests (expected to PASS):
      - A winning TicTacToe episode has the outcome reward (+1.0) present at the last step in addition to the VPR oracle reward for that move.
      - A losing TicTacToe episode has outcome reward 0.0 at the last step (only VPR oracle reward applies).
      - Setting `outcome_reward_scale=0.0` disables the terminal bonus; the final step's reward equals the VPR oracle reward only.
      - Smoke scripts log outcome rewards as a separate metric alongside VPR oracle rewards.
    - Negative Tests (expected to FAIL):
      - Outcome reward does NOT appear at non-final steps — only at the terminal step of each episode.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes all three environments fully integrated into verl-agent with shared infrastructure (`VPRBaseEnvironmentManager`, shared parser, common info schema), complete unit tests for all environments including a Minesweeper posterior brute-force validation test, GRPO smoke scripts for all three environments, Hydra config overrides with paper-default settings, full training pipeline audit, VPR turn-level advantage estimator, standard outcome reward mechanism, and optional VPR PDF reference parsing via `pypdf` if environment definitions require clarification.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes all three environments registered in `make_envs()`, a shared parser, unit tests covering reward correctness and termination for each environment, VPR turn-level advantage estimation with fallback, standard outcome reward at episode end, and GRPO smoke scripts that complete 1–3 steps without crashing with per-step reward and outcome reward confirmed.

### Allowed Choices

- Can use: Ray actors for parallelism; `gem` library for Sudoku and Minesweeper game state transitions; exact minimax with alpha-beta pruning for TicTacToe oracle; frontier decomposition with lexicographic enumeration for Minesweeper posterior; `pytest` for tests; Hydra config inheritance from `ppo_trainer.yaml`; `pypdf` for VPR PDF parsing if needed; `numpy` for reward computation.
- Cannot use: Full interaction history in any VPR environment prompt; GEM's built-in parsing, reward scheme, or action format; nested GEM vectorization inside Ray actors (single parallelism layer only); wall-clock timeouts for oracle computation (use iteration budget N_max=50000 instead); MCTS for TicTacToe oracle (user decision: exact minimax chosen).
- Fixed per design: 1-indexed coordinates for all three environments; GEM first-click safe neighborhood preserved; reveal-only Minesweeper episode success (flagging not required); `<action>...</action>` action format with `<think>...</think>` optional prefix; dense VPR rewards with oracle=+1.0, legal-non-oracle=0.0, configurable penalty (default −1.0); `terminate_on_wrong_digit=True` default for Sudoku.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Shared Parser Module**

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class ParseResult:
    raw_action: str
    action_text: Optional[str]
    parse_ok: bool
    error: Optional[str]

ALIASES = {"open": "reveal", "click": "reveal", "mark": "flag"}

def parse_action_tag(text: str) -> ParseResult:
    raw = text or ""
    matches = re.findall(r"<action>(.*?)</action>", raw, re.DOTALL | re.IGNORECASE)
    if not matches:
        return ParseResult(raw_action=raw, action_text=None, parse_ok=False, error="no_action_tag")
    last = matches[-1].strip()
    if not last:
        return ParseResult(raw_action=raw, action_text=None, parse_ok=False, error="empty_action_tag")
    for alias, canonical in ALIASES.items():
        if last.lower().startswith(alias + " "):
            last = canonical + last[len(alias):]
            break
    return ParseResult(raw_action=raw, action_text=last, parse_ok=True, error=None)
```

**TicTacToe Minimax Oracle**

```python
def minimax(board: list, is_agent_turn: bool, alpha=-2, beta=2) -> int:
    winner = check_winner(board)
    if winner == AGENT: return 1
    if winner == OPPONENT: return -1
    if all(c != EMPTY for c in board): return 0
    if is_agent_turn:
        best = -2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = AGENT
                best = max(best, minimax(board, False, alpha, beta))
                board[i] = EMPTY
                alpha = max(alpha, best)
                if beta <= alpha: break
        return best
    else:
        best = 2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = OPPONENT
                best = min(best, minimax(board, True, alpha, beta))
                board[i] = EMPTY
                beta = min(beta, best)
                if beta <= alpha: break
        return best

def oracle_valid_actions(board: list, agent_mark: str) -> list[str]:
    best_val = max(minimax(board_copy_with(board, i, agent_mark), False)
                   for i in range(9) if board[i] == EMPTY)
    return [str(i + 1) for i in range(9)
            if board[i] == EMPTY
            and minimax(board_copy_with(board, i, agent_mark), False) == best_val]
```

**Minesweeper Posterior Oracle (conceptual)**

```python
def compute_posteriors(revealed_grid, total_mines: int) -> dict[tuple, float]:
    frontier = find_frontier_cells(revealed_grid)
    components = decompose_components(frontier, revealed_grid)
    component_enumerations = []
    total_configs = 0
    mine_counts_at = defaultdict(int)
    n_max = 50000
    steps = 0
    for comp_cells in components:
        valid_assignments = []
        for assignment in enumerate_assignments_lexicographic(comp_cells, revealed_grid):
            if steps >= n_max:
                return fallback_local_deduction(revealed_grid, total_mines)
            steps += 1
            valid_assignments.append(assignment)
        component_enumerations.append((comp_cells, valid_assignments))
    frontier_cells = set(c for comp_cells, _ in component_enumerations for c in comp_cells)
    unconstrained = find_unconstrained_hidden_cells(revealed_grid, frontier_cells)
    for combination in aggregate_with_global_constraint(component_enumerations, total_mines, len(unconstrained)):
        frontier_mines, component_assignment_indices = combination
        remaining = total_mines - frontier_mines
        weight = comb(len(unconstrained), remaining)
        total_configs += weight
        for cell, has_mine in component_assignment_indices:
            if has_mine:
                mine_counts_at[cell] += weight
        for cell in unconstrained:
            mine_counts_at[cell] += weight * remaining / max(len(unconstrained), 1)
    return {cell: mine_counts_at[cell] / total_configs for cell in frontier_cells | set(unconstrained)}
```

**VPRBaseEnvironmentManager skeleton**

```python
class VPRBaseEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        if getattr(config.env, "history_length", 0) != 0:
            raise ValueError(
                f"VPR environments require history_length=0, got {config.env.history_length}"
            )
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        return {"text": self.build_text_obs(infos), "image": None, "anchor": None}, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_observations = {"text": self.build_text_obs(infos), "image": None, "anchor": None}
        for i, info in enumerate(infos):
            info["is_action_valid"] = int(valids[i])
        return next_observations, rewards, dones, infos

    def success_evaluator(self, **kwargs):
        infos = kwargs.get("infos", [{}])
        return {"success": np.array([info.get("terminal_success", False) for info in infos])}
```

**make_envs() registration pattern (per environment)**

```python
elif "vpr_tictactoe" in config.env.env_name.lower():
    from agent_system.environments.env_package.vpr_games.tictactoe import (
        build_tictactoe_envs, tictactoe_projection
    )
    _envs = build_tictactoe_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size,
        group_n=group_n, is_train=True, env_config=config.env
    )
    _val_envs = build_tictactoe_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
        group_n=1, is_train=False, env_config=config.env
    )
    envs = TicTacToeEnvironmentManager(_envs, partial(tictactoe_projection), config)
    val_envs = TicTacToeEnvironmentManager(_val_envs, partial(tictactoe_projection), config)
    return envs, val_envs
```

**Smoke Script Pattern**

```bash
#!/bin/bash
MODEL_PATH="/mnt/project_rlinf/yuanhuining/models/Qwen3-4B/"
python main_ppo.py \
    --config-name ppo_trainer \
    env.env_name=vpr_tictactoe \
    env.seed=0 \
    env.history_length=0 \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    env.rollout.n=2 \
    trainer.total_training_steps=2 \
    model.path="$MODEL_PATH" \
    trainer.logger=[]
```

### Relevant References

- `agent_system/environments/env_manager.py` — `make_envs()` factory and elif-chain registration; existing manager class implementations
- `agent_system/environments/base.py` — `EnvironmentManagerBase` class to subclass
- `agent_system/environments/env_package/gym_cards/envs.py` — simple Ray actor + parallel env builder pattern
- `agent_system/environments/env_package/sokoban/envs.py` — Ray actor wrapping external game logic; `SimpleMemory` usage pattern (to avoid for VPR)
- `agent_system/environments/prompts/gym_cards.py` — prompt template definition pattern
- `agent_system/memory/memory.py` — `SimpleMemory` class (VPR environments do NOT use this; shown as contrast)
- `verl/trainer/config/ppo_trainer.yaml` — base Hydra config; VPR configs use `defaults: [ppo_trainer, _self_]`
- `examples/grpo_trainer/run_sokoban.sh` — GRPO script structure to follow for smoke scripts
- `agent_system/reward_manager/` — contains `EpisodeRewardManager`; audit target for task0 reward-path tracing

## Dependencies and Sequence

### Milestones

1. **Training Pipeline Audit and Reward Architecture**: Establish the per-turn VPR reward and advantage estimation path, and add the standard outcome reward mechanism.
   - Audit: Trace reward tensor from `env.step()` through the reward manager, token-level assignment, and advantage estimation; name source files. Confirm `EpisodeRewardManager` collapse behavior and identify where turn-level rewards are stored.
   - Implement VPR turn-level advantage estimator: normalize rewards per turn position across the batch, with fallback to batch-wide normalization when fewer than 4 same-position turns are available.
   - Implement outcome reward: add configurable terminal bonus at the final step of each episode, additive to VPR oracle reward.

2. **Shared Infrastructure**: Build foundations shared across all three environments.
   - Parser: `ParseResult` dataclass, tag extraction, alias expansion, sentinel handling, never-crash guarantee.
   - Base class: `VPRBaseEnvironmentManager` with `history_length=0` enforcement, standard `{"text", "image", "anchor"}` return, `success_evaluator()` override, `is_action_valid` emission.

3. **TicTacToe Environment**: Full direct implementation (no GEM dependency).
   - Game logic: Board state, move validation, minimax oracle with alpha-beta pruning, random opponent.
   - Ray actor: `TicTacToeWorker`, `TicTacToeMultiProcessEnv`, `build_tictactoe_envs()`.
   - Manager: `TicTacToeEnvironmentManager`, prompt template, `make_envs()` registration.

4. **GEM Feasibility Spike**: Verify GEM API before building GEM-backed environments.
   - Install GEM; instantiate Sudoku and Minesweeper inside Ray actors.
   - Inspect state access (grid, solution, mine map), seeding behavior, action format, completion signals.
   - Prototype `compute_posteriors()` on a 3×3 board; verify against brute-force.

5. **Sudoku Environment**: GEM-backed Sudoku wrapper.
   - Adapter: Map 1-indexed (row, col, digit) to GEM 0-indexed; extract grid and solution; validate blank-cell assumption.
   - Oracle: `solution[row-1][col-1] == digit` check for blank cells.
   - Ray actor: `SudokuWorker`, `SudokuMultiProcessEnv`, `build_sudoku_envs()`.
   - Manager: `SudokuEnvironmentManager`, prompt template (lists blank cells, not all triples), `make_envs()` registration.

6. **Minesweeper Environment**: GEM-backed Minesweeper with posterior oracle.
   - Oracle: `compute_posteriors()` with frontier decomposition, N_max=50000 lexicographic enumeration, global mine-count combinatorial aggregation, oracle_degraded flag.
   - Adapter: 1-indexed to 0-indexed mapping; flag-toggle behavior (flag on flagged cell = un-flag = 0.0); reveal-only completion override (override GEM native completion signal); mine-hit termination; invalid-action sentinel before GEM call.
   - Ray actor: `MinesweeperWorker`, `MinesweeperMultiProcessEnv`, `build_minesweeper_envs()`.
   - Manager: `MinesweeperEnvironmentManager`, prompt template (lists unrevealed cells compactly), `make_envs()` registration.

7. **Integration and Verification**: Wire, test, and confirm end-to-end.
   - Hydra configs with paper-default settings.
   - Full unit test suite.
   - GRPO smoke scripts for all three environments.
   - Confirmation run: pytest green, smoke scripts exit 0, per-step rewards confirmed distinct.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task0 | Audit GRPO training pipeline: trace reward tensor from `env.step()` through `EpisodeRewardManager`, token-level assignment, and advantage computation; confirm `EpisodeRewardManager` collapse behavior; identify where per-step rewards from `batch.non_tensor_batch['rewards']` are stored; produce document naming source files and exact tensor path | AC-8 | analyze | — |
| task0b | Implement VPR turn-level advantage estimator: for each turn position t, normalize rewards across same-position turns in batch; fall back to batch-wide normalization when fewer than 4 same-position turns are available; add unit tests for per-turn normalization and fallback boundary condition | AC-8 | coding | task0 |
| task0c | Implement standard outcome reward mechanism: add configurable terminal bonus (default +1.0 for success, 0.0 for failure) at the final step of each episode, additive to VPR oracle reward; expose `outcome_reward_scale` config param; add unit test and update smoke scripts to log outcome rewards separately | AC-8 | coding | task0 |
| task1 | GEM feasibility spike: install gem, instantiate Sudoku and Minesweeper inside Ray actors, inspect state access (grid, solution, mine positions), seeding, action format, completion signals; prototype posterior oracle on 3×3 board and verify against brute-force | AC-6 | analyze | — |
| task2 | Implement shared parser module: `ParseResult` dataclass, last-action-tag extraction, alias expansion (open→reveal, click→reveal, mark→flag), sentinel path to block GEM calls on parse failure, never-crash guarantee | AC-5 | coding | — |
| task3 | Implement TicTacToe game logic: board state representation, move validation, minimax oracle with alpha-beta pruning (cells 1..9, 1-indexed), `oracle_valid_actions()` function, random opponent | AC-3 | coding | task2 |
| task4 | Implement TicTacToe Ray actor (`TicTacToeWorker`), parallel env (`TicTacToeMultiProcessEnv`), builder (`build_tictactoe_envs()`); implement `TicTacToeEnvironmentManager` (subclasses `VPRBaseEnvironmentManager`); add prompt template; register in `make_envs()` | AC-1, AC-2, AC-4 | coding | task3 |
| task5 | Implement Sudoku GEM adapter: 1-indexed (row, col, digit) to 0-indexed GEM mapping, grid and solution extraction from GEM state, VPR oracle reward computation, `terminate_on_wrong_digit` logic, blank-cell legality check; Ray actor (`SudokuWorker`), parallel env, builder | AC-3 | coding | task1, task2 |
| task6 | Implement `SudokuEnvironmentManager` (subclasses `VPRBaseEnvironmentManager`); add Sudoku prompt template (lists blank cell coordinates, not all (row, col, digit) triples); register in `make_envs()` | AC-1, AC-2, AC-4 | coding | task5 |
| task7 | Implement Minesweeper posterior oracle: frontier identification, connected-component decomposition, lexicographic enumeration with N_max=50000 step budget, global mine-count combinatorial aggregation across components and unconstrained cells, local single-constraint deduction fallback, `oracle_degraded` flag; integer comparison for flag reward (posterior == 1 using exact numerator/denominator) | AC-3 | coding | task1, task2 |
| task8 | Implement Minesweeper GEM adapter: 1-indexed to 0-indexed coordinate mapping, flag-toggle behavior (flag on already-flagged = un-flag = 0.0), reveal-only completion override, mine-hit termination, invalid-action sentinel before GEM call; Ray actor (`MinesweeperWorker`), parallel env, builder | AC-3, AC-5 | coding | task7 |
| task9 | Implement `MinesweeperEnvironmentManager` (subclasses `VPRBaseEnvironmentManager`); add Minesweeper prompt template (lists unrevealed cells and flagged cells compactly; does not list every possible reveal/flag combination); register in `make_envs()` | AC-1, AC-2, AC-4 | coding | task8 |
| task10 | Add `VPRBaseEnvironmentManager` shared base class with `history_length=0` ValueError guard, standard return shape, `success_evaluator()` override; add Hydra config files for all three environments with paper-default settings (Minesweeper: rows=5, cols=5, mines=5; Sudoku: 9×9, 40 blanks; TicTacToe: 3×3, opponent=random, history_length=0) | AC-1, AC-2 | coding | task4, task6, task9 |
| task11 | Add unit tests: parser edge cases (empty, missing tag, repeated tags, aliases, whitespace), seeded reset determinism, grouped-reset identity (group_n=2 replicas identical), TicTacToe minimax reward test at known board state, Sudoku known-solution reward test, Minesweeper 3×3 posterior brute-force comparison, info schema validation for all three envs, make_envs() actor count test, invalid-action termination per-environment | AC-5, AC-6 | coding | task10 |
| task12 | Add GRPO smoke scripts (`grpo_tictactoe_smoke.sh`, `grpo_sudoku_smoke.sh`, `grpo_minesweeper_smoke.sh`) using Qwen3-4B with num_train_steps=2, train_batch_size=2, rollout.n=2; assert per-turn reward preservation and distinct advantages per turn position; assert outcome reward logged at terminal step; add training regression guard test | AC-7 | coding | task11, task0b, task0c |
| task13 | Run `pytest tests/vpr_games/` and all three smoke scripts; verify per-turn advantage normalization produces distinct values per turn position; confirm outcome reward appears only at terminal step; confirm prompt length does not grow across 10 steps | AC-7, AC-8 | analyze | task12 |

## Claude-Codex Deliberation

### Agreements

Both Claude and all Codex passes agreed on:
- `make_envs()` elif-chain is the correct registration pattern for verl-agent
- Single Ray parallelism layer; GEM objects instantiated inside actors to avoid serialization issues
- Shared parser returning structured `ParseResult` before any GEM call; projection functions convert to `(actions, valids)` tuple matching `EnvironmentManagerBase`
- `history_length=0` set in Hydra config; `VPRBaseEnvironmentManager` validates this defensively. This leverages verl-agent's native step-independent multi-turn rollout mechanism (confirmed in README and `rollout_loop.py`): the novel VPR contribution is in the reward/advantage layer, not in creating a new rollout mechanism.
- The default `EpisodeRewardManager` collapses rewards to episode level (confirmed: places single `episode_rewards` value at the last response token). Per-step rewards from `batch.non_tensor_batch['rewards']` are stored in the rollout loop but not used for advantage computation by default — task0b implements the VPR turn-level advantage estimator.
- Train actors: `env_num=train_batch_size`, `group_n=rollout.n`; validation: `group_n=1`, `seed+1000`
- Sudoku oracle is O(1) lookup against stored unique solution from GEM
- Minesweeper posterior: frontier decomposition with global mine-count combinatorial aggregation for unconstrained cells is the correct exact approach
- VPR environments must return `{"text", "image", "anchor"}`, emit `is_action_valid`, and override `success_evaluator()` for verl-agent compatibility
- Invalid-action sentinel path prevents GEM calls with structurally invalid inputs
- Training pipeline must be audited; VPR turn-level advantage estimation and standard outcome reward are first-class requirements

### Resolved Disagreements

- **Coordinate indexing** (Round 1 Codex DISAGREE → DEC-3): Codex cited paper's mixed indexing (0-indexed for TicTacToe/Minesweeper, 1-indexed for Sudoku). Draft specifies uniform 1-indexed. User confirmed: 1-indexed uniformly for all three environments. Coordinate mapping to 0-indexed GEM is handled internally per environment.

- **TicTacToe oracle** (Round 1 Codex DISAGREE → DEC-1): Paper uses MCTS with 10,000 simulations and mixed training opponents. Draft explicitly permits exact minimax given TicTacToe's small size. User confirmed: exact minimax. Documented divergence: for 3×3 TicTacToe, exact minimax and MCTS with sufficient rollouts produce identical `oracle_valid_actions` sets for all reachable states; the practical reward difference is zero.

- **Wall-clock timeout for Minesweeper oracle** (Round 1 Required): Replaced with iteration budget N_max=50000 over lexicographically ordered frontier configurations. Fallback is deterministic local single-constraint deduction in the same sorted order. Oracle behavior is fully reproducible given game state, not machine-speed-dependent.

- **verl-agent manager contract gaps** (Round 1 Required): `VPRBaseEnvironmentManager` returns `{"text", "image", "anchor"}` (image=None, anchor=None for text-only games), emits `is_action_valid`, overrides `success_evaluator()` using `terminal_success`.

- **Per-environment invalid-action semantics** (Rounds 1–2 Required): TicTacToe: invalid/illegal action terminates episode, penalty applied. Minesweeper: invalid/illegal action terminates episode, penalty applied. Sudoku: wrong digit terminates when `terminate_on_wrong_digit=True` (default); applies penalty without termination when `False`.

- **Flagging reward contradiction** (Round 2 Required): Resolved under AC-3's legal-non-oracle rule: flag with posterior exactly 1 = +1.0 (oracle-valid); flag with posterior < 1.0 = 0.0 (legal non-oracle); un-flag (toggle off already-flagged cell) = 0.0 (legal non-oracle). Integer comparison used for posterior=1 check.

- **Minesweeper flag action grammar** (Round 3 Required): `flag` on an unflagged unrevealed cell = flag it (posterior determines reward). `flag` on an already-flagged cell = un-flag it = legal non-oracle (0.0). GEM's native toggle behavior preserved. Flagging a revealed cell or out-of-range cell = invalid action (sentinel path → penalty + termination).

- **Globally weighted posterior aggregation** (Round 3 Required): P(mine at cell c) = (configs_with_mine_at_c) / (total_valid_configs), where "valid" means consistent with all revealed clues AND satisfies global total mine count. Disconnected frontier components aggregated by summing over all combinations of per-component mine counts that satisfy the global constraint, each weighted by C(unconstrained_cells, remaining_mines_after_frontier_assignment).

- **GEM first-click safety** (DEC-2): Preserved — GEM's first-click safe neighborhood is kept; first step always receives +1.0; documented as expected behavior.

- **Minesweeper completion** (DEC-5): Reveal-only — episode success when all non-mine cells are revealed; flagging mines not required for completion. GEM native completion signal overridden in the VPR adapter.

### Convergence Status

- Final Status: `converged`
- Rounds executed: 3 (maximum) during gen-plan; refined via refine-plan (3 comments processed, 0 unresolved)
- All REQUIRED_CHANGES incorporated into the plan as explicit design decisions. All user decisions resolved. All annotation comments processed with no new pending decisions.

## Pending User Decisions

All pending decisions have been resolved.

- DEC-1: TicTacToe oracle verifier
  - Claude Position: Exact minimax (draft specification; provably correct for 3×3)
  - Codex Position: MCTS with 10,000 simulations (paper specification; mixed training opponents)
  - Tradeoff Summary: Exact minimax is deterministic, simpler, and produces identical oracle-valid action sets to MCTS for all 3×3 TicTacToe states
  - Decision Status: **Exact minimax chosen** (user confirmed; matches draft specification)

- DEC-2: GEM first-click safe neighborhood
  - Claude Position: Preserve GEM behavior (simpler; any first reveal is safe)
  - Codex Position: Open question
  - Tradeoff Summary: Preserving avoids forking GEM game logic; first step always gets +1.0 (minor training bias on step 1 only)
  - Decision Status: **Preserve GEM behavior** (user confirmed)

- DEC-3: Coordinate indexing
  - Claude Position: 1-indexed uniformly (draft specification)
  - Codex Position: 0-indexed for TicTacToe and Minesweeper per paper; 1-indexed for Sudoku
  - Tradeoff Summary: Uniform 1-indexed is consistent across environments; small notation deviation from paper; GEM mapping handled internally
  - Decision Status: **1-indexed uniformly for all three environments** (user confirmed; matches draft specification)

- DEC-5: Minesweeper episode completion
  - Claude Position: Follow GEM (flag all mines required for success)
  - Codex Position: Open question
  - Tradeoff Summary: Reveal-only is simpler for the model; reduces required flag oracle signal; requires overriding GEM completion logic
  - Decision Status: **Reveal-only completion** (user confirmed); GEM completion signal overridden in adapter

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

---

--- Original Design Draft Start ---

# Draft: Integrate VPR Environments into `verl-agent` with Humanize Agent Loop

## Goal

Use the Humanize agent loop to implement the three VPR training environments on top of `verl-agent`.

Environment sources:

- **Minesweeper**: integrate through `gem` (`https://github.com/axon-rl/gem`).
- **Sudoku**: integrate through `gem` (`https://github.com/axon-rl/gem`).
- **Tic-Tac-Toe**: implement directly in this repository.

The final implementation should allow `verl-agent` to run standard **GRPO** smoke tests using the local model:

```text
/mnt/project_rlinf/yuanhuining/models/Qwen3-4B/
```

The VPR paper PDF is available at:

```text
/mnt/project_rlinf/yuanhuining/repos/VPR-v2/vpr_arxiv_v2.pdf
```

Use it as reference for environment definitions and rewards. If PDF parsing is needed, install `pypdf` with `pip`.

## Core Design Constraint

Each agent step must be independent:

```text
obs_t -> model outputs action_t -> env.step(action_t) -> reward_t, obs_{t+1}
```

The model input for step `t` should contain only:

- task instruction;
- current observation/state;
- current legal/admissible action information;
- output format instruction.

Do **not** concatenate the full multi-turn interaction history into the model context.

Each step should become its own training/evaluation sequence. Loss should only apply to the current action response.

## Target Environments

Expose the following environment names in `verl-agent`:

```text
vpr_tictactoe
vpr_sudoku
vpr_minesweeper
```

## Shared Environment API

Follow the closest existing `verl-agent` environment convention. Each environment should support:

```python
reset(seed=None, options=None, **kwargs)
step(action: str)
```

Each `step` should return the normal `verl-agent`/gym-style fields and include this metadata in `info`:

```python
{
    "env_name": str,
    "step": int,
    "max_steps": int,
    "raw_action": str,
    "parsed_action": str | None,
    "parse_ok": bool,
    "illegal_action": bool,
    "available_actions": list[str],
    "vpr_reward": float,
    "terminal_success": bool | None,
    "terminal_reason": str | None,
}
```

Default dense reward convention:

```text
oracle-valid action: +1.0
legal but non-oracle action: 0.0
invalid / unparsable / illegal action: configurable penalty, default -1.0
```

Terminal success/failure should be logged in `info`; do not replace dense VPR reward unless explicitly configured.

## Action Format

All prompts should require exactly one final action:

```text
<think>optional brief reasoning</think>
<action>...</action>
```

Parser requirements:

- extract the last `<action>...</action>` block;
- tolerate harmless whitespace and casing differences;
- support simple aliases where useful;
- never crash on malformed model output;
- mark parse failures in `info`.

## Environment 1: Tic-Tac-Toe

Implement this environment directly.

### Observation

Text board with coordinates, current player, and legal cells.

Example action:

```text
<action>3</action>
```

### Action Space

Canonical action is a cell index:

```text
1..9
```

### Opponent

Support at least:

- `random`

Optionally support later:

- `minimax`
- `mcts`
- `mixed`

### Verifier / Reward

Use exact minimax if simpler than MCTS. Because Tic-Tac-Toe is small, exact minimax is acceptable and should be treated as the search oracle.

Reward:

- `1.0` if the action is in the optimal action set;
- `0.0` if legal but not optimal;
- invalid-action penalty otherwise.

### Termination

Terminate on win/loss/draw, invalid action if configured, or max steps.

Extra `info`:

```python
{
    "game_result": "win" | "loss" | "draw" | "ongoing",
    "oracle_valid_actions": list[str],
    "opponent_action": str | None,
}
```

## Environment 2: Sudoku

Use `gem` as the base environment.

### Required Work

1. Inspect `gem` Sudoku API.
2. Write a thin `verl-agent` wrapper.
3. Convert `gem` observations into concise text prompts.
4. Convert model actions into `gem` actions.
5. Add VPR reward using the Sudoku solution/verifier.

### Observation

Render the current 9x9 grid using `.` for blanks.

Canonical action:

```text
<action>row col digit</action>
```

Rows and columns are 1-indexed.

### Verifier / Reward

Given solution grid `G*`, action `(i, j, d)` is oracle-valid iff:

```text
cell (i, j) is currently blank and G*[i, j] == d
```

Reward:

- `1.0` if oracle-valid;
- `0.0` if parsed but wrong;
- invalid-action penalty for filled cell, out-of-range action, or parse failure.

Default strict behavior:

```text
terminate_on_wrong_digit = True
```

Extra `info`:

```python
{
    "num_blanks_remaining": int,
    "completion_rate": float,
}
```

## Environment 3: Minesweeper

Use `gem` as the base environment.

### Required Work

1. Inspect `gem` Minesweeper API.
2. Write a thin `verl-agent` wrapper.
3. Convert `gem` observations into concise text prompts.
4. Convert model actions into `gem` actions.
5. Add posterior-based VPR reward.

### Observation

Render the current board with hidden cells, revealed numbers, and flags.

Canonical actions:

```text
<action>reveal row col</action>
<action>flag row col</action>
```

Rows and columns are 1-indexed.

Aliases:

```text
open/click -> reveal
mark -> flag
```

### Verifier / Reward

Use posterior enumeration over mine configurations consistent with current revealed observations.

Oracle-valid actions:

- `reveal i j` if the cell has minimum posterior mine probability among unrevealed/unflagged cells;
- `flag i j` if posterior mine probability is exactly `1.0`.

Ties are oracle-valid.

Reward:

- `1.0` if oracle-valid;
- `0.0` if legal but non-oracle;
- invalid-action penalty otherwise.

Extra `info`:

```python
{
    "posterior_min_prob": float,
    "posterior_prob_for_action": float,
    "oracle_valid_actions": list[str],
    "completion_rate": float,
}
```

## Prompt Template

Use short bounded prompts. Do not include full history.

Generic template:

```text
You are solving an interactive reasoning task.

Current observation:
{observation}

Available action format:
{action_format}

Available actions:
{available_actions}

Choose exactly one action for the current step.
You may reason briefly in <think>...</think>.
Your final action must be enclosed in <action>...</action>.
```

For Sudoku, avoid listing all `(row, col, digit)` triples unless already compact. Listing blank cells is enough.

## Integration Tasks

1. Inspect current `verl-agent` environment and GRPO config patterns.
2. Inspect `gem` Minesweeper and Sudoku APIs.
3. Add wrappers for:
   - `vpr_sudoku`
   - `vpr_minesweeper`
4. Implement direct environment for:
   - `vpr_tictactoe`
5. Add shared action parser utilities.
6. Register all three environments in the `verl-agent` environment manager/registry.
7. Add prompt builders for all three environments.
8. Add standard GRPO smoke-test configs/scripts using:

```text
model path: /mnt/project_rlinf/yuanhuining/models/Qwen3-4B/
algorithm: standard GRPO
```

9. Add minimal tests for reset, step, parsing, reward, and termination.
10. Verify prompts stay bounded across long rollouts.

## Smoke Tests

Smoke tests should run actual `verl-agent` standard GRPO rollout/training for a very small budget, not just random-policy environment stepping.

Use:

```text
/mnt/project_rlinf/yuanhuining/models/Qwen3-4B/
```

Suggested tiny settings:

```text
num_train_steps: 1-3
rollout episodes: minimal
max_prompt_length: bounded
max_response_length: small
algorithm: standard GRPO
```

Add scripts such as:

```bash
bash examples/vpr_games/grpo_tictactoe_smoke.sh
bash examples/vpr_games/grpo_sudoku_smoke.sh
bash examples/vpr_games/grpo_minesweeper_smoke.sh
```

Each script should confirm:

- environment can be instantiated;
- model can generate an action;
- action can be parsed;
- env returns dense reward;
- GRPO step completes without crashing;
- logged prompt does not contain full interaction history.

## Tests

Add unit tests for:

- action tag parsing;
- malformed output handling;
- seeded reset determinism;
- legal/illegal action handling;
- reward correctness for simple known states;
- terminal success/failure;
- `info` fields.

For Minesweeper, include at least one small deterministic posterior test.

For Sudoku, include at least one puzzle with known solution.

For Tic-Tac-Toe, include at least one board where the optimal move is obvious and test that reward is `1.0`.

## Acceptance Criteria

The task is complete when:

1. `vpr_tictactoe`, `vpr_sudoku`, and `vpr_minesweeper` are registered in `verl-agent`.
2. Sudoku and Minesweeper use `gem` as their base environment.
3. Tic-Tac-Toe is implemented directly.
4. Each environment returns dense VPR rewards per step.
5. Each prompt uses only current observation/action information, not full history.
6. Standard GRPO smoke scripts run with local Qwen3-4B.
7. Invalid model outputs are handled gracefully.
8. Unit tests pass.
9. The implementation can reference the VPR PDF path above, with `pypdf` installed if parsing is needed.

## Implementation Order

1. Inspect `verl-agent` environment + GRPO examples.
2. Inspect `gem` Sudoku/Minesweeper APIs.
3. Build shared parser.
4. Implement and register Tic-Tac-Toe.
5. Wrap and register Sudoku from `gem`.
6. Wrap and register Minesweeper from `gem`.
7. Add prompts.
8. Add unit tests.
9. Add standard GRPO smoke scripts with local Qwen3-4B.
10. Run tests and smoke scripts.

--- Original Design Draft End ---
