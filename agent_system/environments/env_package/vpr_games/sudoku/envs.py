"""Sudoku GEM adapter, Ray actor, parallel env, and builder for VPR."""

from __future__ import annotations

import random
import re
import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.common.parser import (
    normalize_action_format,
    parse_action,
)
from agent_system.environments.env_package.vpr_games.common.rewards import outcome_reward

_ACTION_RE = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)$")


def _parse_sudoku_action(action_text: str):
    """Parse 'row col digit' from action_text. Returns (row, col, digit) 1-indexed or None."""
    if action_text is None:
        return None
    m = _ACTION_RE.match(action_text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _render_sudoku(board, n=9) -> str:
    lines = []
    header = "   " + " ".join(f"C{c+1}" for c in range(n))
    lines.append(header)
    for r in range(n):
        cells = []
        for c in range(n):
            cells.append(str(board[r][c]) if board[r][c] != 0 else ".")
        row_str = f"R{r+1} " + "  ".join(cells)
        lines.append(row_str)
        if (r + 1) % 3 == 0 and r < n - 1:
            lines.append("   " + "-" * (n * 3))
    return "\n".join(lines)


@ray.remote
class SudokuWorker:
    """Ray remote actor holding one GEM Sudoku instance."""

    def __init__(self, seed: int = 0, n: int = 3, clues: int = 40,
                 max_turns: int = 100, invalid_penalty: float = -2.0,
                 terminate_on_wrong_digit: bool = True,
                 terminate_on_invalid_parse: bool = True,
                 max_generation_attempts: int = 256,
                 forced_reward: float = 2.0, mrv_reward: float = 1.0,
                 legal_non_oracle_reward: float = 0.5,
                 wrong_digit_penalty: float = -1.0,
                 cell_error_penalty: float = -2.0,
                 reward_mode: str = "oracle", action_format: str = "action_tag",
                 outcome_success_correct_fills: int | None = None):
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        if outcome_success_correct_fills is not None:
            if (
                isinstance(outcome_success_correct_fills, bool)
                or not isinstance(outcome_success_correct_fills, int)
                or outcome_success_correct_fills <= 0
            ):
                raise ValueError("outcome_success_correct_fills must be a positive integer or null")
            if reward_mode != "outcome":
                raise ValueError("outcome_success_correct_fills requires reward_mode='outcome'")
            if outcome_success_correct_fills > clues:
                raise ValueError("outcome_success_correct_fills cannot exceed the initial blank count")
            if outcome_success_correct_fills > max_turns:
                raise ValueError("outcome_success_correct_fills cannot exceed max_turns")
        from gem.envs.game_env.sudoku import SudokuEnv
        self._env = SudokuEnv(n=n, clues=clues, max_turns=max_turns)
        self._seed = seed
        self._reward_mode = reward_mode
        self._action_format = normalize_action_format(action_format)
        self._outcome_success_correct_fills = outcome_success_correct_fills
        # GEM interprets `clues` as the target number of blank cells to remove,
        # but it abandons a removal when it would break the unique-solution
        # guarantee, so a raw reset can yield fewer blanks than requested. VPR
        # requires the fixed paper-default board, so we treat `clues` as the
        # exact required blank count and retry generation until it is met.
        self._target_blanks = clues
        self._max_generation_attempts = max_generation_attempts
        self._invalid_penalty = float(invalid_penalty)
        self._forced_reward = float(forced_reward)
        self._mrv_reward = float(mrv_reward)
        self._legal_non_oracle_reward = float(legal_non_oracle_reward)
        self._wrong_digit_penalty = float(wrong_digit_penalty)
        self._cell_error_penalty = float(cell_error_penalty)
        self._terminate_on_wrong_digit = terminate_on_wrong_digit
        self._terminate_on_invalid_parse = terminate_on_invalid_parse
        self._step_count = 0
        self._max_steps = max_turns
        self._done = False
        self._correct_fills = 0

    def _count_blanks(self) -> int:
        return sum(cell == 0 for row in self._env.board for cell in row)

    def _initial_blank_count(self) -> int:
        return int(self._env.init_num_empty) if hasattr(self._env, 'init_num_empty') else self._target_blanks

    def _candidate_digits(self, row: int, col: int):
        """Return legal Sudoku digits for a blank cell, using the current board only."""
        board = self._env.board
        if board[row][col] != 0:
            return []
        n = len(board)
        box = int(n ** 0.5)
        used = set(board[row])
        used.update(board[r][col] for r in range(n))
        box_r = (row // box) * box
        box_c = (col // box) * box
        for rr in range(box_r, box_r + box):
            used.update(board[rr][box_c:box_c + box])
        used.discard(0)
        return [d for d in range(1, n + 1) if d not in used]

    def _mrv_oracle_state(self):
        """Pre-execution oracle labels: forced cells first, MRV cells otherwise."""
        candidate_counts = {}
        mrv_cells = []
        min_candidates = None
        n = len(self._env.board)
        for r in range(n):
            for c in range(n):
                if self._env.board[r][c] != 0:
                    continue
                candidates = self._candidate_digits(r, c)
                count = len(candidates)
                candidate_counts[(r, c)] = count
                if min_candidates is None or count < min_candidates:
                    min_candidates = count
                    mrv_cells = [(r, c)]
                elif count == min_candidates:
                    mrv_cells.append((r, c))

        forced_cells = [cell for cell, count in candidate_counts.items() if count == 1]
        oracle_cells = forced_cells if forced_cells else mrv_cells
        return {
            "candidate_counts": candidate_counts,
            "min_candidates": min_candidates,
            "forced_cells": set(forced_cells),
            "mrv_cells": set(mrv_cells),
            "oracle_cells": set(oracle_cells),
            "forced_available": bool(forced_cells),
        }

    def _generate_board(self, base_seed: int):
        """Reset the GEM env to a board with exactly `self._target_blanks` blanks.

        Generation is a pure deterministic function of `base_seed`: attempt 0 uses
        the base seed and each retry derives a distinct seed from it, so the same
        base seed (and therefore grouped replicas sharing a seed) always resolves
        to the identical board. Raises ValueError if no qualifying board is found
        within the attempt budget.
        """
        for attempt in range(self._max_generation_attempts):
            trial_seed = base_seed if attempt == 0 else base_seed + attempt * 1_000_003
            self._env.reset(seed=trial_seed)
            if self._count_blanks() == self._target_blanks:
                return
        raise ValueError(
            f"SudokuWorker could not generate a board with exactly "
            f"{self._target_blanks} blanks from base seed {base_seed} within "
            f"{self._max_generation_attempts} attempts."
        )

    def reset(self, seed=None):
        s = seed if seed is not None else self._seed
        self._generate_board(s)
        self._step_count = 0
        self._done = False
        self._correct_fills = 0
        obs_text = _render_sudoku(self._env.board)
        blanks = self._count_blanks()
        info = {
            "env_name": "vpr_sudoku",
            "step": 0,
            "max_steps": self._max_steps,
            "raw_action": "",
            "parsed_action": None,
            "parse_ok": True,
            "illegal_action": False,
            "available_actions": self._blank_cells(),
            "vpr_reward": 0.0,
            "terminal_success": None,
            "terminal_reason": None,
            "initial_blank_count": self._initial_blank_count(),
            "num_blanks_remaining": blanks,
            "completion_rate": self._completion_rate(blanks),
            "correct_fills": 0,
            "outcome_success_correct_fills": self._outcome_success_correct_fills,
            "outcome_target_completion_rate": 0.0,
            "move_optimal": None,
            "pre_exec_oracle_match": None,
            "legal_non_oracle": False,
            "sudoku_mrv_min_candidates": None,
            "sudoku_candidate_count_for_action": None,
            "sudoku_forced_cell_available": False,
            "sudoku_action_is_mrv_cell": False,
            "sudoku_oracle_tier": None,
            "oracle_action_set_size": 0,
        }
        return obs_text, info

    def _snapshot_state(self):
        return {
            "board": [row[:] for row in self._env.board],
            "full_grid": [row[:] for row in self._env.full_grid],
            "turn_count": int(self._env.turn_count),
            "step_count": int(self._step_count),
            "done": bool(self._done),
            "correct_fills": int(self._correct_fills),
        }

    def _restore_state(self, state):
        self._env.board = [row[:] for row in state["board"]]
        self._env.full_grid = [row[:] for row in state["full_grid"]]
        self._env.turn_count = int(state["turn_count"])
        self._step_count = int(state["step_count"])
        self._done = bool(state["done"])
        self._correct_fills = int(state.get("correct_fills", 0))

    def current_observation_info(self):
        obs_text = _render_sudoku(self._env.board)
        blanks = self._count_blanks()
        target_reached = (
            self._outcome_success_correct_fills is not None
            and self._correct_fills >= self._outcome_success_correct_fills
        )
        terminal_success = (
            True if self._done and (blanks == 0 or target_reached) else None
        )
        terminal_reason = (
            "outcome_target"
            if target_reached
            else ("complete" if blanks == 0 else None)
        )
        info = self._build_info(
            raw="", parsed_action=None, parse_ok=True, illegal=False, vpr_reward=0.0,
            terminal_success=terminal_success,
            terminal_reason=terminal_reason if self._done else None,
            blanks=blanks, move_optimal=None,
        )
        info["observation"] = obs_text
        return obs_text, info

    def snapshot_state(self):
        return self._snapshot_state()

    def restore_state(self, state):
        self._restore_state(state)
        return self.current_observation_info()

    def step_candidate_group(self, raw_texts, selection_mode="best", random_select_prob=0.0):
        """Evaluate candidates from one state, then commit a selected candidate."""
        snapshot = self._snapshot_state()
        candidates = []
        best_idx = 0
        best_reward = None
        for idx, raw_text in enumerate(raw_texts):
            self._restore_state(snapshot)
            obs, reward, done, info = self.step(raw_text)
            info["observation"] = obs
            info["candidate_index"] = idx
            info["env_done"] = bool(done)
            candidates.append((obs, float(reward), bool(done), info))
            if best_reward is None or float(reward) > best_reward:
                best_reward = float(reward)
                best_idx = idx

        selected_idx = best_idx
        selection_type = "best"
        if raw_texts and selection_mode == "mixed" and random.random() < float(random_select_prob):
            selected_idx = random.randrange(len(raw_texts))
            selection_type = "random"
        elif selection_mode == "random" and raw_texts:
            selected_idx = random.randrange(len(raw_texts))
            selection_type = "random"

        self._restore_state(snapshot)
        selected_obs, selected_reward, selected_done, selected_info = self.step(raw_texts[selected_idx])
        selected_info["observation"] = selected_obs
        selected_info["candidate_index"] = selected_idx
        selected_info["best_candidate_index"] = best_idx
        selected_info["state_group_selected"] = True
        selected_info["state_group_selection_type"] = selection_type
        selected_info["state_group_random_selected"] = selection_type == "random"
        selected_info["env_done"] = bool(selected_done)
        return candidates, selected_idx, selected_obs, float(selected_reward), bool(selected_done), selected_info

    def step(self, raw_text: str):
        obs, reward, done, info = self._step_impl(raw_text)
        if self._reward_mode == "outcome":
            reward = outcome_reward(done, info.get("terminal_success"),
                                    info.get("terminal_reason"))
            info["vpr_reward"] = reward
        return obs, reward, done, info

    def _step_impl(self, raw_text: str):
        if self._done:
            return _render_sudoku(self._env.board), 0.0, True, self._terminal_info(raw_text)

        self._step_count += 1
        result = parse_action(raw_text, self._action_format)
        parsed = _parse_sudoku_action(result.action_text) if result.parse_ok else None

        if parsed is None:
            vpr_reward = self._invalid_penalty
            terminate = self._terminate_on_invalid_parse
            if terminate:
                self._done = True
            blanks = self._count_blanks()
            # Terminal fields are only meaningful when the episode actually ends;
            # a non-terminating invalid parse leaves them unset (None).
            term_success = False if terminate else None
            term_reason = "invalid_action" if terminate else None
            info = self._build_info(raw_text, result.action_text, result.parse_ok, True,
                                    vpr_reward, term_success, term_reason, blanks)
            return _render_sudoku(self._env.board), vpr_reward, terminate, info

        row, col, digit = parsed
        # Validate range
        n = len(self._env.board)
        if not (1 <= row <= n and 1 <= col <= n and 1 <= digit <= 9):
            vpr_reward = self._cell_error_penalty
            blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
            self._done = True
            info = self._build_info(raw_text, result.action_text, True, True,
                                    vpr_reward, False, "out_of_range", blanks)
            return _render_sudoku(self._env.board), vpr_reward, True, info

        # Check if cell is blank
        if self._env.board[row - 1][col - 1] != 0:
            vpr_reward = self._cell_error_penalty
            blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
            self._done = True
            info = self._build_info(raw_text, result.action_text, True, True,
                                    vpr_reward, False, "cell_not_blank", blanks)
            return _render_sudoku(self._env.board), vpr_reward, True, info

        oracle_state = self._mrv_oracle_state()
        action_cell = (row - 1, col - 1)
        is_solution_digit = (self._env.full_grid[row - 1][col - 1] == digit)
        is_mrv_cell = action_cell in oracle_state["mrv_cells"]
        forced_available = oracle_state["forced_available"]
        is_forced_cell = action_cell in oracle_state["forced_cells"]
        candidate_count = oracle_state["candidate_counts"].get(action_cell)

        is_oracle = bool(is_mrv_cell and is_solution_digit)
        oracle_tier = "forced" if (is_oracle and is_forced_cell) else ("mrv" if is_oracle else None)
        if is_oracle:
            vpr_reward = self._forced_reward if is_forced_cell else self._mrv_reward
        elif is_solution_digit:
            vpr_reward = self._legal_non_oracle_reward
        else:
            vpr_reward = self._wrong_digit_penalty

        # Apply state update via GEM
        gem_action = f"\\boxed{{{row} {col} {digit}}}"
        _, _, gem_terminated, gem_truncated, _ = self._env.step(gem_action)
        action_applied = self._env.board[row - 1][col - 1] == digit
        if is_solution_digit and action_applied:
            self._correct_fills += 1

        # VPR termination logic
        outcome_target_reached = (
            self._outcome_success_correct_fills is not None
            and self._correct_fills >= self._outcome_success_correct_fills
        )
        done = (
            gem_terminated
            or gem_truncated
            or outcome_target_reached
            or self._step_count >= self._max_steps
        )
        wrong_digit_terminal = ((not is_solution_digit) and self._terminate_on_wrong_digit)
        if wrong_digit_terminal:
            done = True
            vpr_reward = self._wrong_digit_penalty

        self._done = done
        blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
        is_complete = (blanks == 0)
        terminal_success = (is_complete or outcome_target_reached) if done else None
        terminal_reason = None
        if is_complete:
            terminal_reason = "complete"
        elif outcome_target_reached:
            terminal_reason = "outcome_target"
        elif done:
            terminal_reason = "wrong_digit" if wrong_digit_terminal else "timeout"
            if terminal_reason == "timeout":
                vpr_reward = self._invalid_penalty

        info = self._build_info(raw_text, result.action_text, True, False,
                                vpr_reward, terminal_success, terminal_reason, blanks,
                                move_optimal=is_oracle,
                                sudoku_mrv_min_candidates=oracle_state["min_candidates"],
                                sudoku_candidate_count_for_action=candidate_count,
                                sudoku_forced_cell_available=forced_available,
                                sudoku_action_is_mrv_cell=is_mrv_cell,
                                sudoku_oracle_tier=oracle_tier,
                                oracle_action_set_size=len(oracle_state["oracle_cells"]))
        return _render_sudoku(self._env.board), vpr_reward, done, info

    def _blank_cells(self):
        cells = []
        n = len(self._env.board)
        for r in range(n):
            for c in range(n):
                if self._env.board[r][c] == 0:
                    cells.append(f"{r+1} {c+1}")
        return cells

    def _completion_rate(self, blanks):
        total_blanks = self._initial_blank_count()
        if total_blanks == 0:
            return 1.0
        filled = total_blanks - blanks
        return filled / total_blanks

    def _build_info(self, raw, parsed_action, parse_ok, illegal, vpr_reward,
                    terminal_success, terminal_reason, blanks, move_optimal=None,
                    sudoku_mrv_min_candidates=None,
                    sudoku_candidate_count_for_action=None,
                    sudoku_forced_cell_available=False,
                    sudoku_action_is_mrv_cell=False,
                    sudoku_oracle_tier=None,
                    oracle_action_set_size=0):
        total_blanks = self._initial_blank_count()
        filled = max(0, total_blanks - blanks)
        target = self._outcome_success_correct_fills
        target_completion = (
            min(self._correct_fills / target, 1.0) if target is not None else 0.0
        )
        return {
            "env_name": "vpr_sudoku",
            "step": self._step_count,
            "max_steps": self._max_steps,
            "raw_action": raw,
            "parsed_action": parsed_action,
            "parse_ok": parse_ok,
            "illegal_action": illegal,
            "available_actions": self._blank_cells(),
            "vpr_reward": vpr_reward,
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
            "initial_blank_count": total_blanks,
            "num_blanks_remaining": blanks,
            "completion_rate": filled / total_blanks if total_blanks > 0 else 1.0,
            "correct_fills": self._correct_fills,
            "outcome_success_correct_fills": target,
            "outcome_target_completion_rate": target_completion,
            # Whether the action matched the pre-execution MRV/forced-cell oracle
            # (set only on legal digit placements; None on illegal / parse-failure / already-done steps).
            "move_optimal": move_optimal,
            "pre_exec_oracle_match": move_optimal,
            "legal_non_oracle": bool(move_optimal is False and parse_ok and not illegal),
            "sudoku_mrv_min_candidates": sudoku_mrv_min_candidates,
            "sudoku_candidate_count_for_action": sudoku_candidate_count_for_action,
            "sudoku_forced_cell_available": sudoku_forced_cell_available,
            "sudoku_action_is_mrv_cell": sudoku_action_is_mrv_cell,
            "sudoku_oracle_tier": sudoku_oracle_tier,
            "oracle_action_set_size": oracle_action_set_size,
        }

    def _terminal_info(self, raw):
        blanks = sum(cell == 0 for row in self._env.board for cell in row)
        return self._build_info(raw, None, True, False, 0.0, None, "already_done", blanks)

    def close(self):
        pass


class SudokuMultiProcessEnv:
    """Vectorized Sudoku using Ray actors."""

    def __init__(self, workers: list, seeds: list):
        self.workers = workers
        self.seeds = seeds
        # Episode counter: advanced once per reset() so each rollout (i.e. each training
        # step) draws a *fresh* board instead of replaying the same fixed per-slot seed
        # every step. Group replicas keep an identical seed within a step (same base seed
        # + same counter), so GRPO groups stay comparable; the run is still fully
        # reproducible from `env.seed`.
        self._episode = 0

    def reset(self):
        offset = self._episode * 100003  # large prime stride → distinct, non-colliding seeds
        futures = [w.reset.remote(seed=s + offset) for w, s in zip(self.workers, self.seeds)]
        self._episode += 1
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return obs_list, info_list

    def step(self, actions):
        futures = [w.step.remote(act) for w, act in zip(self.workers, actions)]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        info_list = [r[3] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return obs_list, rewards, dones, info_list

    def step_candidate_groups(self, candidate_action_groups, active_indices=None, selection_mode="best", random_select_prob=0.0):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        worker_indices = [int(i) for i in active_indices]
        if len(worker_indices) != len(candidate_action_groups):
            raise ValueError(
                f"active_indices length {len(worker_indices)} does not match "
                f"candidate groups {len(candidate_action_groups)}"
            )
        futures = []
        for group_idx, (worker_idx, actions) in enumerate(zip(worker_indices, candidate_action_groups)):
            futures.append(
                self.workers[worker_idx].step_candidate_group.remote(
                    actions,
                    selection_mode=selection_mode,
                    random_select_prob=random_select_prob
                )
            )
        results = ray.get(futures)
        candidate_results = [r[0] for r in results]
        selected_indices = np.array([r[1] for r in results], dtype=np.int32)
        obs_list = [r[2] for r in results]
        rewards = np.array([r[3] for r in results], dtype=np.float32)
        dones = np.array([r[4] for r in results], dtype=bool)
        info_list = [r[5] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return candidate_results, selected_indices, obs_list, rewards, dones, info_list

    def snapshot_states(self, active_indices=None):
        if active_indices is None:
            active_indices = range(len(self.workers))
        return ray.get([self.workers[int(i)].snapshot_state.remote() for i in active_indices])

    def restore_states(self, snapshots):
        if len(snapshots) > len(self.workers):
            raise ValueError(f"cannot restore {len(snapshots)} snapshots into {len(self.workers)} workers")
        futures = [self.workers[i].restore_state.remote(state) for i, state in enumerate(snapshots)]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return obs_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)


def build_sudoku_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                       is_train: bool = True, env_config=None) -> SudokuMultiProcessEnv:
    total = env_num * group_n
    cfg = getattr(env_config, "sudoku", None)
    n = getattr(cfg, "n", 3)
    clues = getattr(cfg, "clues", 40)
    max_turns = getattr(env_config, "max_steps", 100)
    invalid_penalty = getattr(env_config, "invalid_penalty", -2.0)
    terminate_wrong = getattr(cfg, "terminate_on_wrong_digit", True) if cfg else True
    terminate_invalid = getattr(cfg, "terminate_on_invalid_parse", True) if cfg else True
    max_gen_attempts = getattr(cfg, "max_generation_attempts", 256) if cfg else 256
    forced_reward = getattr(cfg, "forced_reward", 2.0) if cfg else 2.0
    mrv_reward = getattr(cfg, "mrv_reward", 1.0) if cfg else 1.0
    legal_non_oracle_reward = getattr(cfg, "legal_non_oracle_reward", 0.5) if cfg else 0.5
    wrong_digit_penalty = getattr(cfg, "wrong_digit_penalty", -1.0) if cfg else -1.0
    cell_error_penalty = getattr(cfg, "cell_error_penalty", -2.0) if cfg else -2.0
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"
    outcome_success_correct_fills = (
        getattr(cfg, "outcome_success_correct_fills", None) if cfg else None
    )
    action_format = getattr(env_config, "game_action_format", "action_tag")

    resources = getattr(env_config, "resources_per_worker", None)
    worker_kwargs = {}
    if resources is not None:
        from omegaconf import OmegaConf
        worker_kwargs = OmegaConf.to_container(resources, resolve=True)

    RemoteWorker = SudokuWorker.options(**worker_kwargs) if worker_kwargs else SudokuWorker
    workers, seeds = [], []
    for idx in range(total):
        episode_idx = idx // group_n
        actor_seed = seed + episode_idx
        workers.append(RemoteWorker.remote(
            seed=actor_seed, n=n, clues=clues, max_turns=max_turns,
            invalid_penalty=invalid_penalty, terminate_on_wrong_digit=terminate_wrong,
            terminate_on_invalid_parse=terminate_invalid,
            max_generation_attempts=max_gen_attempts,
            forced_reward=forced_reward, mrv_reward=mrv_reward,
            legal_non_oracle_reward=legal_non_oracle_reward,
            wrong_digit_penalty=wrong_digit_penalty,
            cell_error_penalty=cell_error_penalty,
            reward_mode=reward_mode,
            action_format=action_format,
            outcome_success_correct_fills=outcome_success_correct_fills,
        ))
        seeds.append(actor_seed)
    return SudokuMultiProcessEnv(workers=workers, seeds=seeds)
