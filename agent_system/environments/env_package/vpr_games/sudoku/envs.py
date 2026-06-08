"""Sudoku GEM adapter, Ray actor, parallel env, and builder for VPR."""

from __future__ import annotations

import re
import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.common.parser import parse_action_tag
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
                 max_turns: int = 100, invalid_penalty: float = -1.0,
                 terminate_on_wrong_digit: bool = True,
                 terminate_on_invalid_parse: bool = True,
                 max_generation_attempts: int = 256,
                 reward_mode: str = "oracle"):
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        from gem.envs.game_env.sudoku import SudokuEnv
        self._env = SudokuEnv(n=n, clues=clues, max_turns=max_turns)
        self._seed = seed
        self._reward_mode = reward_mode
        # GEM interprets `clues` as the target number of blank cells to remove,
        # but it abandons a removal when it would break the unique-solution
        # guarantee, so a raw reset can yield fewer blanks than requested. VPR
        # requires the fixed paper-default board, so we treat `clues` as the
        # exact required blank count and retry generation until it is met.
        self._target_blanks = clues
        self._max_generation_attempts = max_generation_attempts
        self._invalid_penalty = invalid_penalty
        self._terminate_on_wrong_digit = terminate_on_wrong_digit
        self._terminate_on_invalid_parse = terminate_on_invalid_parse
        self._step_count = 0
        self._max_steps = max_turns
        self._done = False

    def _count_blanks(self) -> int:
        return sum(cell == 0 for row in self._env.board for cell in row)

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
            "num_blanks_remaining": blanks,
            "completion_rate": self._completion_rate(blanks),
            "move_optimal": None,
        }
        return obs_text, info

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
        result = parse_action_tag(raw_text)
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
            vpr_reward = self._invalid_penalty
            blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
            self._done = True
            info = self._build_info(raw_text, result.action_text, True, True,
                                    vpr_reward, False, "out_of_range", blanks)
            return _render_sudoku(self._env.board), vpr_reward, True, info

        # Check if cell is blank
        if self._env.board[row - 1][col - 1] != 0:
            vpr_reward = self._invalid_penalty
            blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
            self._done = True
            info = self._build_info(raw_text, result.action_text, True, True,
                                    vpr_reward, False, "cell_not_blank", blanks)
            return _render_sudoku(self._env.board), vpr_reward, True, info

        # Oracle reward: correct digit?
        is_oracle = (self._env.full_grid[row - 1][col - 1] == digit)
        vpr_reward = 1.0 if is_oracle else 0.0

        # Apply state update via GEM
        gem_action = f"\\boxed{{{row} {col} {digit}}}"
        _, _, gem_terminated, gem_truncated, _ = self._env.step(gem_action)

        # VPR termination logic
        done = gem_terminated or gem_truncated or self._step_count >= self._max_steps
        if not is_oracle and self._terminate_on_wrong_digit:
            done = True
            vpr_reward = self._invalid_penalty

        self._done = done
        blanks = sum(cell == 0 for row_ in self._env.board for cell in row_)
        is_complete = (blanks == 0)
        terminal_success = is_complete if done else None
        terminal_reason = "complete" if is_complete else ("wrong_digit" if not is_oracle else None) if done else None

        info = self._build_info(raw_text, result.action_text, True, False,
                                vpr_reward, terminal_success, terminal_reason, blanks,
                                move_optimal=is_oracle)
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
        n = len(self._env.board)
        total_blanks = self._env.init_num_empty if hasattr(self._env, 'init_num_empty') else 40
        if total_blanks == 0:
            return 1.0
        filled = total_blanks - blanks
        return filled / total_blanks

    def _build_info(self, raw, parsed_action, parse_ok, illegal, vpr_reward,
                    terminal_success, terminal_reason, blanks, move_optimal=None):
        total_blanks = self._env.init_num_empty if hasattr(self._env, 'init_num_empty') else 40
        filled = max(0, total_blanks - blanks)
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
            "num_blanks_remaining": blanks,
            "completion_rate": filled / total_blanks if total_blanks > 0 else 1.0,
            # Whether the filled digit matched the unique solution (set only on legal
            # digit placements; None on illegal / parse-failure / already-done steps).
            "move_optimal": move_optimal,
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
    invalid_penalty = getattr(env_config, "invalid_penalty", -1.0)
    terminate_wrong = getattr(cfg, "terminate_on_wrong_digit", True) if cfg else True
    terminate_invalid = getattr(cfg, "terminate_on_invalid_parse", True) if cfg else True
    max_gen_attempts = getattr(cfg, "max_generation_attempts", 256) if cfg else 256
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"

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
            reward_mode=reward_mode,
        ))
        seeds.append(actor_seed)
    return SudokuMultiProcessEnv(workers=workers, seeds=seeds)
