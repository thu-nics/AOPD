"""Minesweeper GEM adapter, Ray actor, parallel env, and builder for VPR."""

from __future__ import annotations

import re
import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.common.parser import parse_action_tag
from agent_system.environments.env_package.vpr_games.common.rewards import outcome_reward
from agent_system.environments.env_package.vpr_games.minesweeper.oracle import (
    compute_posteriors, get_oracle_actions
)

_REVEAL_RE = re.compile(r"^(reveal|flag)\s+(\d+)\s+(\d+)$", re.IGNORECASE)


def _parse_ms_action(action_text: str):
    """Parse 'reveal|flag row col' (1-indexed). Returns (action_type, row, col) or None."""
    if action_text is None:
        return None
    m = _REVEAL_RE.match(action_text.strip())
    if not m:
        return None
    atype = m.group(1).lower()
    return atype, int(m.group(2)), int(m.group(3))


def _render_board(revealed, grid, flags, rows, cols) -> str:
    lines = []
    header = "   " + " ".join(f"C{c+1}" for c in range(cols))
    lines.append(header)
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            if flags[r][c]:
                row_cells.append("F")
            elif revealed[r][c]:
                v = grid[r][c]
                row_cells.append(str(v) if v >= 0 else "M")
            else:
                row_cells.append(".")
        lines.append(f"R{r+1} " + " ".join(row_cells))
    return "\n".join(lines)


def _board_info(revealed, flags, rows, cols):
    unrevealed = [
        f"{r+1} {c+1}" for r in range(rows) for c in range(cols)
        if not revealed[r][c] and not flags[r][c]
    ]
    flagged = [
        f"{r+1} {c+1}" for r in range(rows) for c in range(cols)
        if flags[r][c]
    ]
    return unrevealed, flagged


@ray.remote
class MinesweeperWorker:
    """Ray remote actor holding one GEM Minesweeper instance."""

    def __init__(self, seed: int = 0, rows: int = 5, cols: int = 5, num_mines: int = 5,
                 max_turns: int = 25, invalid_penalty: float = -1.0,
                 reward_mode: str = "oracle"):
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        from gem.envs.game_env.minesweeper import MinesweeperEnv
        self._env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines, max_turns=max_turns)
        self._seed = seed
        self._reward_mode = reward_mode
        self._rows = rows
        self._cols = cols
        self._num_mines = num_mines
        self._max_steps = max_turns
        self._invalid_penalty = invalid_penalty
        self._step_count = 0
        self._done = False
        self._first_revealed = False

    def reset(self, seed=None):
        s = seed if seed is not None else self._seed
        self._env.reset(seed=s)
        self._step_count = 0
        self._done = False
        self._first_revealed = False
        obs_text = _render_board(self._env.revealed, self._env.grid,
                                 self._env.flags, self._rows, self._cols)
        unrevealed, flagged = _board_info(self._env.revealed, self._env.flags, self._rows, self._cols)
        info = {
            "env_name": "vpr_minesweeper",
            "step": 0,
            "max_steps": self._max_steps,
            "raw_action": "",
            "parsed_action": None,
            "parse_ok": True,
            "illegal_action": False,
            "available_actions": unrevealed,
            "vpr_reward": 0.0,
            "terminal_success": None,
            "terminal_reason": None,
            "posterior_min_prob": None,
            "posterior_prob_for_action": None,
            "oracle_valid_actions": [],
            "completion_rate": 0.0,
            "oracle_degraded": False,
            "flagged_cells": flagged,
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
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            return obs_text, 0.0, True, self._build_info(raw_text, None, True, False, 0.0, None, "already_done", None, None, [])

        self._step_count += 1
        result = parse_action_tag(raw_text)
        parsed = _parse_ms_action(result.action_text) if result.parse_ok else None

        # Invalid parse
        if parsed is None:
            self._done = True
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            info = self._build_info(raw_text, result.action_text, result.parse_ok, True,
                                    self._invalid_penalty, False, "invalid_action", None, None, [])
            return obs_text, self._invalid_penalty, True, info

        action_type, row, col = parsed  # 1-indexed
        r0, c0 = row - 1, col - 1   # 0-indexed for GEM

        # Bounds check
        if not (0 <= r0 < self._rows and 0 <= c0 < self._cols):
            self._done = True
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            info = self._build_info(raw_text, result.action_text, True, True,
                                    self._invalid_penalty, False, "out_of_bounds", None, None, [])
            return obs_text, self._invalid_penalty, True, info

        # Revealed cell = invalid (can't act on revealed cells)
        if self._env.revealed[r0][c0]:
            self._done = True
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            info = self._build_info(raw_text, result.action_text, True, True,
                                    self._invalid_penalty, False, "cell_already_revealed", None, None, [])
            return obs_text, self._invalid_penalty, True, info

        # Trying to reveal a flagged cell = invalid (sentinel before GEM)
        if action_type == "reveal" and self._env.flags[r0][c0]:
            self._done = True
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            info = self._build_info(raw_text, result.action_text, True, True,
                                    self._invalid_penalty, False, "cannot_reveal_flagged_cell", None, None, [])
            return obs_text, self._invalid_penalty, True, info

        # Compute oracle BEFORE executing action (state is current)
        posteriors, oracle_degraded = {}, False
        oracle_actions = []
        post_prob = None
        min_prob = None
        if self._first_revealed and action_type == "reveal":
            posteriors, oracle_degraded = compute_posteriors(
                self._env.revealed, self._env.grid,
                self._rows, self._cols, self._num_mines
            )
            oracle_actions, min_prob, _ = get_oracle_actions(
                posteriors, self._env.revealed, self._env.flags, self._rows, self._cols
            )
            post_prob = posteriors.get((r0, c0), None)
        elif self._first_revealed and action_type == "flag":
            posteriors, oracle_degraded = compute_posteriors(
                self._env.revealed, self._env.grid,
                self._rows, self._cols, self._num_mines
            )
            oracle_actions, min_prob, _ = get_oracle_actions(
                posteriors, self._env.revealed, self._env.flags, self._rows, self._cols
            )
            post_prob = posteriors.get((r0, c0), None)

        # Compute VPR reward
        action_str = f"{action_type} {row} {col}"
        if self._first_revealed:
            vpr_reward = 1.0 if action_str in oracle_actions else 0.0
        else:
            # Before first reveal: any reveal action is safe (GEM first-click safety)
            vpr_reward = 1.0 if action_type == "reveal" else 0.0

        # Execute action via GEM
        if action_type == "flag":
            # Flag toggles
            if self._env.flags[r0][c0]:
                # Un-flag: legal non-oracle
                self._env.flags[r0][c0] = False
                gem_terminated = False
                gem_truncated = False
                vpr_reward = 0.0  # legal non-oracle
            else:
                gem_action = f"\\boxed{{flag {r0} {c0}}}"
                _, _, gem_terminated, gem_truncated, _ = self._env.step(gem_action)
        else:  # reveal
            gem_action = f"\\boxed{{reveal {r0} {c0}}}"
            _, gem_rew, gem_terminated, gem_truncated, _ = self._env.step(gem_action)
            if not self._first_revealed:
                self._first_revealed = True
            # Detect mine hit from the grid value (GEM fail_reward is 0.0, not negative)
            if gem_terminated and self._env.grid[r0][c0] < 0:
                # Mine reveals are legal non-oracle actions (reward=0.0), not invalid actions
                vpr_reward = 0.0
                self._done = True
                obs_text = _render_board(self._env.revealed, self._env.grid,
                                         self._env.flags, self._rows, self._cols)
                info = self._build_info(raw_text, result.action_text, True, False,
                                        vpr_reward, False, "mine_hit", min_prob, post_prob, oracle_actions)
                info["oracle_degraded"] = oracle_degraded
                return obs_text, vpr_reward, True, info

        # Reveal-only completion check (override GEM's flag-all requirement)
        safe_cells_revealed = all(
            self._env.revealed[r][c]
            for r in range(self._rows) for c in range(self._cols)
            if self._env.grid[r][c] != -1  # not a mine
        ) if self._first_revealed else False

        done = safe_cells_revealed or gem_truncated or self._step_count >= self._max_steps
        self._done = done

        terminal_success = safe_cells_revealed if done else None
        terminal_reason = "complete" if safe_cells_revealed else ("timeout" if done else None)

        # Completion rate: fraction of safe cells revealed
        if self._first_revealed:
            total_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                             if self._env.grid[r][c] != -1)
            revealed_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                                if self._env.grid[r][c] != -1 and self._env.revealed[r][c])
            completion_rate = revealed_safe / total_safe if total_safe > 0 else 0.0
        else:
            completion_rate = 0.0

        obs_text = _render_board(self._env.revealed, self._env.grid,
                                 self._env.flags, self._rows, self._cols)
        info = self._build_info(raw_text, result.action_text, True, False,
                                vpr_reward, terminal_success, terminal_reason,
                                min_prob, post_prob, oracle_actions)
        info["oracle_degraded"] = oracle_degraded
        info["completion_rate"] = completion_rate
        return obs_text, vpr_reward, done, info

    def _build_info(self, raw, parsed_action, parse_ok, illegal, vpr_reward,
                    terminal_success, terminal_reason, min_prob, post_prob, oracle_actions):
        unrevealed, flagged = _board_info(self._env.revealed, self._env.flags, self._rows, self._cols)
        total_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                         if self._first_revealed and self._env.grid[r][c] != -1)
        revealed_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                            if self._first_revealed and self._env.grid[r][c] != -1
                            and self._env.revealed[r][c])
        completion = revealed_safe / total_safe if total_safe > 0 else 0.0
        return {
            "env_name": "vpr_minesweeper",
            "step": self._step_count,
            "max_steps": self._max_steps,
            "raw_action": raw,
            "parsed_action": parsed_action,
            "parse_ok": parse_ok,
            "illegal_action": illegal,
            "available_actions": unrevealed,
            "vpr_reward": vpr_reward,
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
            "posterior_min_prob": float(min_prob) if min_prob is not None else None,
            "posterior_prob_for_action": float(post_prob) if post_prob is not None else None,
            "oracle_valid_actions": oracle_actions,
            "completion_rate": completion,
            "oracle_degraded": False,
            "flagged_cells": flagged,
        }

    def close(self):
        pass


class MinesweeperMultiProcessEnv:
    """Vectorized Minesweeper using Ray actors."""

    def __init__(self, workers: list, seeds: list):
        self.workers = workers
        self.seeds = seeds

    def reset(self):
        futures = [w.reset.remote(seed=s) for w, s in zip(self.workers, self.seeds)]
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


def build_minesweeper_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                            is_train: bool = True, env_config=None) -> MinesweeperMultiProcessEnv:
    total = env_num * group_n
    cfg = getattr(env_config, "minesweeper", None)
    rows = getattr(cfg, "rows", 5) if cfg else 5
    cols = getattr(cfg, "cols", 5) if cfg else 5
    num_mines = getattr(cfg, "mines", 5) if cfg else 5
    max_turns = getattr(env_config, "max_steps", 25)
    invalid_penalty = getattr(env_config, "invalid_penalty", -1.0)
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"

    resources = getattr(env_config, "resources_per_worker", None)
    worker_kwargs = {}
    if resources is not None:
        from omegaconf import OmegaConf
        worker_kwargs = OmegaConf.to_container(resources, resolve=True)

    RemoteWorker = MinesweeperWorker.options(**worker_kwargs) if worker_kwargs else MinesweeperWorker
    workers, seeds = [], []
    for idx in range(total):
        episode_idx = idx // group_n
        actor_seed = seed + episode_idx
        workers.append(RemoteWorker.remote(
            seed=actor_seed, rows=rows, cols=cols, num_mines=num_mines,
            max_turns=max_turns, invalid_penalty=invalid_penalty,
            reward_mode=reward_mode,
        ))
        seeds.append(actor_seed)
    return MinesweeperMultiProcessEnv(workers=workers, seeds=seeds)
