"""Minesweeper GEM adapter, Ray actor, parallel env, and builder for VPR."""

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
from agent_system.environments.env_package.vpr_games.minesweeper.oracle import compute_posteriors

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
                 max_turns: int = 25, invalid_penalty: float = -2.0,
                 reward_mode: str = "oracle", auto_reveal_center: bool = False,
                 oracle_reward: float = 2.0, oracle_flag_reward: float = 1.0,
                 oracle_guess_reward: float = 1.0, non_oracle_penalty: float = 0.0,
                 non_oracle_flag_penalty: float = -1.0,
                 oracle_policy: str = "all_oracle_actions",
                 action_format: str = "action_tag"):
        # auto_reveal_center: when True, reset() automatically performs the (always-safe,
        # by GEM first-click guarantee) opening reveal of the center cell, so the agent
        # starts from an informative board with the oracle already active. The opening
        # reveal is NOT counted as an agent step. The low-level worker defaults this to
        # False to preserve first-click test semantics; the training path enables it by
        # default via build_minesweeper_envs / the env config.
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        if oracle_policy not in ("all_oracle_actions", "safe_reveal_first"):
            raise ValueError(
                f"oracle_policy must be 'all_oracle_actions', got {oracle_policy!r}")
        if oracle_policy == "safe_reveal_first":
            oracle_policy = "all_oracle_actions"
        from gem.envs.game_env.minesweeper import MinesweeperEnv
        self._env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines, max_turns=max_turns)
        self._seed = seed
        self._reward_mode = reward_mode
        self._rows = rows
        self._cols = cols
        self._num_mines = num_mines
        self._max_steps = max_turns
        self._invalid_penalty = invalid_penalty
        self._auto_reveal_center = auto_reveal_center
        self._oracle_reward = float(oracle_reward)
        self._oracle_flag_reward = float(oracle_flag_reward)
        self._oracle_guess_reward = float(oracle_guess_reward)
        self._non_oracle_penalty = float(non_oracle_penalty)
        self._non_oracle_flag_penalty = float(non_oracle_flag_penalty)
        self._oracle_policy = oracle_policy
        self._action_format = normalize_action_format(action_format)
        self._step_count = 0
        self._done = False
        self._first_revealed = False

    def reset(self, seed=None):
        s = seed if seed is not None else self._seed
        self._env.reset(seed=s)
        self._step_count = 0
        self._done = False
        self._first_revealed = False
        # Optional opening move: reveal the center cell (always safe via GEM first-click
        # safety). This is a free board reveal, not an agent step, so _step_count stays 0.
        if self._auto_reveal_center:
            rc, cc = self._rows // 2, self._cols // 2
            if not self._env.revealed[rc][cc]:
                self._env.step(f"\\boxed{{reveal {rc} {cc}}}")
            self._first_revealed = True
        completion = 0.0
        if self._first_revealed:
            total_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                             if self._env.grid[r][c] != -1)
            revealed_safe = sum(1 for r in range(self._rows) for c in range(self._cols)
                                if self._env.grid[r][c] != -1 and self._env.revealed[r][c])
            completion = revealed_safe / total_safe if total_safe > 0 else 0.0
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
            "completion_rate": completion,
            "oracle_degraded": False,
            "flagged_cells": flagged,
            "move_optimal": None,
            "pre_exec_oracle_match": None,
            "legal_non_oracle": False,
            "oracle_action_set_size": 0,
            "oracle_guess": False,
            "oracle_policy": self._oracle_policy,
            "oracle_policy_tier": None,
            "oracle_tier": None,
            "safe_reveal_available": False,
            "certain_flag_available": False,
            "guess_required": False,
            "reveal_posterior_margin": None,
        }
        return obs_text, info

    def _snapshot_state(self):
        return {
            "grid": [row[:] for row in self._env.grid],
            "revealed": [row[:] for row in self._env.revealed],
            "flags": [row[:] for row in self._env.flags],
            "first_reveal": bool(self._env.first_reveal),
            "turn_count": int(self._env.turn_count),
            "step_count": int(self._step_count),
            "done": bool(self._done),
            "first_revealed": bool(self._first_revealed),
        }

    def _restore_state(self, state):
        self._env.grid = [row[:] for row in state["grid"]]
        self._env.revealed = [row[:] for row in state["revealed"]]
        self._env.flags = [row[:] for row in state["flags"]]
        self._env.first_reveal = bool(state["first_reveal"])
        self._env.turn_count = int(state["turn_count"])
        self._step_count = int(state["step_count"])
        self._done = bool(state["done"])
        self._first_revealed = bool(state["first_revealed"])

    def current_observation_info(self):
        obs_text = _render_board(self._env.revealed, self._env.grid, self._env.flags, self._rows, self._cols)
        cleared = all(
            self._env.grid[r][c] == -1 or self._env.revealed[r][c]
            for r in range(self._rows) for c in range(self._cols)
        )
        info = self._build_info(
            raw="", parsed_action=None, parse_ok=True, illegal=False, vpr_reward=0.0,
            terminal_success=True if self._done and cleared else None,
            terminal_reason="success" if self._done and cleared else None,
            min_prob=None, post_prob=None, oracle_actions=[], move_optimal=None,
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

    def _policy_oracle_actions(self, posteriors):
        """Select turn-level imitation labels as a union of oracle action types.

        Safe reveals and certain flags are always oracle actions. Minimum-posterior
        guess reveals are oracle actions only when no safe reveal exists.
        """
        unrevealed_unflagged = sorted(
            cell for cell in posteriors
            if not self._env.revealed[cell[0]][cell[1]] and not self._env.flags[cell[0]][cell[1]]
        )
        if not unrevealed_unflagged:
            return {
                "oracle_actions": [],
                "oracle_action_tiers": {},
                "min_prob": None,
                "oracle_policy_tier": None,
                "safe_reveal_available": False,
                "certain_flag_available": False,
                "guess_required": False,
            }

        probs = {cell: posteriors[cell] for cell in unrevealed_unflagged}
        min_prob = min(probs.values())
        eps = 1e-9
        safe_reveals = [cell for cell, p in probs.items() if abs(p) < eps]
        certain_flags = [cell for cell, p in probs.items() if p == 1.0]
        guess_reveals = []
        if not safe_reveals and min_prob is not None and eps < min_prob < 1.0 - eps:
            guess_reveals = [cell for cell, p in probs.items() if abs(p - min_prob) < eps]

        oracle_actions = []
        oracle_action_tiers = {}
        for r, c in safe_reveals:
            action = f"reveal {r+1} {c+1}"
            oracle_actions.append(action)
            oracle_action_tiers[action] = "safe_reveal"
        for r, c in certain_flags:
            action = f"flag {r+1} {c+1}"
            oracle_actions.append(action)
            oracle_action_tiers[action] = "certain_flag"
        for r, c in guess_reveals:
            action = f"reveal {r+1} {c+1}"
            oracle_actions.append(action)
            oracle_action_tiers[action] = "guess"

        return {
            "oracle_actions": oracle_actions,
            "oracle_action_tiers": oracle_action_tiers,
            "min_prob": min_prob,
            "oracle_policy_tier": self._oracle_policy,
            "safe_reveal_available": bool(safe_reveals),
            "certain_flag_available": bool(certain_flags),
            "guess_required": bool(guess_reveals),
        }

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
        result = parse_action(raw_text, self._action_format)
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

        # Compute oracle BEFORE executing action (state is current). Safe reveals
        # and certain flags are both oracle actions; minimum-risk guesses only
        # become oracle actions when no safe reveal exists.
        posteriors, oracle_degraded = {}, False
        oracle_actions = []
        post_prob = None
        min_prob = None
        oracle_policy_tier = None
        oracle_action_tiers = {}
        safe_reveal_available = False
        certain_flag_available = False
        guess_required = False
        if self._first_revealed:
            posteriors, oracle_degraded = compute_posteriors(
                self._env.revealed, self._env.grid,
                self._rows, self._cols, self._num_mines
            )
            policy = self._policy_oracle_actions(posteriors)
            oracle_actions = policy["oracle_actions"]
            oracle_action_tiers = policy["oracle_action_tiers"]
            min_prob = policy["min_prob"]
            oracle_policy_tier = policy["oracle_policy_tier"]
            safe_reveal_available = policy["safe_reveal_available"]
            certain_flag_available = policy["certain_flag_available"]
            guess_required = policy["guess_required"]
            post_prob = posteriors.get((r0, c0), None)
        else:
            # Before first reveal: any unrevealed reveal action is safe by GEM
            # first-click safety. This branch mainly preserves low-level tests;
            # training normally starts after a free center reveal.
            oracle_policy_tier = self._oracle_policy
            safe_reveal_available = True
            oracle_actions = [
                f"reveal {rr+1} {cc+1}"
                for rr in range(self._rows) for cc in range(self._cols)
                if not self._env.revealed[rr][cc] and not self._env.flags[rr][cc]
            ]
            oracle_action_tiers = {action: "safe_reveal" for action in oracle_actions}

        action_str = f"{action_type} {row} {col}"
        move_optimal = action_str in oracle_actions
        oracle_tier = oracle_action_tiers.get(action_str) if move_optimal else None
        if move_optimal:
            if oracle_tier == "certain_flag":
                vpr_reward = self._oracle_flag_reward
            elif oracle_tier == "guess":
                vpr_reward = self._oracle_guess_reward
            else:
                vpr_reward = self._oracle_reward
        else:
            vpr_reward = (
                self._non_oracle_flag_penalty if action_type == "flag"
                else self._non_oracle_penalty
            )
        oracle_guess = bool(move_optimal and oracle_tier == "guess")
        reveal_posterior_margin = (
            float(post_prob - min_prob)
            if action_type == "reveal" and post_prob is not None and min_prob is not None
            else None
        )

        if action_type == "flag" and not move_optimal:
            self._done = True
            obs_text = _render_board(self._env.revealed, self._env.grid,
                                     self._env.flags, self._rows, self._cols)
            info = self._build_info(raw_text, result.action_text, True, False,
                                    vpr_reward, False, "non_oracle_flag",
                                    min_prob, post_prob, oracle_actions,
                                    move_optimal=move_optimal, oracle_guess=oracle_guess,
                                    oracle_policy_tier=oracle_policy_tier,
                                    oracle_tier=oracle_tier,
                                    safe_reveal_available=safe_reveal_available,
                                    certain_flag_available=certain_flag_available,
                                    guess_required=guess_required,
                                    reveal_posterior_margin=reveal_posterior_margin)
            info["oracle_degraded"] = oracle_degraded
            return obs_text, vpr_reward, True, info

        # Execute action via GEM. Non-oracle flags returned above, so any flag
        # reaching this branch is an oracle flag on an unflagged cell.
        if action_type == "flag":
            gem_action = f"\\boxed{{flag {r0} {c0}}}"
            _, _, gem_terminated, gem_truncated, _ = self._env.step(gem_action)
        else:  # reveal
            gem_action = f"\\boxed{{reveal {r0} {c0}}}"
            _, gem_rew, gem_terminated, gem_truncated, _ = self._env.step(gem_action)
            if not self._first_revealed:
                self._first_revealed = True
            # Detect mine hit from the grid value (GEM fail_reward is 0.0, not negative)
            if gem_terminated and self._env.grid[r0][c0] < 0:
                # Mine hit only affects the terminal outcome. For imitation, the reward
                # and move_optimal label are determined by the pre-execution oracle set.
                self._done = True
                obs_text = _render_board(self._env.revealed, self._env.grid,
                                         self._env.flags, self._rows, self._cols)
                info = self._build_info(raw_text, result.action_text, True, False,
                                        vpr_reward, False, "mine_hit", min_prob, post_prob, oracle_actions,
                                        move_optimal=move_optimal, oracle_guess=oracle_guess,
                                        oracle_policy_tier=oracle_policy_tier, oracle_tier=oracle_tier,
                                        safe_reveal_available=safe_reveal_available,
                                        certain_flag_available=certain_flag_available,
                                        guess_required=guess_required,
                                        reveal_posterior_margin=reveal_posterior_margin)
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
        if terminal_reason == "timeout":
            vpr_reward = self._invalid_penalty

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
                                min_prob, post_prob, oracle_actions,
                                move_optimal=move_optimal, oracle_guess=oracle_guess,
                                oracle_policy_tier=oracle_policy_tier, oracle_tier=oracle_tier,
                                safe_reveal_available=safe_reveal_available,
                                certain_flag_available=certain_flag_available,
                                guess_required=guess_required,
                                reveal_posterior_margin=reveal_posterior_margin)
        info["oracle_degraded"] = oracle_degraded
        info["completion_rate"] = completion_rate
        return obs_text, vpr_reward, done, info

    def _build_info(self, raw, parsed_action, parse_ok, illegal, vpr_reward,
                    terminal_success, terminal_reason, min_prob, post_prob, oracle_actions,
                    move_optimal=None, oracle_guess=False, oracle_policy_tier=None,
                    oracle_tier=None, safe_reveal_available=False,
                    certain_flag_available=False, guess_required=False,
                    reveal_posterior_margin=None):
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
            # Whether the action matched the safe-cell oracle (set only on legal moves;
            # None on illegal / parse-failure / already-done steps).
            "move_optimal": move_optimal,
            "pre_exec_oracle_match": move_optimal,
            "legal_non_oracle": bool(move_optimal is False and parse_ok and not illegal),
            "oracle_action_set_size": int(len(oracle_actions) if oracle_actions else 0),
            "oracle_guess": bool(oracle_guess),
            "oracle_policy": self._oracle_policy,
            "oracle_policy_tier": oracle_policy_tier,
            "oracle_tier": oracle_tier,
            "safe_reveal_available": bool(safe_reveal_available),
            "certain_flag_available": bool(certain_flag_available),
            "guess_required": bool(guess_required),
            "reveal_posterior_margin": (
                float(reveal_posterior_margin) if reveal_posterior_margin is not None else None
            ),
        }

    def close(self):
        pass


class MinesweeperMultiProcessEnv:
    """Vectorized Minesweeper using Ray actors."""

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


def build_minesweeper_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                            is_train: bool = True, env_config=None) -> MinesweeperMultiProcessEnv:
    total = env_num * group_n
    cfg = getattr(env_config, "minesweeper", None)
    rows = getattr(cfg, "rows", 5) if cfg else 5
    cols = getattr(cfg, "cols", 5) if cfg else 5
    num_mines = getattr(cfg, "mines", 5) if cfg else 5
    max_turns = getattr(env_config, "max_steps", 25)
    invalid_penalty = getattr(env_config, "invalid_penalty", -2.0)
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"
    # Training default: ON — start every episode from a center reveal.
    auto_reveal_center = getattr(cfg, "auto_reveal_center", True) if cfg else True
    oracle_reward = getattr(cfg, "oracle_reward", 2.0) if cfg else 2.0
    oracle_flag_reward = getattr(cfg, "oracle_flag_reward", 1.0) if cfg else 1.0
    oracle_guess_reward = getattr(cfg, "oracle_guess_reward", 1.0) if cfg else 1.0
    non_oracle_penalty = getattr(cfg, "non_oracle_penalty", 0.0) if cfg else 0.0
    non_oracle_flag_penalty = getattr(cfg, "non_oracle_flag_penalty", -1.0) if cfg else -1.0
    oracle_policy = getattr(cfg, "oracle_policy", "all_oracle_actions") if cfg else "all_oracle_actions"
    action_format = getattr(env_config, "game_action_format", "action_tag")

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
            reward_mode=reward_mode, auto_reveal_center=auto_reveal_center,
            oracle_reward=oracle_reward, oracle_flag_reward=oracle_flag_reward,
            oracle_guess_reward=oracle_guess_reward,
            non_oracle_penalty=non_oracle_penalty,
            non_oracle_flag_penalty=non_oracle_flag_penalty,
            oracle_policy=oracle_policy,
            action_format=action_format,
        ))
        seeds.append(actor_seed)
    return MinesweeperMultiProcessEnv(workers=workers, seeds=seeds)
