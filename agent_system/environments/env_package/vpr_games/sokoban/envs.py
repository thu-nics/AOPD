"""Sokoban adapter, Ray actor, parallel env, and builder for VPR."""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.common.parser import (
    normalize_action_format,
    parse_action,
)
from agent_system.environments.env_package.vpr_games.common.rewards import outcome_reward

_ACTION_TO_ID = {
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
}
_ID_TO_ACTION = {v: k for k, v in _ACTION_TO_ID.items()}
_MOVES = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def _parse_sokoban_action(action_text: Optional[str]) -> Optional[int]:
    if action_text is None:
        return None
    return _ACTION_TO_ID.get(action_text.strip().lower())


def _boxes(room_state):
    return set(map(tuple, np.argwhere((room_state == 3) | (room_state == 4))))


def _apply_action(room_fixed, room_state, action_id: int):
    """Return the next room_state for an effective Sokoban action, or None."""
    move = _MOVES.get(action_id)
    if move is None:
        return None
    player_locations = np.argwhere(room_state == 5)
    if len(player_locations) == 0:
        return None
    player_pos = tuple(player_locations[0])
    next_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
    rows, cols = room_fixed.shape
    if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols):
        return None
    if room_fixed[next_pos] == 0:
        return None

    boxes = _boxes(room_state)
    next_state = room_state.copy()
    if next_pos in boxes:
        box_next = (next_pos[0] + move[0], next_pos[1] + move[1])
        if not (0 <= box_next[0] < rows and 0 <= box_next[1] < cols):
            return None
        if room_fixed[box_next] == 0 or box_next in boxes:
            return None
        next_state[next_pos] = room_fixed[next_pos]
        next_state[box_next] = 3 if room_fixed[box_next] == 2 else 4

    next_state[player_pos] = room_fixed[player_pos]
    next_state[next_pos] = 5
    return next_state


def _is_solved(room_state) -> bool:
    return not np.any(room_state == 4)


def _path_progress_completion(initial_shortest_path_len, remaining_shortest_path_len,
                              solved: bool = False) -> float:
    """Return normalized oracle-path progress while keeping the metric in [0, 1]."""
    if solved:
        return 1.0
    if (
        initial_shortest_path_len is None
        or initial_shortest_path_len <= 0
        or remaining_shortest_path_len is None
    ):
        return 0.0
    progress = (
        float(initial_shortest_path_len) - float(remaining_shortest_path_len)
    ) / float(initial_shortest_path_len)
    return float(np.clip(progress, 0.0, 1.0))


def _maybe_flip_process_reward(move_optimal: bool, oracle_reward: float, legal_non_oracle_reward: float,
                               reward_noise_prob: float):
    """Flip oracle/non-oracle process reward with probability p for reward-noise ablations."""
    reward = oracle_reward if move_optimal else legal_non_oracle_reward
    p = float(reward_noise_prob)
    if p <= 0.0:
        return reward, False
    if random.random() >= p:
        return reward, False
    flipped = legal_non_oracle_reward if move_optimal else oracle_reward
    return flipped, True


def _shortest_first_actions(room_fixed, room_state, max_depth: int):
    """Return all first actions that lie on a shortest solution path."""
    if _is_solved(room_state):
        return [], 0

    start = room_state.copy()
    queue = deque([(start, [])])
    seen_depth = {}
    best_depth = None
    first_actions = set()

    while queue:
        state, path = queue.popleft()
        if best_depth is not None and len(path) >= best_depth:
            continue
        if len(path) >= max_depth:
            continue

        key = state.tobytes()
        prev_depth = seen_depth.get(key)
        if prev_depth is not None and prev_depth < len(path):
            continue
        seen_depth[key] = len(path)

        for action_id in (1, 2, 3, 4):
            next_state = _apply_action(room_fixed, state, action_id)
            if next_state is None:
                continue
            next_path = path + [action_id]
            if _is_solved(next_state):
                depth = len(next_path)
                if best_depth is None or depth < best_depth:
                    best_depth = depth
                    first_actions = {next_path[0]}
                elif depth == best_depth:
                    first_actions.add(next_path[0])
                continue
            queue.append((next_state, next_path))

    if best_depth is None:
        return [], None
    return sorted(first_actions), best_depth


@ray.remote
class SokobanWorker:
    """Ray remote actor holding one text Sokoban instance for VPR."""

    def __init__(self, seed: int = 0, dim_room=(6, 6), num_boxes: int = 1,
                 max_steps: int = 15, search_depth: int = 30,
                 invalid_penalty: float = -2.0, oracle_reward: float = 2.0,
                 legal_non_oracle_reward: float = 0.0,
                 reward_mode: str = "oracle", reward_noise_prob: float = 0.0,
                 mode: str = "tiny_rgb_array", action_format: str = "action_tag"):
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        if not 0.0 <= float(reward_noise_prob) <= 1.0:
            raise ValueError(f"reward_noise_prob must be in [0, 1], got {reward_noise_prob!r}")
        from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv

        self._env = SokobanEnv(
            mode,
            dim_room=tuple(dim_room),
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth,
        )
        self._seed = seed
        self._mode = mode
        self._max_steps = int(max_steps)
        self._search_depth = int(search_depth)
        self._invalid_penalty = float(invalid_penalty)
        self._oracle_reward = float(oracle_reward)
        self._legal_non_oracle_reward = float(legal_non_oracle_reward)
        self._reward_mode = reward_mode
        self._reward_noise_prob = float(reward_noise_prob)
        self._action_format = normalize_action_format(action_format)
        self._step_count = 0
        self._done = False
        self._cached_state_key = None
        self._cached_depth_limit = None
        self._cached_oracle_actions = []
        self._cached_shortest_path_len = None
        self._initial_shortest_path_len = None

    def reset(self, seed=None):
        base_seed = seed if seed is not None else self._seed
        max_reset_attempts = 4  # initial seed + up to 3 reseeds
        last_obs = None
        last_actions = []
        last_shortest_len = None
        last_seed = base_seed
        last_attempt = 0

        for attempt in range(max_reset_attempts):
            s = base_seed + attempt
            obs, _ = self._env.reset(seed=s)
            self._step_count = 0
            self._done = False
            self._clear_oracle_cache()
            oracle_actions, shortest_len = self._oracle_for_current_state(self._search_depth)
            last_obs = obs
            last_actions = oracle_actions
            last_shortest_len = shortest_len
            last_seed = s
            last_attempt = attempt
            if shortest_len is not None:
                break

        self._initial_shortest_path_len = last_shortest_len
        info = self._build_info(
            raw="", parsed_action=None, parse_ok=True, illegal=False,
            action_effective=None, vpr_reward=0.0, terminal_success=None,
            terminal_reason=None, oracle_actions=last_actions, move_optimal=None,
            shortest_path_len=last_shortest_len,
            remaining_shortest_path_len=last_shortest_len,
        )
        info["reset_seed"] = int(last_seed)
        info["reset_retry_count"] = int(last_attempt)
        info["initial_oracle_found"] = last_shortest_len is not None
        return last_obs, info

    def _snapshot_state(self):
        return {
            "room_fixed": self._env.room_fixed.copy(),
            "room_state": self._env.room_state.copy(),
            "box_mapping": self._env.box_mapping.copy(),
            "player_position": self._env.player_position.copy(),
            "num_env_steps": int(self._env.num_env_steps),
            "reward_last": float(self._env.reward_last),
            "boxes_on_target": int(self._env.boxes_on_target),
            "step_count": int(self._step_count),
            "done": bool(self._done),
            "cached_state_key": self._cached_state_key,
            "cached_depth_limit": self._cached_depth_limit,
            "cached_oracle_actions": list(self._cached_oracle_actions),
            "cached_shortest_path_len": self._cached_shortest_path_len,
            "initial_shortest_path_len": self._initial_shortest_path_len,
        }

    def _restore_state(self, state):
        self._env.room_fixed = state["room_fixed"].copy()
        self._env.room_state = state["room_state"].copy()
        self._env.box_mapping = state["box_mapping"].copy()
        self._env.player_position = state["player_position"].copy()
        self._env.num_env_steps = int(state["num_env_steps"])
        self._env.reward_last = float(state["reward_last"])
        self._env.boxes_on_target = int(state["boxes_on_target"])
        self._step_count = int(state["step_count"])
        self._done = bool(state["done"])
        self._cached_state_key = state.get("cached_state_key")
        self._cached_depth_limit = state.get("cached_depth_limit")
        self._cached_oracle_actions = list(state.get("cached_oracle_actions") or [])
        self._cached_shortest_path_len = state.get("cached_shortest_path_len")
        self._initial_shortest_path_len = state.get("initial_shortest_path_len")

    def current_observation_info(self):
        oracle_actions, shortest_len = self._oracle_for_current_state(self._search_depth)
        info = self._build_info(
            raw="", parsed_action=None, parse_ok=True, illegal=False, action_effective=None,
            vpr_reward=0.0,
            terminal_success=True if self._done and self._env.boxes_on_target == self._env.num_boxes else None,
            terminal_reason="success" if self._done and self._env.boxes_on_target == self._env.num_boxes else None,
            oracle_actions=oracle_actions, move_optimal=None, shortest_path_len=shortest_len,
            remaining_shortest_path_len=shortest_len,
        )
        obs = info["observation"]
        return obs, info

    def snapshot_state(self):
        return self._snapshot_state()

    def restore_state(self, state):
        self._restore_state(state)
        return self.current_observation_info()

    def _state_key(self):
        return self._env.room_state.tobytes()

    def _clear_oracle_cache(self):
        self._cached_state_key = None
        self._cached_depth_limit = None
        self._cached_oracle_actions = []
        self._cached_shortest_path_len = None

    def _cache_oracle(self, actions, shortest_path_len, depth_limit):
        self._cached_state_key = self._state_key()
        self._cached_depth_limit = int(depth_limit)
        self._cached_oracle_actions = list(actions)
        self._cached_shortest_path_len = shortest_path_len

    def _oracle_for_current_state(self, depth_limit: int):
        depth_limit = int(max(0, depth_limit))
        key = self._state_key()
        if (
            self._cached_state_key == key
            and self._cached_depth_limit == depth_limit
        ):
            return list(self._cached_oracle_actions), self._cached_shortest_path_len
        actions, shortest_path_len = _shortest_first_actions(
            self._env.room_fixed, self._env.room_state, depth_limit
        )
        self._cache_oracle(actions, shortest_path_len, depth_limit)
        return actions, shortest_path_len

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
            reward = outcome_reward(done, info.get("terminal_success"), info.get("terminal_reason"))
            info["vpr_reward"] = reward
        return obs, float(reward), bool(done), info

    def _step_impl(self, raw_text: str):
        if self._done:
            obs = self._env.render(self._mode)
            solved = bool(self._env.success())
            remaining_shortest_len = (
                0 if solved else self._oracle_for_current_state(self._search_depth)[1]
            )
            return obs, 0.0, True, self._build_info(
                raw=raw_text, parsed_action=None, parse_ok=True, illegal=True,
                action_effective=False, vpr_reward=0.0,
                terminal_success=solved,
                terminal_reason="already_done", oracle_actions=[],
                move_optimal=None, shortest_path_len=None,
                remaining_shortest_path_len=remaining_shortest_len,
            )

        self._step_count += 1
        result = parse_action(raw_text, self._action_format)
        action_id = _parse_sokoban_action(result.action_text) if result.parse_ok else None

        if action_id is None:
            self._done = True
            obs = self._env.render(self._mode)
            remaining_shortest_len = self._oracle_for_current_state(self._search_depth)[1]
            return obs, self._invalid_penalty, True, self._build_info(
                raw=raw_text, parsed_action=result.action_text, parse_ok=result.parse_ok,
                illegal=True, action_effective=False, vpr_reward=self._invalid_penalty,
                terminal_success=False, terminal_reason="invalid_action",
                oracle_actions=[], move_optimal=None, shortest_path_len=None,
                remaining_shortest_path_len=remaining_shortest_len,
            )

        pre_action_depth = self._search_depth
        oracle_ids, shortest_len = self._oracle_for_current_state(pre_action_depth)
        next_state = _apply_action(self._env.room_fixed, self._env.room_state, action_id)
        if next_state is None:
            self._done = True
            obs = self._env.render(self._mode)
            info = self._build_info(
                raw=raw_text, parsed_action=result.action_text, parse_ok=True,
                illegal=True, action_effective=False, vpr_reward=self._invalid_penalty,
                terminal_success=False, terminal_reason="invalid_action",
                oracle_actions=oracle_ids, move_optimal=None,
                shortest_path_len=shortest_len,
                remaining_shortest_path_len=shortest_len,
            )
            return obs, self._invalid_penalty, True, info

        move_optimal = action_id in oracle_ids
        vpr_reward, reward_noisy = _maybe_flip_process_reward(
            move_optimal, self._oracle_reward, self._legal_non_oracle_reward, self._reward_noise_prob
        )

        obs, _, env_done, env_info = self._env.step(action_id)
        success = bool(env_info.get("won", False) or self._env.success())
        remaining_depth = self._search_depth
        post_oracle_ids, post_shortest_len = ([], 0) if success else self._oracle_for_current_state(remaining_depth)
        timed_out = bool(env_done or self._step_count >= self._max_steps)
        unsolvable = (not success) and (not timed_out) and post_shortest_len is None
        done = bool(success or timed_out or unsolvable)
        terminal_success = success if done else None
        terminal_reason = None
        if done:
            if success:
                terminal_reason = "complete"
            elif unsolvable:
                terminal_reason = "deadlock"
                vpr_reward = self._invalid_penalty
                reward_noisy = False
            else:
                terminal_reason = "timeout"
                vpr_reward = self._invalid_penalty
                reward_noisy = False
        else:
            self._cache_oracle(post_oracle_ids, post_shortest_len, remaining_depth)
        self._done = done

        info = self._build_info(
            raw=raw_text, parsed_action=result.action_text, parse_ok=True,
            illegal=False, action_effective=True, vpr_reward=vpr_reward,
            terminal_success=terminal_success, terminal_reason=terminal_reason,
            oracle_actions=oracle_ids, move_optimal=move_optimal,
            shortest_path_len=shortest_len,
            remaining_shortest_path_len=post_shortest_len,
        )
        info["reward_noise_applied"] = bool(reward_noisy)
        info["reward_noise_prob"] = float(self._reward_noise_prob)
        return obs, vpr_reward, done, info

    def _build_info(self, raw, parsed_action, parse_ok, illegal, action_effective,
                    vpr_reward, terminal_success, terminal_reason, oracle_actions,
                    move_optimal, shortest_path_len, remaining_shortest_path_len):
        obs = self._env.render(self._mode)
        completion = _path_progress_completion(
            self._initial_shortest_path_len,
            remaining_shortest_path_len,
            solved=_is_solved(self._env.room_state),
        )
        return {
            "env_name": "vpr_sokoban",
            "step": self._step_count,
            "max_steps": self._max_steps,
            "raw_action": raw,
            "parsed_action": parsed_action,
            "parse_ok": bool(parse_ok),
            "illegal_action": bool(illegal),
            "action_effective": action_effective,
            "available_actions": ["up", "down", "left", "right"],
            "vpr_reward": float(vpr_reward),
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
            "observation": obs,
            "completion_rate": completion,
            "boxes_on_target": int(self._env.boxes_on_target),
            "num_boxes": int(self._env.num_boxes),
            "oracle_valid_actions": [_ID_TO_ACTION[a] for a in oracle_actions],
            "oracle_action_set_size": int(len(oracle_actions)),
            "sokoban_shortest_path_len": (
                int(shortest_path_len) if shortest_path_len is not None else None
            ),
            "sokoban_initial_shortest_path_len": (
                int(self._initial_shortest_path_len)
                if self._initial_shortest_path_len is not None else None
            ),
            "sokoban_remaining_shortest_path_len": (
                int(remaining_shortest_path_len)
                if remaining_shortest_path_len is not None else None
            ),
            "move_optimal": move_optimal,
            "pre_exec_oracle_match": move_optimal,
            "legal_non_oracle": bool(move_optimal is False and parse_ok and not illegal),
        }


class SokobanMultiProcessEnv:
    """Vectorized Sokoban using Ray actors."""

    def __init__(self, workers: list, seeds: list):
        self.workers = workers
        self.seeds = seeds
        self._episode = 0

    def reset(self):
        offset = self._episode * 100003
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
            info["env_done"] = bool(info.get("terminal_reason") is not None)
        return obs_list, rewards, dones, info_list

    def step_candidate_groups(self, candidate_action_groups, active_indices=None,
                              selection_mode="best", random_select_prob=0.0):
        if active_indices is None:
            active_indices = list(range(len(candidate_action_groups)))
        futures = []
        for group_idx, env_idx in enumerate(active_indices):
            futures.append(
                self.workers[env_idx].step_candidate_group.remote(
                    candidate_action_groups[group_idx],
                    selection_mode=selection_mode,
                    random_select_prob=random_select_prob
                )
            )
        results = ray.get(futures)
        candidate_results = []
        selected_indices = []
        obs_list = []
        rewards = []
        dones = []
        infos = []
        for result in results:
            candidates, selected_idx, obs, reward, done, info = result
            candidate_results.append(candidates)
            selected_indices.append(selected_idx)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            candidate_results,
            selected_indices,
            obs_list,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

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


def build_sokoban_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                       is_train: bool = True, env_config=None) -> SokobanMultiProcessEnv:
    total = env_num * group_n
    cfg = getattr(env_config, "sokoban", None)
    dim_room = tuple(getattr(cfg, "dim_room", [6, 6])) if cfg else (6, 6)
    num_boxes = getattr(cfg, "num_boxes", 1) if cfg else 1
    search_depth = getattr(cfg, "search_depth", 30) if cfg else 30
    mode = getattr(cfg, "mode", "tiny_rgb_array") if cfg else "tiny_rgb_array"
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"
    oracle_reward = getattr(cfg, "oracle_reward", 2.0) if cfg else 2.0
    legal_non_oracle_reward = getattr(cfg, "legal_non_oracle_reward", 0.0) if cfg else 0.0
    reward_noise_prob = getattr(cfg, "reward_noise_prob", 0.0) if cfg else 0.0
    action_format = getattr(env_config, "game_action_format", "action_tag")
    invalid_penalty = getattr(env_config, "invalid_penalty", -2.0)
    max_steps = getattr(env_config, "max_steps", 15)

    resources = getattr(env_config, "resources_per_worker", None)
    worker_kwargs = {}
    if resources is not None:
        from omegaconf import OmegaConf
        worker_kwargs = OmegaConf.to_container(resources, resolve=True)

    RemoteWorker = SokobanWorker.options(**worker_kwargs) if worker_kwargs else SokobanWorker
    workers, seeds = [], []
    for idx in range(total):
        episode_idx = idx // group_n
        actor_seed = seed + episode_idx
        workers.append(RemoteWorker.remote(
            seed=actor_seed,
            dim_room=dim_room,
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth,
            invalid_penalty=invalid_penalty,
            oracle_reward=oracle_reward,
            legal_non_oracle_reward=legal_non_oracle_reward,
            reward_mode=reward_mode,
            reward_noise_prob=reward_noise_prob,
            mode=mode,
            action_format=action_format,
        ))
        seeds.append(actor_seed)
    return SokobanMultiProcessEnv(workers=workers, seeds=seeds)
