"""Heterogeneous vector environment for mixed-game VPR training."""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np
import ray
from omegaconf import OmegaConf
from agent_system.environments.env_package.math_reasoning.envs import MathReasoningEnvs



GAME_ORDER = ("sokoban", "sudoku", "minesweeper")
TASK_ORDER = ("math", *GAME_ORDER)


def interleave_counts(counts: Mapping[str, int]) -> list[str]:
    """Spread each game across the worker order using weighted fair scheduling."""
    normalized = {game: int(counts.get(game, 0)) for game in TASK_ORDER}
    if any(count < 0 for count in normalized.values()):
        raise ValueError("mixed trajectory counts must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("mixed trajectory counts must contain at least one trajectory")

    used = {game: 0 for game in TASK_ORDER}
    labels = []
    for slot in range(total):
        candidates = [game for game in TASK_ORDER if used[game] < normalized[game]]
        game = max(
            candidates,
            key=lambda candidate: (
                normalized[candidate] * (slot + 1) / total - used[candidate],
                -TASK_ORDER.index(candidate),
            ),
        )
        labels.append(game)
        used[game] += 1
    return labels


def interleave_grouped_counts(counts: Mapping[str, int], group_n: int) -> list[str]:
    """Interleave base groups, then keep each group's members contiguous."""
    if isinstance(group_n, bool) or not isinstance(group_n, int) or group_n <= 0:
        raise ValueError("mixed rollout group_n must be a positive integer")
    return [
        task
        for task in interleave_counts(counts)
        for _ in range(group_n)
    ]


def _game_env_config(env_config, game: str):
    config = OmegaConf.create(OmegaConf.to_container(env_config, resolve=True))
    game_config = config[game]
    config.max_steps = int(game_config.max_steps)
    config.invalid_penalty = float(game_config.invalid_penalty)
    return config


class MixedVPRMultiProcessEnv:
    def __init__(self, workers, seeds, games):
        if not (len(workers) == len(seeds) == len(games)):
            raise ValueError("workers, seeds, and games must have equal lengths")
        self.workers = workers
        self.seeds = seeds
        self.games = games
        self._math_global_indices = [index for index, task in enumerate(games) if task == "math"]
        self._math_local_by_global = {
            global_index: local_index
            for local_index, global_index in enumerate(self._math_global_indices)
        }
        self._math_envs = MathReasoningEnvs(len(self._math_global_indices))
        self._episode = 0

    def _annotate(self, game, observation, info):
        info["observation"] = observation
        info["vpr_game"] = game
        return info

    def reset(self, kwargs=None):
        if kwargs is None:
            if self._math_global_indices:
                raise ValueError("mixed math rows require env_kwargs")
            kwargs = [{} for _ in self.games]
        if len(kwargs) != len(self.games):
            raise ValueError(f"expected {len(self.games)} env kwargs, got {len(kwargs)}")
        for expected_task, item in zip(self.games, kwargs, strict=True):
            row_task = str(item.get("task", expected_task))
            if row_task != expected_task:
                raise ValueError(
                    f"mixed row task {row_task!r} does not match environment slot {expected_task!r}"
                )

        offset = self._episode * 100003
        self._episode += 1
        observations = [""] * len(self.games)
        infos = [{} for _ in self.games]

        math_kwargs = [dict(kwargs[index]) for index in self._math_global_indices]
        if math_kwargs:
            math_observations, math_infos = self._math_envs.reset(math_kwargs)
            for global_index, observation, info in zip(
                self._math_global_indices, math_observations, math_infos, strict=True
            ):
                observations[global_index] = observation
                infos[global_index] = self._annotate("math", observation, info)

        game_indices = [index for index, task in enumerate(self.games) if task != "math"]
        futures = [
            self.workers[index].reset.remote(seed=self.seeds[index] + offset)
            for index in game_indices
        ]
        for index, result in zip(game_indices, ray.get(futures), strict=True):
            observation, info = result
            observations[index] = observation
            infos[index] = self._annotate(self.games[index], observation, info)
        return observations, infos

    def step(self, actions):
        if len(actions) != len(self.workers):
            raise ValueError(f"expected {len(self.workers)} actions, got {len(actions)}")
        observations = [""] * len(self.games)
        rewards = np.zeros(len(self.games), dtype=np.float32)
        dones = np.zeros(len(self.games), dtype=bool)
        infos = [{} for _ in self.games]

        if self._math_global_indices:
            math_actions = [actions[index] for index in self._math_global_indices]
            math_results = self._math_envs.step(math_actions)
            for local_index, global_index in enumerate(self._math_global_indices):
                observation = math_results[0][local_index]
                observations[global_index] = observation
                rewards[global_index] = math_results[1][local_index]
                dones[global_index] = math_results[2][local_index]
                infos[global_index] = self._annotate(
                    "math", observation, math_results[3][local_index]
                )

        game_indices = [index for index, task in enumerate(self.games) if task != "math"]
        futures = [self.workers[index].step.remote(actions[index]) for index in game_indices]
        for index, result in zip(game_indices, ray.get(futures), strict=True):
            observation, reward, done, info = result
            observations[index] = observation
            rewards[index] = reward
            dones[index] = done
            infos[index] = self._annotate(self.games[index], observation, info)
        return observations, rewards, dones, infos

    def step_candidate_groups(
        self,
        candidate_action_groups,
        active_indices=None,
        selection_mode="best",
        random_select_prob=0.0,
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        worker_indices = [int(index) for index in active_indices]
        if len(worker_indices) != len(candidate_action_groups):
            raise ValueError("active_indices must align with candidate_action_groups")

        results = [None] * len(worker_indices)
        game_positions = [
            position
            for position, worker_index in enumerate(worker_indices)
            if self.games[worker_index] != "math"
        ]
        game_futures = [
            self.workers[worker_indices[position]].step_candidate_group.remote(
                candidate_action_groups[position],
                selection_mode=selection_mode,
                random_select_prob=random_select_prob,
            )
            for position in game_positions
        ]
        for position, result in zip(game_positions, ray.get(game_futures), strict=True):
            results[position] = result

        math_positions = [
            position
            for position, worker_index in enumerate(worker_indices)
            if self.games[worker_index] == "math"
        ]
        if math_positions:
            math_results = self._math_envs.step_candidate_groups(
                [candidate_action_groups[position] for position in math_positions],
                active_indices=[
                    self._math_local_by_global[worker_indices[position]]
                    for position in math_positions
                ],
                selection_mode=selection_mode,
                random_select_prob=random_select_prob,
            )
            for local_position, position in enumerate(math_positions):
                results[position] = (
                    math_results[0][local_position],
                    int(math_results[1][local_position]),
                    math_results[2][local_position],
                    float(math_results[3][local_position]),
                    bool(math_results[4][local_position]),
                    math_results[5][local_position],
                )

        candidate_results = []
        selected_indices = []
        observations = []
        rewards = []
        dones = []
        infos = []
        for worker_index, result in zip(worker_indices, results, strict=True):
            game = self.games[worker_index]
            candidates, selected_index, observation, reward, done, info = result
            annotated_candidates = []
            for candidate_observation, candidate_reward, candidate_done, candidate_info in candidates:
                annotated_candidates.append(
                    (
                        candidate_observation,
                        candidate_reward,
                        candidate_done,
                        self._annotate(game, candidate_observation, candidate_info),
                    )
                )
            candidate_results.append(annotated_candidates)
            selected_indices.append(selected_index)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(self._annotate(game, observation, info))
        return (
            candidate_results,
            np.asarray(selected_indices, dtype=np.int32),
            observations,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def snapshot_states(self, active_indices=None):
        raise NotImplementedError("mixed VPR does not support snapshot-based VinePPO rollouts")

    def restore_states(self, snapshots):
        raise NotImplementedError("mixed VPR does not support snapshot-based VinePPO rollouts")

    def close(self):
        self._math_envs.close()
        for worker in self.workers:
            if worker is not None:
                ray.kill(worker)


def build_mixed_vpr_envs(
    seed: int,
    counts: Mapping[str, int],
    env_config,
    *,
    is_train: bool,
    group_n: int = 1,
):
    from agent_system.environments.env_package.vpr_games.minesweeper.envs import build_minesweeper_envs
    from agent_system.environments.env_package.vpr_games.sokoban.envs import build_sokoban_envs
    from agent_system.environments.env_package.vpr_games.sudoku.envs import build_sudoku_envs

    normalized = {task: int(counts.get(task, 0)) for task in TASK_ORDER}
    builders = {
        "sokoban": build_sokoban_envs,
        "sudoku": build_sudoku_envs,
        "minesweeper": build_minesweeper_envs,
    }
    seed_offsets = {"sokoban": 0, "sudoku": 10000, "minesweeper": 20000}
    if isinstance(group_n, bool) or not isinstance(group_n, int) or group_n <= 0:
        raise ValueError("mixed rollout group_n must be a positive integer")

    pools = {
        "math": deque(
            (None, seed + 30000 + index)
            for index in range(normalized["math"] * group_n)
        )
    }
    for game in GAME_ORDER:
        if normalized[game] == 0:
            pools[game] = deque()
            continue
        vector_env = builders[game](
            seed=seed + seed_offsets[game],
            env_num=normalized[game],
            group_n=group_n,
            is_train=is_train,
            env_config=_game_env_config(env_config, game),
        )
        pools[game] = deque(zip(vector_env.workers, vector_env.seeds))

    workers, seeds, games = [], [], []
    for task in interleave_grouped_counts(normalized, group_n):
        worker, worker_seed = pools[task].popleft()
        workers.append(worker)
        seeds.append(worker_seed)
        games.append(task)
    return MixedVPRMultiProcessEnv(workers, seeds, games)
