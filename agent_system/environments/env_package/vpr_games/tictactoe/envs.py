"""TicTacToe Ray actor, parallel env, and builder for VPR integration."""

from __future__ import annotations

import random

import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.tictactoe.game import TicTacToeGame
from agent_system.environments.env_package.vpr_games.common.parser import parse_action_tag


@ray.remote
class TicTacToeWorker:
    """Ray remote actor holding one TicTacToe instance."""

    def __init__(self, seed: int = 0, opponent: str = "random",
                 invalid_action_terminates: bool = True,
                 max_steps: int = 9, invalid_penalty: float = -1.0,
                 reward_mode: str = "oracle", agent_player: str = "X",
                 mcts_max_simulations: int = 1000, mcts_uct_c: float = 2.0,
                 mcts_rollout_count: int = 1):
        self._game = TicTacToeGame(
            opponent=opponent,
            invalid_action_terminates=invalid_action_terminates,
            max_steps=max_steps,
            invalid_penalty=invalid_penalty,
            seed=seed,
            reward_mode=reward_mode,
            agent_player=agent_player,
            mcts_max_simulations=mcts_max_simulations,
            mcts_uct_c=mcts_uct_c,
            mcts_rollout_count=mcts_rollout_count,
        )
        self._seed = seed

    def reset(self, seed=None):
        s = seed if seed is not None else self._seed
        obs, info = self._game.reset(seed=s)
        return obs, info

    def step(self, raw_text: str):
        result = parse_action_tag(raw_text)
        obs, reward, done, info = self._game.step(
            action_text=result.action_text,
            parse_ok=result.parse_ok,
            raw_action=raw_text,
        )
        return obs, float(reward), bool(done), info

    def current_observation_info(self):
        obs, info = self._game.current_observation_info()
        info["observation"] = obs
        return obs, info

    def snapshot_state(self):
        return self._game.snapshot_state()

    def restore_state(self, state):
        obs, info = self._game.restore_state(state)
        info["observation"] = obs
        return obs, info

    def step_candidate_group(self, raw_texts, selection_mode="best", random_select_prob=0.0):
        """Evaluate candidates from the same TicTacToe state, then commit one."""
        snapshot = self._game.snapshot_state()
        candidates = []
        best_idx = 0
        best_reward = None
        for idx, raw_text in enumerate(raw_texts):
            self._game.restore_state(snapshot)
            obs, reward, done, info = self.step(raw_text)
            info["observation"] = obs
            info["candidate_index"] = idx
            info["env_done"] = bool(done)
            candidates.append((obs, float(reward), bool(done), info))
            if best_reward is None or float(reward) > best_reward:
                best_reward = float(reward)
                best_idx = idx

        if not raw_texts:
            self._game.restore_state(snapshot)
            obs, info = self.current_observation_info()
            return candidates, 0, obs, 0.0, bool(info.get("terminal_success") is not None), info

        selected_idx = best_idx
        selection_type = "best"
        if selection_mode == "mixed" and random.random() < float(random_select_prob):
            selected_idx = random.randrange(len(raw_texts))
            selection_type = "random"
        elif selection_mode == "random":
            selected_idx = random.randrange(len(raw_texts))
            selection_type = "random"

        self._game.restore_state(snapshot)
        selected_obs, selected_reward, selected_done, selected_info = self.step(raw_texts[selected_idx])
        selected_info["observation"] = selected_obs
        selected_info["candidate_index"] = selected_idx
        selected_info["best_candidate_index"] = best_idx
        selected_info["state_group_selected"] = True
        selected_info["state_group_selection_type"] = selection_type
        selected_info["state_group_random_selected"] = selection_type == "random"
        selected_info["env_done"] = bool(selected_done)
        return candidates, selected_idx, selected_obs, float(selected_reward), bool(selected_done), selected_info

    def close(self):
        pass


class TicTacToeMultiProcessEnv:
    """Vectorized TicTacToe using Ray actors."""

    def __init__(self, workers: list, seeds: list):
        self.workers = workers
        self.seeds = seeds
        self._batch_size = len(workers)
        # Episode counter: advanced once per reset() so each rollout (i.e. each training
        # step) draws a *fresh* random-opponent realization instead of replaying the same
        # fixed per-slot seed every step. Group replicas keep an identical seed within a
        # step (same base seed + same counter), so GRPO groups stay comparable; the run is
        # still fully reproducible from `env.seed`.
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

    def step_candidate_groups(self, candidate_action_groups, active_indices=None,
                              selection_mode="best", random_select_prob=0.0,
                              ):
        if active_indices is None:
            active_indices = list(range(self._batch_size))
        futures = [
            self.workers[env_idx].step_candidate_group.remote(
                candidate_action_groups[pos], selection_mode=selection_mode,
                random_select_prob=random_select_prob,
            )
            for pos, env_idx in enumerate(active_indices)
        ]
        results = ray.get(futures) if futures else []
        candidate_results = [r[0] for r in results]
        selected_indices = [int(r[1]) for r in results]
        obs_list = [r[2] for r in results]
        rewards = np.array([r[3] for r in results], dtype=np.float32)
        dones = np.array([r[4] for r in results], dtype=bool)
        info_list = [r[5] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return candidate_results, selected_indices, obs_list, rewards, dones, info_list

    def snapshot_states(self, active_indices=None):
        if active_indices is None:
            active_indices = list(range(self._batch_size))
        futures = [self.workers[i].snapshot_state.remote() for i in active_indices]
        return ray.get(futures) if futures else []

    def restore_states(self, snapshots):
        futures = [w.restore_state.remote(state) for w, state in zip(self.workers, snapshots)]
        results = ray.get(futures) if futures else []
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        for obs, info in zip(obs_list, info_list):
            info["observation"] = obs
        return obs_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)


def build_tictactoe_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                          is_train: bool = True, env_config=None) -> TicTacToeMultiProcessEnv:
    total = env_num * group_n
    cfg = getattr(env_config, "tictactoe", None)
    opponent_type = getattr(cfg, "opponent", "random") if cfg else "random"
    reward_mode = getattr(cfg, "reward_mode", "oracle") if cfg else "oracle"
    agent_player = getattr(cfg, "agent_player", "X") if cfg else "X"
    mcts_cfg = getattr(cfg, "mcts", None) if cfg else None
    mcts_max_simulations = getattr(mcts_cfg, "max_simulations", 1000) if mcts_cfg else 1000
    mcts_uct_c = getattr(mcts_cfg, "uct_c", 2.0) if mcts_cfg else 2.0
    mcts_rollout_count = getattr(mcts_cfg, "rollout_count", 1) if mcts_cfg else 1
    invalid_penalty = getattr(env_config, "invalid_penalty", -1.0)
    max_steps = getattr(env_config, "max_steps", 9)

    resources = getattr(env_config, "resources_per_worker", None)
    worker_kwargs = {}
    if resources is not None:
        from omegaconf import OmegaConf
        worker_kwargs = OmegaConf.to_container(resources, resolve=True)

    RemoteWorker = TicTacToeWorker.options(**worker_kwargs) if worker_kwargs else TicTacToeWorker
    workers = []
    seeds = []
    for idx in range(total):
        # All group_n replicas of the same episode share the same episode seed
        episode_idx = idx // group_n
        actor_seed = seed + episode_idx
        workers.append(RemoteWorker.remote(
            seed=actor_seed,
            opponent=opponent_type,
            invalid_action_terminates=True,
            max_steps=max_steps,
            invalid_penalty=invalid_penalty,
            reward_mode=reward_mode,
            agent_player=agent_player,
            mcts_max_simulations=mcts_max_simulations,
            mcts_uct_c=mcts_uct_c,
            mcts_rollout_count=mcts_rollout_count,
        ))
        seeds.append(actor_seed)

    return TicTacToeMultiProcessEnv(workers=workers, seeds=seeds)
