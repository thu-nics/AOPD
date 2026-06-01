"""TicTacToe Ray actor, parallel env, and builder for VPR integration."""

from __future__ import annotations

import numpy as np
import ray

from agent_system.environments.env_package.vpr_games.tictactoe.game import TicTacToeGame
from agent_system.environments.env_package.vpr_games.common.parser import parse_action_tag


@ray.remote
class TicTacToeWorker:
    """Ray remote actor holding one TicTacToe instance."""

    def __init__(self, seed: int = 0, opponent: str = "random",
                 invalid_action_terminates: bool = True,
                 max_steps: int = 9, invalid_penalty: float = -1.0):
        self._game = TicTacToeGame(
            opponent=opponent,
            invalid_action_terminates=invalid_action_terminates,
            max_steps=max_steps,
            invalid_penalty=invalid_penalty,
            seed=seed,
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

    def close(self):
        pass


class TicTacToeMultiProcessEnv:
    """Vectorized TicTacToe using Ray actors."""

    def __init__(self, workers: list, seeds: list):
        self.workers = workers
        self.seeds = seeds
        self._batch_size = len(workers)

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


def build_tictactoe_envs(seed: int = 0, env_num: int = 1, group_n: int = 1,
                          is_train: bool = True, env_config=None) -> TicTacToeMultiProcessEnv:
    total = env_num * group_n
    opponent = getattr(env_config, "tictactoe", None)
    opponent_type = getattr(opponent, "opponent", "random") if opponent else "random"
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
        ))
        seeds.append(actor_seed)

    return TicTacToeMultiProcessEnv(workers=workers, seeds=seeds)
