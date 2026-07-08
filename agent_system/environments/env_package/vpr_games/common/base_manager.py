"""VPRBaseEnvironmentManager — shared base for all VPR environments."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from agent_system.environments.env_manager import EnvironmentManagerBase


class VPRBaseEnvironmentManager(EnvironmentManagerBase):
    """Base manager for VPR environments.

    Enforces history_length=0 (Markovian observations), returns the standard
    verl-agent observation dict, emits is_action_valid, and overrides
    success_evaluator() to use terminal_success from info.
    """

    def __init__(self, envs, projection_f, config):
        history_len = getattr(config.env, "history_length", 0)
        if history_len != 0:
            raise ValueError(
                f"VPR environments require history_length=0, got {history_len}. "
                "Set env.history_length=0 in your Hydra config."
            )
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs=None) -> tuple:
        obs, infos = self.envs.reset()
        observations = {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }
        return observations, infos

    def step(self, text_actions: List[str]) -> tuple:
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_observations = {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }
        for info in infos:
            # Derive is_action_valid from parse_ok and illegal_action reported by the worker
            info["is_action_valid"] = int(info.get("parse_ok", True) and not info.get("illegal_action", False))
        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        raise NotImplementedError("Subclasses must implement build_text_obs()")

    def success_evaluator(self, total_infos=None, total_batch_list=None,
                          episode_rewards=None, episode_lengths=None, **kwargs) -> Dict[str, np.ndarray]:
        """Compute per-trajectory env metrics for logging.

        Every returned key is a length-`batch_size` array; the rollout loop averages
        it over trajectories and the trainer logs it under ``episode/<key>`` (train)
        and ``val/<key>`` (val). Keys here are namespaced ``env/...`` so the trainer's
        metric filter (``success_rate`` substring OR ``env/`` prefix) picks them up.

        Common metrics (all VPR envs):
          * ``env/success_rate``       — terminal success (win / solved / cleared).
          * ``env/valid_action_rate``  — fraction of steps with a parseable, legal action.
          * ``env/oracle_hit_rate``    — fraction of measurable moves matching the oracle
            (reads the per-step ``move_optimal`` flag; only counts steps where it is set,
            so it is meaningful regardless of reward_mode).
        Subclasses add env-specific metrics by overriding ``_trajectory_metrics``.
        """
        if total_infos is None:
            return {"env/success_rate": np.array([])}
        batch_size = len(total_infos)
        success = np.zeros(batch_size, dtype=np.float32)
        valid_rate = np.zeros(batch_size, dtype=np.float32)
        oracle_rate = np.zeros(batch_size, dtype=np.float32)
        extra: Dict[str, list] = {}
        for i, episode_info_list in enumerate(total_infos):
            for step_info in reversed(episode_info_list):
                if step_info.get("terminal_success") is not None:
                    success[i] = float(bool(step_info["terminal_success"]))
                    break
            valids = [int(bool(si.get("is_action_valid", 1))) for si in episode_info_list]
            valid_rate[i] = float(np.mean(valids)) if valids else 0.0
            opt = [bool(si["move_optimal"]) for si in episode_info_list
                   if si.get("move_optimal") is not None]
            oracle_rate[i] = float(np.mean(opt)) if opt else 0.0
            for k, v in self._trajectory_metrics(episode_info_list).items():
                extra.setdefault(k, [0.0] * batch_size)[i] = float(v)
        out: Dict[str, np.ndarray] = {
            "env/success_rate": success,
            "env/valid_action_rate": valid_rate,
            "env/oracle_hit_rate": oracle_rate,
        }
        out.update({k: np.array(v, dtype=np.float32) for k, v in extra.items()})
        return out

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        """Per-trajectory, env-specific scalar metrics. Override in subclasses.

        Returns a mapping of ``env/<name>`` → scalar (typically a rate in [0, 1]).
        """
        return {}

    def snapshot_states(self, active_indices=None) -> list:
        return self.envs.snapshot_states(active_indices=active_indices)

    def restore_states(self, snapshots: list) -> tuple:
        obs, infos = self.envs.restore_states(snapshots)
        observations = {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }
        return observations, infos

    def close(self) -> None:
        self.envs.close()
