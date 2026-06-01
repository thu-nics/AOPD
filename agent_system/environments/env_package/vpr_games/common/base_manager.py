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
        if total_infos is None:
            return {"success": np.array([])}
        batch_size = len(total_infos)
        success = np.zeros(batch_size, dtype=bool)
        for i, episode_info_list in enumerate(total_infos):
            for step_info in reversed(episode_info_list):
                if step_info.get("terminal_success") is not None:
                    success[i] = bool(step_info["terminal_success"])
                    break
        return {"success": success}

    def close(self) -> None:
        self.envs.close()
