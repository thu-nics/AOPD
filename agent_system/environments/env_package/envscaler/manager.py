"""Manager and metrics for mixed AWM plus EnvScaler semantic training."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_system.environments.env_package.awm.runtime.manager import (
    AWMEnvironmentManager,
    awm_projection,
)


class MixedAgenticEnvironmentManager(AWMEnvironmentManager):
    @staticmethod
    def _observations(infos: list[dict[str, Any]]) -> dict[str, Any]:
        families = [str(info.get("agentic_env_family") or "awm") for info in infos]
        return {
            "text": [str(info.get("observation", "")) for info in infos],
            "chat": [list(info.get("chat") or []) for info in infos],
            "tools": [list(info.get("tools") or []) for info in infos],
            "prompt_protocol": families,
            "image": None,
            "anchor": None,
        }

    def success_evaluator(
        self,
        total_infos=None,
        total_batch_list=None,
        episode_rewards=None,
        episode_lengths=None,
        **kwargs,
    ):
        output = super().success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            **kwargs,
        )
        if total_infos is None:
            return output
        families = []
        successes = np.zeros(len(total_infos), dtype=np.float32)
        success_valid = np.zeros(len(total_infos), dtype=np.float32)
        checker_fraction = np.zeros(len(total_infos), dtype=np.float32)
        conversation_success = np.zeros(len(total_infos), dtype=np.float32)
        for index, episode in enumerate(total_infos):
            family = next(
                (str(item.get("agentic_env_family")) for item in episode if item.get("agentic_env_family")),
                "awm",
            )
            families.append(family)
            terminal = [item for item in episode if item.get("terminal_success") is not None]
            if terminal:
                successes[index] = float(bool(terminal[-1]["terminal_success"]))
                success_valid[index] = 1.0
            if family == "envscaler" and episode:
                checker_fraction[index] = float(episode[-1].get("checker_fraction", 0.0) or 0.0)
                conversation_success[index] = float(bool(episode[-1].get("conversation_success", False)))
        valid = success_valid.astype(bool)
        if valid.any():
            output["env/success_rate"] = np.full(
                len(total_infos),
                float(successes[valid].mean()),
                dtype=np.float32,
            )
        family_values = np.asarray(families, dtype=object)
        for family in ("awm", "envscaler"):
            mask = family_values == family
            output[f"env/{family}/trajectory_count"] = np.asarray([mask.sum()], dtype=np.float32)
            valid_mask = mask & valid
            if valid_mask.any():
                output[f"env/{family}/success_rate"] = successes[valid_mask]
        envscaler_mask = family_values == "envscaler"
        if envscaler_mask.any():
            output["env/envscaler/checker_fraction"] = checker_fraction[envscaler_mask]
            output["env/envscaler/conversation_success_rate"] = conversation_success[envscaler_mask]
        return output


__all__ = [
    "MixedAgenticEnvironmentManager",
    "awm_projection",
]
