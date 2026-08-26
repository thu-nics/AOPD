"""Manager and metrics for mixed AWM plus EnvScaler agentic OPD training."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_system.environments.env_package.awm.runtime.manager import (
    AWMEnvironmentManager,
    awm_projection,
)
from agent_system.environments.teacher_reward import (
    teacher_selection_diagnostics,
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
        user_stop = np.zeros(len(total_infos), dtype=np.float32)
        decision_limit = np.zeros(len(total_infos), dtype=np.float32)
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
                terminal_reason = str(episode[-1].get("terminal_reason") or "")
                user_stop[index] = float(terminal_reason == "user_stop")
                decision_limit[index] = float(terminal_reason == "decision_limit")
        valid = success_valid.astype(bool)
        output["env/trajectory_count"] = np.asarray([len(total_infos)], dtype=np.float32)
        output["env/terminal_outcome_count"] = np.asarray([valid.sum()], dtype=np.float32)
        output["env/terminal_outcome_coverage"] = success_valid
        output["env/success_rate_all"] = successes
        if valid.any():
            output["env/success_rate"] = np.full(
                len(total_infos),
                float(successes[valid].mean()),
                dtype=np.float32,
            )
        family_values = np.asarray(families, dtype=object)
        candidate_episodes = total_batch_list or [[] for _ in total_infos]
        for family in ("awm", "envscaler"):
            mask = family_values == family
            indices = np.flatnonzero(mask).tolist()
            family_candidates = [candidate_episodes[index] for index in indices]
            family_selected = [total_infos[index] for index in indices]
            family_diagnostics = teacher_selection_diagnostics(family_candidates, family_selected)
            for name, value in family_diagnostics.items():
                output[f"env/{family}/{name}"] = np.asarray([value], dtype=np.float32)
            output[f"env/{family}/trajectory_count"] = np.asarray([mask.sum()], dtype=np.float32)
            output[f"env/{family}/trajectory_share"] = np.asarray([float(mask.mean())], dtype=np.float32)
            valid_mask = mask & valid
            output[f"env/{family}/terminal_outcome_count"] = np.asarray([valid_mask.sum()], dtype=np.float32)
            if valid_mask.any():
                output[f"env/{family}/success_rate"] = successes[valid_mask]
            if mask.any():
                output[f"env/{family}/success_rate_all"] = successes[mask]
                output[f"env/{family}/terminal_outcome_coverage"] = success_valid[mask]
                for metric_name in (
                    "valid_action_rate",
                    "runtime_failure_rate",
                    "runtime_policy_error_rate",
                    "runtime_policy_continued_rate",
                    "runtime_policy_terminated_rate",
                ):
                    parent_values = np.asarray(output[f"env/{metric_name}"])
                    if len(parent_values) == len(total_infos):
                        output[f"env/{family}/{metric_name}"] = parent_values[mask]
        awm_mask = family_values == "awm"
        if awm_mask.any():
            output["env/terminal_judge_coverage"] = success_valid[awm_mask]
            output["env/awm/terminal_judge_coverage"] = success_valid[awm_mask]
        envscaler_mask = family_values == "envscaler"
        if envscaler_mask.any():
            output["env/envscaler/checker_fraction"] = checker_fraction[envscaler_mask]
            output["env/envscaler/conversation_success_rate"] = conversation_success[envscaler_mask]
            output["env/envscaler/user_stop_rate"] = user_stop[envscaler_mask]
            output["env/envscaler/decision_limit_rate"] = decision_limit[envscaler_mask]
        return output


__all__ = [
    "MixedAgenticEnvironmentManager",
    "awm_projection",
]
