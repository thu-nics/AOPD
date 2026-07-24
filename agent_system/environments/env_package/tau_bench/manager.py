"""Environment manager for Tau Bench VPR and outcome rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase


def tau_projection(text_actions):
    return list(text_actions), [True] * len(text_actions)


class TauBenchEnvironmentManager(EnvironmentManagerBase):
    def reset(self, kwargs=None):
        _, infos = self.envs.reset(kwargs=kwargs)
        return self._observations(infos), infos

    @staticmethod
    def _observations(infos: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "text": [str(info.get("observation", "")) for info in infos],
            "chat": [list(info.get("chat") or []) for info in infos],
            "tools": [list(info.get("tools") or []) for info in infos],
            "image": None,
            "anchor": None,
        }

    def step(self, text_actions):
        actions, _ = self.projection_f(text_actions)
        _, rewards, dones, infos = self.envs.step(actions)
        return self._observations(infos), rewards, dones, infos

    def state_group_step(self, candidate_text_action_groups, active_indices=None):
        results = self.envs.step_candidate_groups(
            candidate_text_action_groups,
            active_indices=active_indices,
        )
        candidate_results, selected_indices, _, rewards, dones, infos = results
        return (
            candidate_results,
            selected_indices,
            self._observations(infos),
            rewards,
            dones,
            infos,
        )

    def success_evaluator(
        self,
        total_infos=None,
        total_batch_list=None,
        episode_rewards=None,
        episode_lengths=None,
        **kwargs,
    ):
        if total_infos is None:
            return {"env/success_rate": np.array([], dtype=np.float32)}
        batch_size = len(total_infos)
        domains = []
        success = np.zeros(batch_size, dtype=np.float32)
        valid_rate = np.zeros(batch_size, dtype=np.float32)
        oracle_hit_rate = np.zeros(batch_size, dtype=np.float32)
        oracle_set_size = np.zeros(batch_size, dtype=np.float32)
        protocol_reward = np.zeros(batch_size, dtype=np.float32)
        for index, episode in enumerate(total_infos):
            domains.append(next((str(info.get("tau_domain")) for info in episode if info.get("tau_domain")), "unknown"))
            terminal = [info for info in episode if info.get("terminal_success") is not None]
            if terminal:
                success[index] = float(bool(terminal[-1]["terminal_success"]))
            if episode:
                valid_rate[index] = float(
                    np.mean([float(bool(info.get("is_action_valid", 1))) for info in episode])
                )
                hits = [float(bool(info["move_optimal"])) for info in episode if info.get("move_optimal") is not None]
                oracle_hit_rate[index] = float(np.mean(hits)) if hits else 0.0
                sizes = [float(info["oracle_set_size"]) for info in episode if info.get("oracle_set_size") is not None]
                oracle_set_size[index] = float(np.mean(sizes)) if sizes else 0.0
                protocol_reward[index] = max(float(info.get("protocol_reward", 0.0)) for info in episode)
        output = {
            "env/success_rate": success,
            "env/valid_action_rate": valid_rate,
            "env/oracle_hit_rate": oracle_hit_rate,
            "env/oracle_set_size": oracle_set_size,
            "env/protocol_reward": protocol_reward,
        }
        domain_array = np.asarray(domains, dtype=object)
        for domain in ("airline", "retail"):
            mask = domain_array == domain
            output[f"env/{domain}/trajectory_count"] = np.asarray([mask.sum()], dtype=np.float32)
            if mask.any():
                output[f"env/{domain}/success_rate"] = success[mask]
                output[f"env/{domain}/valid_action_rate"] = valid_rate[mask]
                output[f"env/{domain}/oracle_hit_rate"] = oracle_hit_rate[mask]
                output[f"env/{domain}/oracle_set_size"] = oracle_set_size[mask]
                output[f"env/{domain}/protocol_reward"] = protocol_reward[mask]
        return output
