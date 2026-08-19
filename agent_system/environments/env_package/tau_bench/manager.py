"""Environment manager for Tau Bench agentic OPD and outcome rollouts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import ray

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.environments.teacher_reward import teacher_selection_diagnostics

TRANSFER_TOOL_NAME = "transfer_to_human_agents"
TRANSFER_HANDOFF_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
TRANSFER_STOP_TOKEN = "###TRANSFER###"


def _parsed_action(info: dict[str, Any]) -> dict[str, Any]:
    try:
        action = json.loads(str(info.get("parsed_action") or ""))
    except (TypeError, ValueError):
        return {}
    return action if isinstance(action, dict) else {}


def _is_transfer_tool_call(info: dict[str, Any]) -> bool:
    action = _parsed_action(info)
    return action.get("kind") == "tool" and action.get("name") == TRANSFER_TOOL_NAME


def _is_transfer_handoff(info: dict[str, Any]) -> bool:
    action = _parsed_action(info)
    return action.get("kind") == "message" and str(action.get("content") or "").strip() == TRANSFER_HANDOFF_MESSAGE


def tau_projection(text_actions):
    return list(text_actions), [True] * len(text_actions)


class TauBenchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, *, oracle_actor=None):
        super().__init__(envs, projection_f, config)
        self.oracle_actor = oracle_actor

    def reset(self, kwargs=None):
        _, infos = self.envs.reset(kwargs=kwargs)
        return self._observations(infos), infos

    @staticmethod
    def _observations(infos: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "text": [str(info.get("observation", "")) for info in infos],
            "chat": [list(info.get("chat") or []) for info in infos],
            "tools": [list(info.get("tools") or []) for info in infos],
            "prompt_protocol": ["tau"] * len(infos),
            "image": None,
            "anchor": None,
        }

    def step(self, text_actions):
        actions, _ = self.projection_f(text_actions)
        _, rewards, dones, infos = self.envs.step(actions)
        return self._observations(infos), rewards, dones, infos

    def start_teacher_preflight(self, *, active_indices, visible_chats):
        return self.envs.start_teacher_preflight(
            active_indices=active_indices,
            visible_chats=visible_chats,
        )

    def finish_teacher_preflight(self, pending):
        return self.envs.finish_teacher_preflight(pending)

    def state_group_step(
        self,
        candidate_text_action_groups,
        active_indices=None,
        visible_chats=None,
    ):
        results = self.envs.step_candidate_groups(
            candidate_text_action_groups,
            active_indices=active_indices,
            visible_chats=visible_chats,
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
        candidate_episodes = total_batch_list or [[] for _ in range(batch_size)]
        domains = []
        success = np.zeros(batch_size, dtype=np.float32)
        valid_rate = np.zeros(batch_size, dtype=np.float32)
        oracle_hit_rate = np.zeros(batch_size, dtype=np.float32)
        oracle_set_size = np.zeros(batch_size, dtype=np.float32)
        protocol_reward = np.zeros(batch_size, dtype=np.float32)
        transfer_tool_call = np.zeros(batch_size, dtype=np.float32)
        transfer_handoff = np.zeros(batch_size, dtype=np.float32)
        transfer_acknowledged = np.zeros(batch_size, dtype=np.float32)
        decision_limit = np.zeros(batch_size, dtype=np.float32)
        for index, episode in enumerate(total_infos):
            domains.append(
                next(
                    (str(info.get("tau_domain")) for info in episode if info.get("tau_domain")),
                    "unknown",
                )
            )
            terminal = [info for info in episode if info.get("terminal_success") is not None]
            if terminal:
                success[index] = float(bool(terminal[-1]["terminal_success"]))
            if episode:
                valid_rate[index] = float(np.mean([float(bool(info.get("is_action_valid", 1))) for info in episode]))
                hits = [float(bool(info["move_optimal"])) for info in episode if info.get("move_optimal") is not None]
                oracle_hit_rate[index] = float(np.mean(hits)) if hits else 0.0
                sizes = [float(info["oracle_set_size"]) for info in episode if info.get("oracle_set_size") is not None]
                oracle_set_size[index] = float(np.mean(sizes)) if sizes else 0.0
                protocol_reward[index] = max(float(info.get("protocol_reward", 0.0)) for info in episode)
                transfer_tool_call[index] = float(any(_is_transfer_tool_call(info) for info in episode))
                transfer_handoff[index] = float(any(_is_transfer_handoff(info) for info in episode))
                transfer_acknowledged[index] = float(any(TRANSFER_STOP_TOKEN in str(info.get("observation") or "") for info in episode))
                decision_limit[index] = float(any(bool(info.get("decision_limit_reached", False)) for info in episode) or str(episode[-1].get("terminal_reason") or "") == "decision_limit")
        transfer_ack_failure = transfer_handoff * (1.0 - transfer_acknowledged)
        handoff_mask = transfer_handoff.astype(bool)
        transfer_ack_success_given_handoff = transfer_acknowledged[handoff_mask] if handoff_mask.any() else np.asarray([0.0], dtype=np.float32)
        output = {
            "env/trajectory_count": np.asarray([batch_size], dtype=np.float32),
            "env/success_rate": success,
            "env/valid_action_rate": valid_rate,
            "env/oracle_hit_rate": oracle_hit_rate,
            "env/oracle_set_size": oracle_set_size,
            "env/protocol_reward": protocol_reward,
            "env/transfer_tool_call_rate": transfer_tool_call,
            "env/transfer_handoff_rate": transfer_handoff,
            "env/transfer_handoff_count": np.asarray([handoff_mask.sum()], dtype=np.float32),
            "env/transfer_acknowledged_rate": transfer_acknowledged,
            "env/transfer_ack_failure_rate": transfer_ack_failure,
            "env/transfer_ack_success_rate_given_handoff": (transfer_ack_success_given_handoff),
            "env/decision_limit_rate": decision_limit,
        }
        domain_array = np.asarray(domains, dtype=object)
        for domain in ("airline", "retail"):
            mask = domain_array == domain
            output[f"env/{domain}/trajectory_count"] = np.asarray([mask.sum()], dtype=np.float32)
            output[f"env/{domain}/trajectory_share"] = np.asarray([float(mask.mean())], dtype=np.float32)
            if mask.any():
                output[f"env/{domain}/success_rate"] = success[mask]
                output[f"env/{domain}/valid_action_rate"] = valid_rate[mask]
                output[f"env/{domain}/oracle_hit_rate"] = oracle_hit_rate[mask]
                output[f"env/{domain}/oracle_set_size"] = oracle_set_size[mask]
                output[f"env/{domain}/protocol_reward"] = protocol_reward[mask]
                output[f"env/{domain}/transfer_tool_call_rate"] = transfer_tool_call[mask]
                output[f"env/{domain}/transfer_handoff_rate"] = transfer_handoff[mask]
                output[f"env/{domain}/transfer_acknowledged_rate"] = transfer_acknowledged[mask]
                output[f"env/{domain}/transfer_ack_failure_rate"] = transfer_ack_failure[mask]
                domain_handoff_mask = mask & handoff_mask
                output[f"env/{domain}/transfer_handoff_count"] = np.asarray([domain_handoff_mask.sum()], dtype=np.float32)
                output[f"env/{domain}/transfer_ack_success_rate_given_handoff"] = transfer_acknowledged[domain_handoff_mask] if domain_handoff_mask.any() else np.asarray([0.0], dtype=np.float32)
                output[f"env/{domain}/decision_limit_rate"] = decision_limit[mask]
        output.update({f"env/{name}": np.asarray([value], dtype=np.float32) for name, value in teacher_selection_diagnostics(candidate_episodes, total_infos).items()})
        if self.oracle_actor is not None:
            stats = ray.get(self.oracle_actor.get_stats.remote())
            for name, value in stats.items():
                output[f"env/oracle_{name}"] = np.full(batch_size, float(value), dtype=np.float64)
        return output
