"""Environment manager for AWM semantic and outcome rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import ray

from agent_system.environments.base import EnvironmentManagerBase


def awm_projection(text_actions):
    return list(text_actions), [True] * len(text_actions)


class AWMEnvironmentManager(EnvironmentManagerBase):
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
            "prompt_protocol": ["awm"] * len(infos),
            "image": None,
            "anchor": None,
        }

    def step(self, text_actions):
        actions, _ = self.projection_f(text_actions)
        _, rewards, dones, infos = self.envs.step(actions)
        return self._observations(infos), rewards, dones, infos

    def prepare_state_groups(self, *, active_indices, visible_chats):
        return self.envs.prepare_state_groups(
            active_indices=active_indices,
            visible_chats=visible_chats,
        )

    def terminate_context_overflows(self, *, active_indices, diagnostics):
        return self.envs.terminate_context_overflows(
            active_indices=active_indices,
            diagnostics=diagnostics,
        )

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
        success = np.zeros(batch_size, dtype=np.float32)
        valid_rate = np.zeros(batch_size, dtype=np.float32)
        teacher_reward = np.zeros(batch_size, dtype=np.float32)
        masked_rate = np.zeros(batch_size, dtype=np.float32)
        protocol_reward = np.zeros(batch_size, dtype=np.float32)
        teacher_failure_rate = np.zeros(batch_size, dtype=np.float32)
        matcher_failure_rate = np.zeros(batch_size, dtype=np.float32)
        teacher_invalid_rate = np.zeros(batch_size, dtype=np.float32)
        frequency_sensitive_rate = np.zeros(batch_size, dtype=np.float32)
        action_kind_disagreement_rate = np.zeros(batch_size, dtype=np.float32)
        runtime_failure = np.zeros(batch_size, dtype=np.float32)
        runtime_failure_confirmed = np.zeros(batch_size, dtype=np.float32)
        runtime_infrastructure_pending = np.zeros(batch_size, dtype=np.float32)
        context_overflow = np.zeros(batch_size, dtype=np.float32)
        context_overflow_prompt_tokens = np.zeros(batch_size, dtype=np.float32)
        context_overflow_excess_tokens = np.zeros(batch_size, dtype=np.float32)
        for index, episode in enumerate(total_infos):
            rows = candidate_episodes[index] if index < len(candidate_episodes) else []
            terminal = [info for info in episode if info.get("terminal_success") is not None]
            if terminal:
                success[index] = float(bool(terminal[-1]["terminal_success"]))
            if rows:
                valid_rate[index] = float(np.mean([float(bool(row.get("is_action_valid", 1))) for row in rows]))
                frequencies = [float(row["teacher_frequency"]) for row in rows if row.get("teacher_frequency") is not None]
                teacher_reward[index] = float(np.mean(frequencies)) if frequencies else 0.0
                masks = [float(bool(row.get("semantic_train_mask", True))) for row in rows]
                masked_rate[index] = 1.0 - float(np.mean(masks))
            elif episode:
                # Vanilla/outcome rollouts have one executed row per item. This
                # fallback also keeps the manager useful in focused unit tests.
                valid_actions = [
                    info
                    for info in episode
                    if info.get("action_kind")
                    not in {
                        "teacher_failure",
                        "matcher_failure",
                        "context_overflow",
                    }
                ]
                if valid_actions:
                    valid_rate[index] = float(np.mean([float(bool(info.get("is_action_valid", 1))) for info in valid_actions]))
                    frequencies = [float(info["teacher_frequency"]) for info in valid_actions if info.get("teacher_frequency") is not None]
                    teacher_reward[index] = float(np.mean(frequencies)) if frequencies else 0.0
                    masks = [float(bool(info.get("semantic_train_mask", True))) for info in valid_actions]
                    masked_rate[index] = 1.0 - float(np.mean(masks))
            if episode:
                protocol_reward[index] = max(float(info.get("protocol_reward", 0.0)) for info in episode)
                teacher_failure_rate[index] = float(np.mean([float(bool(info.get("teacher_failure", False))) for info in episode]))
                matcher_failure_rate[index] = float(np.mean([float(bool(info.get("matcher_failure", False))) for info in episode]))
                invalid_samples = sum(int(info.get("teacher_invalid_sample_count", 0)) for info in episode)
                teacher_samples = sum(int(info.get("teacher_sample_count", 0)) for info in episode)
                teacher_invalid_rate[index] = invalid_samples / max(teacher_samples, 1)
                frequency_sensitive_rate[index] = float(np.mean([float(bool(info.get("frequency_sensitive_group", False))) for info in episode if not info.get("teacher_failure", False)] or [0.0]))
                action_kind_disagreement_rate[index] = float(np.mean([float(bool(info.get("teacher_action_kind_disagreement", False))) for info in episode if not info.get("teacher_failure", False)] or [0.0]))
                runtime_failure[index] = float(any(bool(info.get("runtime_failure", False)) for info in episode))
                runtime_failure_confirmed[index] = float(any(bool(info.get("runtime_failure_confirmed", False)) for info in episode))
                runtime_infrastructure_pending[index] = float(any(bool(info.get("runtime_infrastructure_pending", False)) for info in episode))
                overflow_infos = [info for info in episode if bool(info.get("context_overflow", False))]
                if overflow_infos:
                    context_overflow[index] = 1.0
                    context_overflow_prompt_tokens[index] = max(float(info.get("context_prompt_tokens", 0) or 0) for info in overflow_infos)
                    context_overflow_excess_tokens[index] = max(float(info.get("context_excess_tokens", 0) or 0) for info in overflow_infos)
        overflow_count = float(np.sum(context_overflow))
        if overflow_count:
            context_overflow_prompt_tokens.fill(float(np.sum(context_overflow_prompt_tokens) / overflow_count))
            context_overflow_excess_tokens.fill(float(np.sum(context_overflow_excess_tokens) / overflow_count))
        metrics = {
            "env/success_rate": success,
            "env/valid_action_rate": valid_rate,
            "env/teacher_frequency": teacher_reward,
            "env/semantic_masked_rate": masked_rate,
            "env/protocol_reward": protocol_reward,
            "env/teacher_failure_rate": teacher_failure_rate,
            "env/matcher_failure_rate": matcher_failure_rate,
            "env/teacher_invalid_sample_rate": teacher_invalid_rate,
            "env/frequency_sensitive_group_rate": frequency_sensitive_rate,
            "env/teacher_action_kind_disagreement_rate": (action_kind_disagreement_rate),
            "env/runtime_failure_rate": runtime_failure,
            "env/runtime_failure_confirmed_rate": runtime_failure_confirmed,
            "env/runtime_infrastructure_pending_rate": runtime_infrastructure_pending,
            "env/context_overflow_rate": context_overflow,
            "env/context_overflow_prompt_tokens_mean": context_overflow_prompt_tokens,
            "env/context_overflow_excess_tokens_mean": context_overflow_excess_tokens,
        }
        if self.oracle_actor is not None:
            stats = ray.get(self.oracle_actor.get_stats.remote())
            for name, value in stats.items():
                metrics[f"env/oracle_{name}"] = np.full(batch_size, float(value), dtype=np.float64)
        return metrics
