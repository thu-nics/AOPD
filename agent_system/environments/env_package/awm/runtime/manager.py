"""Environment manager for AWM agentic OPD and outcome rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import ray

from agent_system.environments.base import EnvironmentManagerBase
from agent_system.environments.teacher_reward import teacher_selection_diagnostics


def awm_projection(text_actions):
    return list(text_actions), [True] * len(text_actions)


class AWMEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, *, oracle_actor=None):
        super().__init__(envs, projection_f, config)
        self.oracle_actor = oracle_actor

    def reset(self, kwargs=None):
        rows = [] if kwargs is None else list(kwargs)
        schedule_steps = {int(row["schedule_step"]) for row in rows if row.get("schedule_step") is not None}
        if len(schedule_steps) > 1:
            raise ValueError("one AWM training batch must contain exactly one schedule_step")
        schedule_step = next(iter(schedule_steps), None)
        if schedule_step is not None:
            schedule_slots = [row.get("schedule_slot") for row in rows]
            if schedule_slots != list(range(len(schedule_slots))):
                raise RuntimeError("AWM task-level resume requires ordered zero-based schedule_slot values")
        current_step = int(getattr(getattr(self.config.env, "rollout", None), "current_step", 0) or 0)
        if schedule_step is not None:
            expected = max(current_step - 1, 0)
            if schedule_step != expected:
                raise RuntimeError(f"AWM task-level resume mismatch: global step {current_step} requires schedule_step={expected}, received {schedule_step}")
        _, infos = self.envs.reset(
            kwargs=rows,
            schedule_step=schedule_step,
        )
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

    def start_teacher_preflight(self, *, active_indices, visible_chats):
        return self.envs.start_teacher_preflight(
            active_indices=active_indices,
            visible_chats=visible_chats,
        )

    def finish_teacher_preflight(self, pending):
        return self.envs.finish_teacher_preflight(pending)

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
        group_metadata=None,
    ):
        results = self.envs.step_candidate_groups(
            candidate_text_action_groups,
            active_indices=active_indices,
            visible_chats=visible_chats,
            group_metadata=group_metadata,
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
        success_all = np.zeros(batch_size, dtype=np.float32)
        terminal_judge_coverage = np.zeros(batch_size, dtype=np.float32)
        terminal_reward = np.zeros(batch_size, dtype=np.float32)
        terminal_complete = np.zeros(batch_size, dtype=np.float32)
        terminal_incomplete = np.zeros(batch_size, dtype=np.float32)
        terminal_agent_error = np.zeros(batch_size, dtype=np.float32)
        terminal_server_error = np.zeros(batch_size, dtype=np.float32)
        terminal_judge_error = np.zeros(batch_size, dtype=np.float32)
        terminal_other_error = np.zeros(batch_size, dtype=np.float32)
        valid_rate = np.zeros(batch_size, dtype=np.float32)
        teacher_reward = np.zeros(batch_size, dtype=np.float32)
        masked_rate = np.zeros(batch_size, dtype=np.float32)
        protocol_reward = np.zeros(batch_size, dtype=np.float32)
        matcher_failure_rate = np.zeros(batch_size, dtype=np.float32)
        frequency_sensitive_rate = np.zeros(batch_size, dtype=np.float32)
        action_kind_disagreement_rate = np.zeros(batch_size, dtype=np.float32)
        runtime_failure = np.zeros(batch_size, dtype=np.float32)
        runtime_policy_error = np.zeros(batch_size, dtype=np.float32)
        runtime_policy_continued = np.zeros(batch_size, dtype=np.float32)
        runtime_policy_terminated = np.zeros(batch_size, dtype=np.float32)
        context_overflow = np.zeros(batch_size, dtype=np.float32)
        context_overflow_prompt_tokens = np.zeros(batch_size, dtype=np.float32)
        context_overflow_excess_tokens = np.zeros(batch_size, dtype=np.float32)
        for index, episode in enumerate(total_infos):
            rows = candidate_episodes[index] if index < len(candidate_episodes) else []
            terminal = [info for info in episode if info.get("terminal_label") is not None]
            if terminal:
                terminal_info = terminal[-1]
                label = str(terminal_info.get("terminal_label") or "")
                is_valid = bool(terminal_info.get("terminal_outcome_valid", False))
                terminal_judge_coverage[index] = float(is_valid)
                success_all[index] = float(label == "complete")
                terminal_complete[index] = float(label == "complete")
                terminal_incomplete[index] = float(label == "incomplete")
                terminal_agent_error[index] = float(label == "agent_error")
                terminal_server_error[index] = float(label == "server_error")
                terminal_judge_error[index] = float(label == "judge_error")
                terminal_other_error[index] = float(
                    label
                    not in {
                        "complete",
                        "incomplete",
                        "agent_error",
                        "server_error",
                        "judge_error",
                    }
                )
                if is_valid:
                    terminal_reward[index] = float(terminal_info.get("terminal_reward", 0.0) or 0.0)
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
                matcher_failure_rate[index] = float(np.mean([float(bool(info.get("matcher_failure", False))) for info in episode]))
                frequency_sensitive_rate[index] = float(np.mean([float(bool(info.get("frequency_sensitive_group", False))) for info in episode if not info.get("teacher_failure", False)] or [0.0]))
                action_kind_disagreement_rate[index] = float(np.mean([float(bool(info.get("teacher_action_kind_disagreement", False))) for info in episode if not info.get("teacher_failure", False)] or [0.0]))
                runtime_failure[index] = float(any(bool(info.get("runtime_failure", False)) for info in episode))
                runtime_policy_error[index] = float(any(bool(info.get("runtime_policy_error", False)) for info in episode))
                runtime_policy_continued[index] = float(any(bool(info.get("runtime_policy_continued", False)) for info in episode))
                runtime_policy_terminated[index] = float(any(bool(info.get("runtime_policy_terminated", False)) for info in episode))
                overflow_infos = [info for info in episode if bool(info.get("context_overflow", False))]
                if overflow_infos:
                    context_overflow[index] = 1.0
                    context_overflow_prompt_tokens[index] = max(float(info.get("context_prompt_tokens", 0) or 0) for info in overflow_infos)
                    context_overflow_excess_tokens[index] = max(float(info.get("context_excess_tokens", 0) or 0) for info in overflow_infos)
        valid_terminal_count = float(np.sum(terminal_judge_coverage))
        if valid_terminal_count:
            success.fill(float(np.sum(terminal_complete) / valid_terminal_count))
            terminal_reward.fill(float(np.sum(terminal_reward) / valid_terminal_count))
        overflow_count = float(np.sum(context_overflow))
        if overflow_count:
            context_overflow_prompt_tokens.fill(float(np.sum(context_overflow_prompt_tokens) / overflow_count))
            context_overflow_excess_tokens.fill(float(np.sum(context_overflow_excess_tokens) / overflow_count))
        teacher_states = [info for episode in total_infos for info in episode if "teacher_sample_count" in info]
        teacher_failure_state_rate = sum(bool(info.get("teacher_failure", False)) for info in teacher_states) / len(teacher_states) if teacher_states else 0.0
        metrics = {
            "env/trajectory_count": np.asarray([batch_size], dtype=np.float32),
            "env/terminal_outcome_count": np.asarray([valid_terminal_count], dtype=np.float32),
            "env/success_rate": success,
            "env/success_rate_all": success_all,
            "env/terminal_judge_coverage": terminal_judge_coverage,
            "env/terminal_reward_mean": terminal_reward,
            "env/terminal_complete_rate": terminal_complete,
            "env/terminal_incomplete_rate": terminal_incomplete,
            "env/terminal_agent_error_rate": terminal_agent_error,
            "env/terminal_server_error_rate": terminal_server_error,
            "env/terminal_judge_error_rate": terminal_judge_error,
            "env/terminal_other_error_rate": terminal_other_error,
            "env/valid_action_rate": valid_rate,
            "env/teacher_frequency": teacher_reward,
            "env/semantic_masked_rate": masked_rate,
            "env/protocol_reward": protocol_reward,
            "env/teacher_failure_state_rate": np.asarray([teacher_failure_state_rate], dtype=np.float32),
            "env/matcher_failure_rate": matcher_failure_rate,
            "env/frequency_sensitive_group_rate": frequency_sensitive_rate,
            "env/teacher_action_kind_disagreement_rate": (action_kind_disagreement_rate),
            "env/runtime_failure_rate": runtime_failure,
            "env/context_overflow_rate": context_overflow,
            "env/runtime_policy_error_rate": runtime_policy_error,
            "env/runtime_policy_continued_rate": runtime_policy_continued,
            "env/runtime_policy_terminated_rate": runtime_policy_terminated,
            "env/context_overflow_prompt_tokens_mean": context_overflow_prompt_tokens,
            "env/context_overflow_excess_tokens_mean": context_overflow_excess_tokens,
        }
        metrics.update({f"env/{name}": np.asarray([value], dtype=np.float32) for name, value in teacher_selection_diagnostics(candidate_episodes, total_infos).items()})
        if self.oracle_actor is not None:
            stats = ray.get(self.oracle_actor.get_stats.remote())
            for name, value in stats.items():
                metrics[f"env/oracle_{name}"] = np.full(batch_size, float(value), dtype=np.float64)
        return metrics
