"""Heterogeneous AWM and EnvScaler semantic vector environment."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import ray

from agent_system.environments.env_package.awm.runtime.envs import (
    AWMRuntimeFailureRecorder,
    AWMWorker,
)
from agent_system.environments.rollout_progress import validate_no_progress_config

from .runtime import EnvScalerWorker

FAMILY_ORDER = ("awm", "envscaler")


def interleave_families(counts: Mapping[str, int]) -> list[str]:
    normalized = {family: int(counts.get(family, 0)) for family in FAMILY_ORDER}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("mixed agentic family counts must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("mixed agentic family counts must be positive")
    used = {family: 0 for family in FAMILY_ORDER}
    labels = []
    for slot in range(total):
        candidates = [family for family in FAMILY_ORDER if used[family] < normalized[family]]
        selected = max(
            candidates,
            key=lambda family: (
                normalized[family] * (slot + 1) / total - used[family],
                -FAMILY_ORDER.index(family),
            ),
        )
        labels.append(selected)
        used[selected] += 1
    return labels


class MixedAgenticVectorEnv:
    def __init__(self, workers, seeds, families, runtime_recorder=None):
        if not (len(workers) == len(seeds) == len(families)):
            raise ValueError("mixed agentic workers, seeds, and families must align")
        self.workers = workers
        self.seeds = seeds
        self.families = list(families)
        self.runtime_recorder = runtime_recorder
        self._episode = 0

    def _family_for_row(self, row: Mapping[str, Any]) -> str:
        explicit = str(row.get("env_family") or "").lower()
        if explicit:
            return explicit
        if row.get("scenario") and "task_idx" in row:
            return "awm"
        if "task_index" in row:
            return "envscaler"
        return ""

    def reset(self, kwargs=None, schedule_step=None):
        if kwargs is None or len(kwargs) != len(self.workers):
            raise ValueError(f"expected {len(self.workers)} mixed agentic env kwargs")
        episode = self._episode if schedule_step is None else int(schedule_step)
        if episode < 0:
            raise ValueError("mixed-agentic schedule_step must be non-negative")
        offset = episode * 100003
        self._episode += 1
        futures = []
        for worker, seed, family, row in zip(self.workers, self.seeds, self.families, kwargs, strict=True):
            row_family = self._family_for_row(row)
            if row_family != family:
                raise ValueError(f"mixed slot expects {family}, received {row_family or 'unknown'}")
            if family == "awm":
                futures.append(
                    worker.reset.remote(
                        scenario=str(row["scenario"]),
                        task_idx=int(row["task_idx"]),
                        seed=seed + offset,
                    )
                )
            else:
                futures.append(
                    worker.reset.remote(
                        task_index=int(row["task_index"]),
                        seed=seed + offset,
                    )
                )
        results = ray.get(futures)
        return [item[0] for item in results], [item[1] for item in results]

    def step(self, actions):
        results = ray.get([worker.step.remote(action) for worker, action in zip(self.workers, actions, strict=True)])
        return (
            [item[0] for item in results],
            np.asarray([item[1] for item in results], dtype=np.float32),
            np.asarray([item[2] for item in results], dtype=bool),
            [item[3] for item in results],
        )

    def start_teacher_preflight(self, *, active_indices, visible_chats):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(visible_chats):
            raise ValueError("active indices and visible chats must align")
        return [
            self.workers[index].prepare_teacher_supervision.remote(chat)
            for index, chat in zip(indices, visible_chats, strict=True)
        ]

    @staticmethod
    def finish_teacher_preflight(pending):
        return ray.get(pending)

    def terminate_context_overflows(self, *, active_indices, diagnostics):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(diagnostics):
            raise ValueError("active indices and context diagnostics must align")
        return ray.get([self.workers[index].terminate_context_overflow.remote(item) for index, item in zip(indices, diagnostics, strict=True)])

    def inspect_no_progress_candidate_groups(
        self, candidate_action_groups, active_indices=None
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        indices = [int(index) for index in active_indices]
        if len(indices) != len(candidate_action_groups):
            raise ValueError("active indices and candidate groups must align")
        return ray.get(
            [
                self.workers[index].inspect_no_progress_resample.remote(group)
                for index, group in zip(indices, candidate_action_groups, strict=True)
            ]
        )

    def step_candidate_groups(
        self,
        candidate_action_groups,
        active_indices=None,
        visible_chats=None,
        group_metadata=None,
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        indices = [int(index) for index in active_indices]
        if len(indices) != len(candidate_action_groups):
            raise ValueError("active indices and candidate groups must align")
        if visible_chats is None:
            visible_chats = [None] * len(indices)
        if len(visible_chats) != len(indices):
            raise ValueError("visible chats and candidate groups must align")
        if group_metadata is None:
            group_metadata = [None] * len(indices)
        if len(group_metadata) != len(indices):
            raise ValueError("group metadata and candidate groups must align")
        results = ray.get(
            [
                self.workers[index].step_candidate_group.remote(
                    group, visible_chat=chat, group_metadata=metadata
                )
                for index, group, chat, metadata in zip(
                    indices,
                    candidate_action_groups,
                    visible_chats,
                    group_metadata,
                    strict=True,
                )
            ]
        )
        return (
            [item[0] for item in results],
            np.asarray([item[1] for item in results], dtype=np.int32),
            [item[2] for item in results],
            np.asarray([item[3] for item in results], dtype=np.float32),
            np.asarray([item[4] for item in results], dtype=bool),
            [item[5] for item in results],
        )

    def close(self):
        ray.get([worker.close.remote() for worker in self.workers])
        for worker in self.workers:
            ray.kill(worker)
        if self.runtime_recorder is not None:
            ray.kill(self.runtime_recorder)


def build_mixed_agentic_envs(
    *,
    seed: int,
    counts: Mapping[str, int],
    env_config,
    oracle_actor,
):
    labels = interleave_families(counts)
    worker_options = dict(getattr(env_config, "resources_per_worker", {}) or {})
    awm_factory = AWMWorker.options(**worker_options) if worker_options else AWMWorker
    envscaler_factory = EnvScalerWorker.options(**worker_options) if worker_options else EnvScalerWorker
    awm = env_config.awm
    teacher_reward = env_config.teacher_reward
    rollout_config = env_config.rollout
    no_progress_config = rollout_config.no_progress_resample
    no_progress_enabled, _, no_progress_min_streak = validate_no_progress_config(
        enabled=no_progress_config.enabled,
        max_rounds=no_progress_config.max_rounds,
        min_repeat_streak=no_progress_config.min_repeat_streak,
    )
    runtime_config = awm.runtime_failures
    runtime_judge = runtime_config.judge
    terminal = awm.terminal_judge
    runtime_recorder = AWMRuntimeFailureRecorder.remote(str(runtime_config.path))
    workers = []
    seeds = []
    for index, family in enumerate(labels):
        worker_seed = int(seed) + index
        if family == "awm":
            worker = awm_factory.remote(
                base_url=str(awm.base_url),
                max_steps=int(awm.train_max_steps),
                max_history_exchanges=(int(env_config.context.max_history_exchanges) if env_config.context.max_history_exchanges is not None else None),
                verifier_mode=str(awm.verifier_mode),
                reward_mode="semantic",
                oracle_actor=oracle_actor,
                runtime_recorder=runtime_recorder,
                runtime_judge_enabled=bool(runtime_config.enabled and runtime_judge.enabled),
                runtime_judge_confidence_threshold=int(runtime_judge.confidence_threshold),
                frequency_bonus_scale=float(teacher_reward.frequency_bonus_scale),
                teacher_reward_mode=str(teacher_reward.mode),
                prefer_nonrepeat_argmax=bool(rollout_config.prefer_nonrepeat_argmax),
                no_progress_resample_enabled=bool(
                    no_progress_enabled
                ),
                no_progress_resample_min_streak=int(
                    no_progress_min_streak
                ),
                use_privileged_teacher_context=bool(
                    getattr(awm.oracle, "use_privileged_context", False)
                ),
                terminal_judge_api_base=str(terminal.api_base),
                terminal_judge_api_key_env=str(terminal.api_key_env),
                terminal_judge_model=str(terminal.model),
                seed=worker_seed,
            )
        else:
            config = env_config.envscaler
            worker = envscaler_factory.remote(
                source_root=str(config.source_root),
                max_steps=int(config.train_max_steps),
                oracle_actor=oracle_actor,
                user_model=str(config.user_simulator.model),
                user_api_base=str(config.user_simulator.api_base),
                user_api_key_env=str(config.user_simulator.api_key_env),
                user_temperature=float(config.user_simulator.temperature),
                user_reasoning_enabled=bool(config.user_simulator.reasoning_enabled),
                user_timeout_seconds=float(config.user_simulator.timeout_seconds),
                user_max_retries=int(config.user_simulator.max_retries),
                frequency_bonus_scale=float(teacher_reward.frequency_bonus_scale),
                teacher_reward_mode=str(teacher_reward.mode),
                prefer_nonrepeat_argmax=bool(rollout_config.prefer_nonrepeat_argmax),
                no_progress_resample_enabled=bool(
                    no_progress_enabled
                ),
                no_progress_resample_min_streak=int(
                    no_progress_min_streak
                ),
                use_privileged_teacher_context=bool(
                    getattr(config.oracle, "use_privileged_context", False)
                ),
                seed=worker_seed,
            )
        workers.append(worker)
        seeds.append(worker_seed)
    return MixedAgenticVectorEnv(
        workers,
        seeds,
        labels,
        runtime_recorder=runtime_recorder,
    )
