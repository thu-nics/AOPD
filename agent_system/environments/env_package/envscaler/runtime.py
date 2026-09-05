"""Conversation semantic runtime for pinned EnvScaler RL tasks."""

from __future__ import annotations

import asyncio
import json
import random
import traceback
from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

import ray

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    append_exchange,
    canonical_action,
    openai_tools,
    parse_action,
    score_candidates,
    state_fingerprint,
    tool_schema_hash,
    validate_action,
)
from agent_system.environments.env_package.awm.runtime.envs import (
    frequency_sensitive_group,
    validate_teacher_multiset,
)
from agent_system.environments.env_package.awm.runtime.oracle import (
    build_teacher_messages,
)
from agent_system.environments.prompts.agentic_opd import (
    ENVSCALER_PROMPT_PROTOCOL,
    envscaler_system_prompt,
    prompt_hash,
)
from agent_system.environments.rollout_progress import (
    NoProgressTracker,
    select_history_aware_with_appearance_counterfactual,
    validate_progress_config,
)
from agent_system.environments.teacher_reward import (
    DEFAULT_FREQUENCY_BONUS_SCALE,
    DEFAULT_TEACHER_REWARD_MODE,
    validate_teacher_reward_config,
)

from .runtime_judge import build_envscaler_runtime_judge_evidence
from .source import (
    DEFAULT_SOURCE_ROOT,
    build_environment_instance,
    checker_summary,
    evaluate_checkers,
    load_envscaler_source,
    restore_state,
    state_dict,
    validate_tool_contract,
)
from .user_simulator import STOP, ProviderUserSimulator

ENVSCALER_PROTOCOL_VERSION = 9


def agent_system_prompt(environment: Mapping[str, Any]) -> str:
    return envscaler_system_prompt(environment)


@ray.remote(max_concurrency=8)
class EnvScalerWorker:
    def __init__(
        self,
        *,
        source_root: str = str(DEFAULT_SOURCE_ROOT),
        max_steps: int = 40,
        oracle_actor=None,
        user_provider: str = "deepseek",
        user_model: str = "deepseek-v4-flash",
        user_api_base: str = "https://api.deepseek.com",
        user_api_key_env: str = "DEEPSEEK_API_KEY",
        user_temperature: float = 1.0,
        user_reasoning_enabled: bool | None = None,
        user_timeout_seconds: float = 300,
        user_max_retries: int = 3,
        runtime_judge_enabled: bool = False,
        runtime_judge_confidence_threshold: int = 80,
        frequency_bonus_scale: float = DEFAULT_FREQUENCY_BONUS_SCALE,
        teacher_reward_mode: str = DEFAULT_TEACHER_REWARD_MODE,
        use_privileged_teacher_context: bool = False,
        prefer_nonrepeat_argmax: bool = False,
        repeat_reward_cap_enabled: bool = False,
        repeat_reward_cap_min_streak: int = 3,
        repeat_reward_cap_value: float = 0.0,
        repeat_termination_enabled: bool = False,
        repeat_termination_max_streak: int = 4,
        seed: int = 0,
    ):
        self.source_root = str(source_root)
        self.max_steps = int(max_steps)
        self.oracle_actor = oracle_actor
        self.user_config = {
            "provider": str(user_provider),
            "model": str(user_model),
            "api_base": str(user_api_base),
            "api_key_env": str(user_api_key_env),
            "temperature": float(user_temperature),
            "reasoning_enabled": (None if user_reasoning_enabled is None else bool(user_reasoning_enabled)),
            "timeout_seconds": float(user_timeout_seconds),
            "max_retries": int(user_max_retries),
        }
        self.seed = int(seed)
        self.runtime_judge_enabled = bool(runtime_judge_enabled)
        self.runtime_judge_confidence_threshold = int(runtime_judge_confidence_threshold)
        if not 0 <= self.runtime_judge_confidence_threshold <= 100:
            raise ValueError("EnvScaler runtime judge confidence threshold must be in [0, 100]")
        self.teacher_reward_mode, self.frequency_bonus_scale = validate_teacher_reward_config(
            teacher_reward_mode,
            frequency_bonus_scale,
        )
        self.use_privileged_teacher_context = bool(use_privileged_teacher_context)
        self.prefer_nonrepeat_argmax = bool(prefer_nonrepeat_argmax)
        (
            self.repeat_reward_cap_enabled,
            self.repeat_reward_cap_min_streak,
            self.repeat_reward_cap_value,
            self.repeat_termination_enabled,
            self.repeat_termination_max_streak,
        ) = validate_progress_config(
            repeat_reward_cap_enabled=repeat_reward_cap_enabled,
            repeat_reward_cap_min_streak=repeat_reward_cap_min_streak,
            repeat_reward_cap_value=repeat_reward_cap_value,
            repeat_termination_enabled=repeat_termination_enabled,
            repeat_termination_max_streak=repeat_termination_max_streak,
        )
        self._no_progress = NoProgressTracker()
        self._last_selected_canonical_action: str | None = None
        self._rng = random.Random(seed)
        self._source = None
        self._runtime = None
        self._task: dict[str, Any] = {}
        self._environment: dict[str, Any] = {}
        self._initial_state: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []
        self._chat: list[dict[str, Any]] = []
        self._simulator: ProviderUserSimulator | None = None
        self._task_index = -1
        self._step = 0
        self._done = False
        self._last_observation = ""
        self._last_info: dict[str, Any] = {}
        self._prepared_teacher_supervision: dict[str, Any] | None = None
        self._reset_failure: str | None = None

    def _checks(self) -> dict[str, Any]:
        results = evaluate_checkers(
            self._task,
            self._initial_state,
            state_dict(self._runtime),
        )
        return {**checker_summary(results), "checker_results": results}

    def _annotate(self, **updates: Any) -> dict[str, Any]:
        task_id = str(self._task.get("task_id") or "")
        summary = (
            self._checks()
            if self._runtime is not None
            else {
                "checker_count": 0,
                "checker_passed": 0,
                "checker_fraction": 0.0,
                "checker_errors": [],
                "checker_results": [],
                "state_complete": False,
            }
        )
        info = {
            "envscaler_protocol_version": ENVSCALER_PROTOCOL_VERSION,
            "agent_prompt_protocol": ENVSCALER_PROMPT_PROTOCOL,
            "agent_prompt_hash": (prompt_hash(str(self._chat[0].get("content") or "")) if self._chat else ""),
            "teacher_reward_mode": self.teacher_reward_mode,
            "frequency_bonus_scale": self.frequency_bonus_scale,
            "teacher_context_mode": ("privileged" if self.use_privileged_teacher_context else "student_visible"),
            "agentic_env_family": "envscaler",
            "envscaler_task_id": task_id,
            "envscaler_task_index": self._task_index,
            "envscaler_env_id": str(self._task.get("env_id") or ""),
            "vpr_game": "envscaler",
            "step": self._step,
            "max_steps": self.max_steps,
            "observation": self._last_observation,
            "chat": deepcopy(self._chat),
            "tools": openai_tools(self._tools),
            "tool_schema_hash": tool_schema_hash(self._tools),
            "tool_calling": 0,
            "terminal_success": None,
            "terminal_reward": None,
            "terminal_outcome_valid": False,
            "protocol_reward": 0.0,
            "conversation_success": None,
            "semantic_train_mask": True,
            "runtime_train_mask": True,
            **summary,
            **self._last_info,
        }
        info.update(updates)
        return info

    def _observation_info(self):
        return self._last_observation, self._annotate()

    def _finalize_without_action(self, reason: str) -> dict[str, Any]:
        """Finalize an infrastructure/preflight stop for outcome diagnostics."""
        summary = self._checks()
        self._done = True
        self._last_info = {
            **self._last_info,
            "terminal_reason": str(reason),
            "terminal_success": summary["state_complete"],
            "conversation_success": summary["state_complete"],
            "terminal_reward": summary["checker_fraction"],
            "terminal_outcome_valid": True,
            "protocol_reward": float(summary["checker_fraction"]),
            "semantic_train_mask": False,
            "runtime_train_mask": False,
            "state_group_advanced": False,
            **summary,
        }
        return summary

    async def reset(self, *, task_index: int, seed: int | None = None):
        self._source = load_envscaler_source(self.source_root)
        self._task_index = int(task_index)
        if not 0 <= self._task_index < len(self._source.tasks):
            raise IndexError(f"EnvScaler task_index out of range: {self._task_index}")
        self._task = deepcopy(self._source.tasks[self._task_index])
        self._environment = deepcopy(self._source.environments[str(self._task["env_id"])])
        self._runtime = build_environment_instance(self._environment, self._task)
        self._tools = validate_tool_contract(self._environment, self._runtime)
        self._initial_state = state_dict(self._runtime)
        actual_seed = self.seed if seed is None else int(seed)
        self._rng.seed(actual_seed)
        self._no_progress.reset()
        self._last_selected_canonical_action = None
        self._simulator = ProviderUserSimulator(**self.user_config)
        self._step = 0
        self._done = False
        self._prepared_teacher_supervision = None
        self._reset_failure = None
        try:
            initial_user = await asyncio.to_thread(self._simulator.start, str(self._task["task"]))
        except Exception as exc:
            initial_user = "User simulator unavailable."
            self._reset_failure = f"{type(exc).__name__}: {exc}"
        self._chat = [
            {"role": "system", "content": agent_system_prompt(self._environment)},
            {"role": "user", "content": initial_user},
        ]
        self._last_observation = initial_user
        self._last_info = {
            "native_direct_tools": True,
            "user_simulator_failure": bool(self._reset_failure),
            "user_simulator_error": self._reset_failure,
        }
        return self._observation_info()

    def _validate(self, raw_action: str) -> AWMAction:
        return validate_action(parse_action(raw_action), self._tools)

    def _validate_visible_chat(self, visible_chat: list[dict[str, Any]] | None) -> None:
        if visible_chat is None:
            return
        if len(visible_chat) < 2 or visible_chat[:2] != self._chat[:2]:
            raise ValueError("EnvScaler visible chat must preserve the system/user prefix")
        logical = iter(self._chat[2:])
        for expected in visible_chat[2:]:
            if not any(candidate == expected for candidate in logical):
                raise ValueError("EnvScaler visible chat is not an ordered logical-history view")

    async def _classify_execution_exception(
        self,
        *,
        action: AWMAction,
        exception: Exception,
        traceback_text: str,
        state_before: Mapping[str, Any],
        state_at_exception: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.runtime_judge_enabled or self.oracle_actor is None:
            return {"status": "normal"}
        evidence = build_envscaler_runtime_judge_evidence(
            source_identity=(self._source.identity if self._source is not None else {}),
            task=self._task,
            environment=self._environment,
            visible_chat=self._chat,
            tools=openai_tools(self._tools),
            failed_action=action.to_dict(),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback_text=traceback_text,
            state_before=state_before,
            state_at_exception=state_at_exception,
        )
        verdict = None
        judge_error = None
        status = "masked"
        try:
            raw_verdict = await self.oracle_actor.classify_envscaler_runtime_failure.remote(
                evidence=evidence,
            )
            if not isinstance(raw_verdict, Mapping):
                raise TypeError("EnvScaler runtime judge verdict must be an object")
            verdict = dict(raw_verdict)
            confidence = verdict.get("classification_confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, int):
                raise TypeError("EnvScaler runtime judge confidence must be an integer")
            if verdict.get("error_class") == "policy_execution_error" and confidence >= self.runtime_judge_confidence_threshold and verdict.get("post_error_state") == "unchanged":
                status = "policy_penalized_continued"
        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {exc}"
        return {
            "status": status,
            "runtime_policy_error": status == "policy_penalized_continued",
            "runtime_judge": verdict,
            "runtime_judge_error": judge_error,
        }

    async def _execute(self, raw_action: str, action: AWMAction):
        self._step += 1
        execution_error = None
        simulator_error = None
        user_simulator_stop = False
        terminal_reason = None
        runtime_train_mask = True
        runtime = {"status": "normal"}

        if action.kind == "tool":
            snapshot = state_dict(self._runtime)
            try:
                value = getattr(self._runtime, action.name or "")(**(action.arguments or {}))
                response = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception as exc:
                exception_traceback = traceback.format_exc()
                try:
                    state_at_exception = state_dict(self._runtime)
                except Exception:
                    state_at_exception = {}
                restore_state(self._runtime, snapshot)
                execution_error = f"{type(exc).__name__}: {exc}"
                response = json.dumps(
                    {"error": execution_error, "state_restored": True},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                runtime = await self._classify_execution_exception(
                    action=action,
                    exception=exc,
                    traceback_text=exception_traceback,
                    state_before=snapshot,
                    state_at_exception=state_at_exception,
                )
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=response,
                tool_call_id=f"call_{self._step}",
            )
            self._last_observation = f"Tool response:\n{response}"
            if runtime["status"] == "masked":
                runtime_train_mask = False
                self._done = True
                terminal_reason = "runtime_masked"
        elif action.kind == "message":
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=None,
            )
            summary = self._checks()
            if summary["state_complete"]:
                self._done = True
                terminal_reason = "verifier_assisted_stop"
                self._last_observation = STOP
            else:
                try:
                    reply = await asyncio.to_thread(self._simulator.reply, action.content or "")
                    if reply == STOP:
                        user_simulator_stop = True
                        self._done = True
                        terminal_reason = "user_stop"
                        self._last_observation = STOP
                    else:
                        self._chat.append({"role": "user", "content": reply})
                        self._last_observation = reply
                except Exception as exc:
                    simulator_error = f"{type(exc).__name__}: {exc}"
                    runtime_train_mask = False
                    self._done = True
                    terminal_reason = "user_simulator_failure"
                    self._last_observation = "User simulator infrastructure failure."
        else:
            error = action.error or "invalid action"
            response = json.dumps({"error": error}, ensure_ascii=False)
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=response,
            )
            self._last_observation = f"Invalid action: {error}"

        if not self._done and self._step >= self.max_steps:
            self._done = True
            terminal_reason = "decision_limit"

        summary = self._checks()
        terminal_success = summary["state_complete"] if self._done else None
        terminal_reward = summary["checker_fraction"] if self._done else None
        self._last_info = {
            "execution_error": execution_error,
            "local_state_restored": bool(execution_error),
            "user_simulator_failure": bool(simulator_error),
            "user_simulator_error": simulator_error,
            "user_simulator_stop": user_simulator_stop,
            "runtime_train_mask": runtime_train_mask,
            "runtime_failure": not runtime_train_mask,
            "runtime_policy_error": bool(runtime.get("runtime_policy_error", False)),
            "runtime_policy_continued": (runtime["status"] == "policy_penalized_continued"),
            "runtime_policy_terminated": False,
            "runtime_judge_verdict": runtime.get("runtime_judge"),
            "runtime_judge_error": runtime.get("runtime_judge_error"),
            "runtime_judge_error_class": (runtime["runtime_judge"].get("error_class") if isinstance(runtime.get("runtime_judge"), Mapping) else None),
            "runtime_judge_confidence": (runtime["runtime_judge"].get("classification_confidence") if isinstance(runtime.get("runtime_judge"), Mapping) else None),
            "terminal_reason": terminal_reason,
            "terminal_success": terminal_success,
            "conversation_success": terminal_success,
            "terminal_reward": terminal_reward,
            "terminal_outcome_valid": self._done,
            "protocol_reward": float(terminal_reward or 0.0),
            **summary,
        }
        return self._done

    async def step(self, raw_action: str):
        if self._done:
            return self._last_observation, 0.0, True, self._annotate(terminal_reason="already_done")
        action = self._validate(raw_action)
        done = await self._execute(raw_action, action)
        reward = -1.0 if action.kind == "invalid" else 0.0
        return (
            self._last_observation,
            reward,
            done,
            self._annotate(
                raw_action=raw_action,
                parsed_action=canonical_action(action),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                tool_calling=int(action.kind == "tool"),
            ),
        )

    async def terminate_context_overflow(self, diagnostics: Mapping[str, Any]):
        self._done = True
        self._prepared_teacher_supervision = None
        summary = self._checks()
        self._last_info = {
            "action_kind": "context_overflow",
            "semantic_train_mask": False,
            "runtime_train_mask": False,
            "runtime_failure": False,
            "state_group_advanced": False,
            "terminal_reason": "context_budget_exceeded",
            "terminal_success": summary["state_complete"],
            "conversation_success": summary["state_complete"],
            "terminal_reward": summary["checker_fraction"],
            "terminal_outcome_valid": True,
            "protocol_reward": summary["checker_fraction"],
            "context_overflow": True,
            **summary,
            **dict(diagnostics),
        }
        return self._annotate()

    def _teacher_privileged_context(self) -> dict[str, Any]:
        return {
            "task_id": str(self._task.get("task_id") or ""),
            "canonical_task": str(self._task.get("task") or ""),
            "checklist": [str(item.get("check_item") or "") for item in self._task.get("checklist_with_func") or [] if str(item.get("check_item") or "").strip()],
        }

    async def prepare_teacher_supervision(self, visible_chat: list[dict[str, Any]] | None = None):
        if self.oracle_actor is None:
            raise RuntimeError("EnvScaler semantic rollout requires an oracle")
        self._validate_visible_chat(visible_chat)
        supervision_chat = self._chat if visible_chat is None else visible_chat
        teacher_messages = build_teacher_messages(
            supervision_chat,
            privileged_context=(self._teacher_privileged_context() if self.use_privileged_teacher_context else None),
            use_privileged_context=self.use_privileged_teacher_context,
        )
        fingerprint = state_fingerprint(
            f"envscaler:{self._task.get('env_id')}",
            self._task_index,
            (teacher_messages if self.use_privileged_teacher_context else supervision_chat),
            self._tools,
        )
        if self._reset_failure:
            self._finalize_without_action("user_simulator_failure")
            return False, self._annotate(
                action_kind="user_simulator_failure",
                semantic_train_mask=False,
                runtime_train_mask=False,
                teacher_failure=True,
                teacher_error=self._reset_failure,
                state_fingerprint=fingerprint,
                terminal_reason="user_simulator_failure",
                state_group_advanced=False,
            )
        samples = []
        actions = []
        try:
            samples = await self.oracle_actor.sample_multiset.remote(
                state_fingerprint=fingerprint,
                messages=teacher_messages,
                tools=openai_tools(self._tools),
                previous_canonical_action=self._last_selected_canonical_action,
                no_progress_repeat_streak=self._no_progress.repeat_streak,
            )
            if len(samples) > 3:
                raise RuntimeError(f"teacher returned {len(samples)} samples; expected at most 3")
            actions = validate_teacher_multiset(samples, self._tools)
            if not actions:
                raise RuntimeError("teacher multiset has no valid action")
        except Exception as exc:
            self._finalize_without_action("teacher_failure")
            self._prepared_teacher_supervision = None
            return False, self._annotate(
                action_kind="teacher_failure",
                semantic_train_mask=False,
                runtime_train_mask=False,
                teacher_failure=True,
                teacher_error=f"{type(exc).__name__}: {exc}",
                teacher_sample_count=len(samples),
                teacher_invalid_sample_count=len(samples) - len(actions),
                state_fingerprint=fingerprint,
                terminal_reason="teacher_failure",
                state_group_advanced=False,
            )
        self._prepared_teacher_supervision = {
            "state_fingerprint": fingerprint,
            "teacher_samples": samples,
            "teacher_actions": actions,
            "teacher_multiset": [action.to_dict() for action in actions],
            "teacher_invalid_sample_count": len(samples) - len(actions),
            "teacher_action_kind_disagreement": len({action.kind for action in actions}) > 1,
        }
        return True, self._annotate(
            action_kind="teacher_preflight",
            semantic_train_mask=False,
            teacher_frequency=0,
            teacher_multiset=self._prepared_teacher_supervision["teacher_multiset"],
            teacher_sample_count=len(samples),
            teacher_invalid_sample_count=len(samples) - len(actions),
            teacher_failure=False,
            state_fingerprint=fingerprint,
            state_group_advanced=False,
        )

    async def step_candidate_group(
        self,
        raw_actions: list[str],
        visible_chat: list[dict[str, Any]] | None = None,
        group_metadata: Mapping[str, Any] | None = None,
    ):
        self._validate_visible_chat(visible_chat)
        supervision_chat = self._chat if visible_chat is None else visible_chat
        teacher_messages = build_teacher_messages(
            supervision_chat,
            privileged_context=(self._teacher_privileged_context() if self.use_privileged_teacher_context else None),
            use_privileged_context=self.use_privileged_teacher_context,
        )
        fingerprint = state_fingerprint(
            f"envscaler:{self._task.get('env_id')}",
            self._task_index,
            (teacher_messages if self.use_privileged_teacher_context else supervision_chat),
            self._tools,
        )
        prepared = self._prepared_teacher_supervision
        self._prepared_teacher_supervision = None
        if prepared is None or prepared["state_fingerprint"] != fingerprint:
            raise RuntimeError("EnvScaler candidates require matching teacher-first preflight")
        candidates = [self._validate(raw) for raw in raw_actions]
        teacher_actions = prepared["teacher_actions"]
        message_positions = [index for index, action in enumerate(candidates) if action.kind == "message"]
        teacher_messages = [action.content or "" for action in teacher_actions if action.kind == "message"]
        message_counts: dict[int, int] = {}
        matcher_matrix = []
        if message_positions and teacher_messages:
            try:
                matched = await self.oracle_actor.match_message_pairs.remote(
                    teacher_messages,
                    [candidates[index].content or "" for index in message_positions],
                )
                counts = matched["counts"]
                matcher_matrix = matched["matrix"]
                if len(counts) != len(message_positions):
                    raise ValueError("matcher returned wrong candidate count")
                for count, row in zip(counts, matcher_matrix, strict=True):
                    if not isinstance(count, int) or len(row) != len(teacher_messages) or count != sum(bool(value) for value in row):
                        raise ValueError("matcher returned invalid Boolean matrix")
                message_counts = dict(zip(message_positions, counts, strict=True))
            except Exception as exc:
                self._finalize_without_action("matcher_failure")
                results = []
                for raw, action in zip(raw_actions, candidates, strict=True):
                    results.append(
                        (
                            self._last_observation,
                            0.0,
                            False,
                            self._annotate(
                                raw_action=raw,
                                parsed_action=canonical_action(action),
                                action_kind=action.kind,
                                semantic_train_mask=False,
                                matcher_failure=True,
                                matcher_error=f"{type(exc).__name__}: {exc}",
                                state_group_advanced=False,
                                terminal_success=None,
                                terminal_reward=None,
                                terminal_outcome_valid=False,
                                protocol_reward=0.0,
                                **dict(group_metadata or {}),
                            ),
                        )
                    )
                info = self._annotate(
                    action_kind="matcher_failure",
                    semantic_train_mask=False,
                    matcher_failure=True,
                    matcher_error=f"{type(exc).__name__}: {exc}",
                    terminal_reason="matcher_failure",
                    state_group_advanced=False,
                    **dict(group_metadata or {}),
                )
                return results, -1, self._last_observation, 0.0, True, info

        raw_scored = score_candidates(
            candidates,
            teacher_actions,
            message_match_counts=message_counts,
            teacher_sample_count=len(prepared["teacher_samples"]),
            frequency_bonus_scale=self.frequency_bonus_scale,
            teacher_reward_mode=self.teacher_reward_mode,
        )
        canonical_actions = [canonical_action(action) for action in candidates]
        prospective_repeats = (
            self._no_progress.prospective_repeat_flags(
                action_kinds=[action.kind for action in candidates],
                canonical_actions=canonical_actions,
                min_streak=self.repeat_reward_cap_min_streak,
            )
            if self.repeat_reward_cap_enabled
            else [False] * len(candidates)
        )
        repeat_reward_capped = [
            bool(repeated and float(item.reward or 0.0) > self.repeat_reward_cap_value)
            for repeated, item in zip(
                prospective_repeats,
                raw_scored,
                strict=True,
            )
        ]
        scored = [
            replace(
                item,
                reward=min(float(item.reward or 0.0), self.repeat_reward_cap_value),
                selection_score=min(
                    float(item.selection_score),
                    self.repeat_reward_cap_value,
                ),
            )
            if capped
            else item
            for item, capped in zip(raw_scored, repeat_reward_capped, strict=True)
        ]
        frequency_sensitive = frequency_sensitive_group(scored)
        appearance_scores = [-1.0 if action.kind == "invalid" else (1.0 if item.teacher_frequency > 0 else 0.0) for action, item in zip(candidates, scored, strict=True)]
        appearance_scores = [
            min(score, self.repeat_reward_cap_value) if capped else score
            for score, capped in zip(
                appearance_scores,
                repeat_reward_capped,
                strict=True,
            )
        ]
        selection = select_history_aware_with_appearance_counterfactual(
            [item.selection_score for item in scored],
            appearance_scores,
            canonical_actions,
            (self._last_selected_canonical_action if self.prefer_nonrepeat_argmax else None),
            self._rng,
        )
        selected_index = selection.selected_index
        appearance_index = selection.appearance_index
        selected_action = candidates[selected_index]
        appearance_action = candidates[appearance_index]
        frequency_changed_selection = canonical_action(selected_action) != canonical_action(appearance_action)
        done = await self._execute(raw_actions[selected_index], selected_action)
        repeat_streak_before = self._no_progress.repeat_streak
        self._no_progress.record(
            action_kind=selected_action.kind,
            canonical_action=canonical_actions[selected_index],
            observation=self._last_observation,
        )
        self._last_selected_canonical_action = canonical_actions[selected_index]
        if not done and self.repeat_termination_enabled and self._no_progress.reached(self.repeat_termination_max_streak):
            summary = self._checks()
            self._done = True
            done = True
            self._last_info = {
                **self._last_info,
                "terminal_reason": "no_progress_repeat_limit",
                "terminal_success": summary["state_complete"],
                "conversation_success": summary["state_complete"],
                "terminal_reward": summary["checker_fraction"],
                "terminal_outcome_valid": True,
                "protocol_reward": float(summary["checker_fraction"]),
                **summary,
            }
        runtime_train_mask = bool(self._last_info.get("runtime_train_mask", True))
        penalized_action = canonical_action(selected_action) if self._last_info.get("runtime_policy_error", False) else None
        results = []
        teacher_multiset = prepared["teacher_multiset"]
        for index, (raw, action, item) in enumerate(zip(raw_actions, candidates, scored, strict=True)):
            selected = index == selected_index
            runtime_policy_penalty = penalized_action is not None and canonical_action(action) == penalized_action
            reward = -1.0 if runtime_policy_penalty else float(item.reward or 0.0)
            info = self._annotate(
                raw_action=raw,
                parsed_action=canonical_actions[index],
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=bool(action.kind == "invalid" or runtime_policy_penalty),
                is_action_valid=int(action.kind != "invalid" and not runtime_policy_penalty),
                move_optimal=bool(item.teacher_frequency > 0 and not runtime_policy_penalty),
                legal_non_oracle=bool(action.kind != "invalid" and item.teacher_frequency == 0 and not runtime_policy_penalty),
                semantic_train_mask=bool(item.semantic_train_mask and runtime_train_mask),
                runtime_train_mask=runtime_train_mask,
                runtime_policy_penalty=runtime_policy_penalty,
                selection_score=float(item.selection_score),
                raw_selection_score=float(raw_scored[index].selection_score),
                raw_semantic_reward=float(raw_scored[index].reward or 0.0),
                prospective_no_progress_repeat=prospective_repeats[index],
                repeat_reward_capped=repeat_reward_capped[index],
                no_progress_repeat_streak_before=repeat_streak_before,
                no_progress_repeat_streak_after=(self._no_progress.repeat_streak if selected else repeat_streak_before),
                teacher_frequency=item.teacher_frequency,
                teacher_multiset=teacher_multiset,
                teacher_multiset_size=len(teacher_multiset),
                teacher_sample_count=len(prepared["teacher_samples"]),
                teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=frequency_sensitive,
                appearance_counterfactual_selected=index == appearance_index,
                appearance_counterfactual_action_kind=appearance_action.kind,
                frequency_changed_selection=frequency_changed_selection,
                frequency_changed_selection_to_tool=bool(frequency_changed_selection and selected_action.kind == "tool"),
                frequency_changed_selection_to_message=bool(frequency_changed_selection and selected_action.kind == "message"),
                teacher_failure=False,
                matcher_failure=False,
                matcher_matrix=matcher_matrix,
                state_fingerprint=fingerprint,
                state_group_selection_type=selection.selection_type,
                state_group_random_select_prob=0.0,
                state_group_advanced=bool(selected and runtime_train_mask),
                nonrepeat_alternative_available=selection.nonrepeat_alternative_available,
                nonrepeat_preference_applied=selection.nonrepeat_preference_applied,
                **dict(group_metadata or {}),
                terminal_success=(self._last_info.get("terminal_success") if selected and done else None),
                terminal_reason=(self._last_info.get("terminal_reason") if selected and done else None),
                terminal_reward=(self._last_info.get("terminal_reward") if selected and done else None),
                terminal_outcome_valid=bool(selected and done),
                protocol_reward=(float(self._last_info.get("protocol_reward", 0.0)) if selected and done else 0.0),
            )
            results.append(
                (
                    self._last_observation,
                    reward,
                    bool(selected and done),
                    info,
                )
            )
        selected_info = results[selected_index][3]
        return (
            results,
            selected_index,
            self._last_observation,
            (float(results[selected_index][1]) if runtime_train_mask else 0.0),
            done,
            selected_info,
        )

    async def close(self):
        return None
