"""Conversation semantic runtime for pinned EnvScaler RL tasks."""

from __future__ import annotations

import asyncio
import json
import random
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
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
    select_uniform_argmax,
    validate_teacher_multiset,
)
from agent_system.environments.env_package.awm.runtime.oracle import (
    build_expert_messages,
)

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
from .user_simulator import STOP, DeepSeekUserSimulator

ENVSCALER_PROTOCOL_VERSION = 3


def agent_system_prompt(environment: Mapping[str, Any]) -> str:
    rules = "\n".join(f"- {rule}" for rule in environment.get("constraints_rules") or [])
    introduction = str(environment.get("environment_introduction") or "")
    return f"""You are operating in an interactive environment through native function calling.

Environment:
{introduction}

Rules:
{rules}

At each decision take exactly one action: call exactly one available function, or send
exactly one ordinary assistant message to the user. Never combine a function call with a
message and never call multiple functions. Use ordinary messages to ask for missing
information and to tell the user when the task is complete."""


def _frequency_sensitive_group(scored) -> bool:
    eligible = [item for item in scored if item.semantic_train_mask]
    if len(eligible) < 2:
        return False
    rewards = np.asarray([float(item.reward) for item in eligible])
    binary = np.where(rewards > 0, 1.0, rewards)
    return bool(not np.array_equal(rewards, binary))


@ray.remote(max_concurrency=8)
class EnvScalerWorker:
    def __init__(
        self,
        *,
        source_root: str = str(DEFAULT_SOURCE_ROOT),
        max_steps: int = 40,
        oracle_actor=None,
        user_model: str = "deepseek-v4-flash",
        user_api_base: str = "https://api.deepseek.com",
        user_api_key_env: str = "DEEPSEEK_API_KEY",
        user_temperature: float = 1.0,
        user_reasoning_enabled: bool = False,
        user_timeout_seconds: float = 300,
        user_max_retries: int = 3,
        seed: int = 0,
    ):
        self.source_root = str(source_root)
        self.max_steps = int(max_steps)
        self.oracle_actor = oracle_actor
        self.user_config = {
            "model": str(user_model),
            "api_base": str(user_api_base),
            "api_key_env": str(user_api_key_env),
            "temperature": float(user_temperature),
            "reasoning_enabled": bool(user_reasoning_enabled),
            "timeout_seconds": float(user_timeout_seconds),
            "max_retries": int(user_max_retries),
        }
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._source = None
        self._runtime = None
        self._task: dict[str, Any] = {}
        self._environment: dict[str, Any] = {}
        self._initial_state: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []
        self._chat: list[dict[str, Any]] = []
        self._simulator: DeepSeekUserSimulator | None = None
        self._task_index = -1
        self._step = 0
        self._done = False
        self._last_observation = ""
        self._last_info: dict[str, Any] = {}
        self._prepared_supervision: dict[str, Any] | None = None
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
        self._simulator = DeepSeekUserSimulator(**self.user_config)
        self._step = 0
        self._done = False
        self._prepared_supervision = None
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

    async def _execute(self, raw_action: str, action: AWMAction):
        self._step += 1
        execution_error = None
        simulator_error = None
        user_simulator_stop = False
        terminal_reason = None
        runtime_train_mask = True

        if action.kind == "tool":
            snapshot = state_dict(self._runtime)
            try:
                value = getattr(self._runtime, action.name or "")(**(action.arguments or {}))
                response = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception as exc:
                restore_state(self._runtime, snapshot)
                execution_error = f"{type(exc).__name__}: {exc}"
                response = json.dumps(
                    {"error": execution_error, "state_restored": True},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=response,
                tool_call_id=f"call_{self._step}",
            )
            self._last_observation = f"Tool response:\n{response}"
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
        self._prepared_supervision = None
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

    async def prepare_state_group(self, visible_chat: list[dict[str, Any]] | None = None):
        if self.oracle_actor is None:
            raise RuntimeError("EnvScaler semantic rollout requires an oracle")
        self._validate_visible_chat(visible_chat)
        supervision_chat = self._chat if visible_chat is None else visible_chat
        fingerprint = state_fingerprint(
            f"envscaler:{self._task.get('env_id')}",
            self._task_index,
            supervision_chat,
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
                messages=build_expert_messages(supervision_chat),
                tools=openai_tools(self._tools),
            )
            if len(samples) != 3:
                raise RuntimeError(f"teacher returned {len(samples)} samples instead of 3")
            actions = validate_teacher_multiset(samples, self._tools)
            if not actions:
                raise RuntimeError("teacher multiset has no valid action")
        except Exception as exc:
            self._finalize_without_action("teacher_failure")
            self._prepared_supervision = None
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
        self._prepared_supervision = {
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
            teacher_multiset=self._prepared_supervision["teacher_multiset"],
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
    ):
        self._validate_visible_chat(visible_chat)
        supervision_chat = self._chat if visible_chat is None else visible_chat
        fingerprint = state_fingerprint(
            f"envscaler:{self._task.get('env_id')}",
            self._task_index,
            supervision_chat,
            self._tools,
        )
        prepared = self._prepared_supervision
        self._prepared_supervision = None
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
                )
                return results, -1, self._last_observation, 0.0, True, info

        scored = score_candidates(
            candidates,
            teacher_actions,
            message_match_counts=message_counts,
        )
        selected_index = select_uniform_argmax([item.selection_score for item in scored], self._rng)
        done = await self._execute(raw_actions[selected_index], candidates[selected_index])
        runtime_train_mask = bool(self._last_info.get("runtime_train_mask", True))
        results = []
        teacher_multiset = prepared["teacher_multiset"]
        for index, (raw, action, item) in enumerate(zip(raw_actions, candidates, scored, strict=True)):
            selected = index == selected_index
            info = self._annotate(
                raw_action=raw,
                parsed_action=canonical_action(action),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                move_optimal=bool(item.teacher_frequency > 0),
                legal_non_oracle=bool(action.kind != "invalid" and item.teacher_frequency == 0),
                semantic_train_mask=bool(item.semantic_train_mask and runtime_train_mask),
                runtime_train_mask=runtime_train_mask,
                teacher_frequency=item.teacher_frequency,
                teacher_multiset=teacher_multiset,
                teacher_multiset_size=len(teacher_multiset),
                teacher_sample_count=len(prepared["teacher_samples"]),
                teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=_frequency_sensitive_group(scored),
                teacher_failure=False,
                matcher_failure=False,
                matcher_matrix=matcher_matrix,
                state_fingerprint=fingerprint,
                state_group_selection_type="uniform_argmax",
                state_group_random_select_prob=0.0,
                state_group_advanced=bool(selected and runtime_train_mask),
                terminal_success=(self._last_info.get("terminal_success") if selected and done else None),
                terminal_reason=(self._last_info.get("terminal_reason") if selected and done else None),
                terminal_reward=(self._last_info.get("terminal_reward") if selected and done else None),
                terminal_outcome_valid=bool(selected and done),
                protocol_reward=(float(self._last_info.get("protocol_reward", 0.0)) if selected and done else 0.0),
            )
            results.append(
                (
                    self._last_observation,
                    float(item.reward or 0.0),
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
