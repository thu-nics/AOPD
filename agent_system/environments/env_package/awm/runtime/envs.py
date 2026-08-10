"""Ray workers for AgentWorldModel semantic and outcome training."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import ray

from .actions import (
    AWMAction,
    append_exchange,
    build_native_chat,
    canonical_action,
    normalize_tools,
    openai_tools,
    parse_action,
    score_candidates,
    state_fingerprint,
    tool_schema_hash,
    validate_action,
)
from .failures import (
    AWMRuntimeFailureRecorder,
    deterministic_error_signature,
    infrastructure_error,
    judgeable_tool_error,
    task_id,
)
from .oracle import build_expert_messages

AWM_OPENENV_COMMIT = "5298e0d91c6cd55d5f3a81259d5b2a9a1e05eff0"
AWM_DATASET_REVISION = "dde80a0283fe781bdc51656bce57063dc5650213"
AWM_DATASET_NAME = "Snowflake/AgentWorldModel-1K"
AWM_PROTOCOL_VERSION = 11


def select_uniform_argmax(scores: Sequence[float], rng: random.Random) -> int:
    if not scores:
        raise ValueError("cannot select from an empty candidate group")
    maximum = max(scores)
    return rng.choice([index for index, score in enumerate(scores) if score == maximum])


def validate_teacher_multiset(
    teacher_samples: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> list[AWMAction]:
    """Keep only schema-valid tool/message samples, preserving order and duplicates."""
    actions = []
    for sample in teacher_samples:
        try:
            action = validate_action(AWMAction(**dict(sample["action"])), tools)
        except (KeyError, TypeError, ValueError):
            continue
        if action.kind in {"tool", "message"}:
            actions.append(action)
    return actions


def _observation_dict(result: Any) -> dict[str, Any]:
    observation = getattr(result, "observation", result)
    if hasattr(observation, "model_dump"):
        return observation.model_dump(mode="json")
    if isinstance(observation, Mapping):
        return dict(observation)
    return {"value": str(observation)}


def _tool_response_text(result: Any) -> str:
    payload = _observation_dict(result)
    if payload.get("tool_result") is not None:
        value = payload["tool_result"]
    elif payload.get("result") is not None:
        value = payload["result"]
    elif payload.get("error"):
        value = {"error": payload["error"]}
    else:
        value = payload
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@ray.remote(max_concurrency=8)
class AWMWorker:
    def __init__(
        self,
        *,
        base_url: str,
        max_steps: int,
        history_window: int,
        verifier_mode: str,
        reward_mode: str,
        oracle_actor=None,
        runtime_recorder=None,
        runtime_judge_enabled: bool = False,
        runtime_judge_confidence_threshold: int = 80,
        terminal_judge_api_base: str | None = None,
        terminal_judge_api_key_env: str | None = None,
        terminal_judge_model: str | None = None,
        seed: int = 0,
    ):
        if reward_mode not in {"semantic", "outcome"}:
            raise ValueError(f"unsupported AWM reward_mode: {reward_mode}")
        if verifier_mode != "sql":
            raise ValueError("AWM worker requires verifier_mode=sql")
        history_window = int(history_window)
        if history_window < 0:
            raise ValueError("AWM history_window must be non-negative")
        self.base_url = str(base_url)
        self.max_steps = int(max_steps)
        self.history_window = history_window
        self.verifier_mode = verifier_mode
        self.reward_mode = reward_mode
        self.oracle_actor = oracle_actor
        self.runtime_recorder = runtime_recorder
        self.seed = int(seed)
        self.terminal_judge_api_base = str(terminal_judge_api_base or "")
        self.terminal_judge_model = str(terminal_judge_model or "")
        self.terminal_judge_api_key_env = str(terminal_judge_api_key_env or "")
        self.terminal_judge_api_key = os.environ.get(self.terminal_judge_api_key_env, "") if self.terminal_judge_api_key_env else ""
        self.runtime_judge_enabled = bool(runtime_judge_enabled)
        self.runtime_judge_confidence_threshold = int(runtime_judge_confidence_threshold)
        if not 0 <= self.runtime_judge_confidence_threshold <= 100:
            raise ValueError("AWM runtime judge confidence threshold must be in [0, 100]")
        self._rng = random.Random(seed)
        self._env = None
        self._scenario = ""
        self._task_idx = -1
        self._task = ""
        self._tools: list[dict[str, Any]] = []
        self._chat: list[dict[str, Any]] = []
        self._step = 0
        self._done = False
        self._last_observation = ""
        self._last_info: dict[str, Any] = {}
        self._prepared_supervision: dict[str, Any] | None = None
        self._actual_seed = int(seed)

        self._terminal_result: tuple[float, dict[str, Any], dict[str, Any]] | None = None

    async def _close_env(self) -> None:
        if self._env is None:
            return
        try:
            await self._env.__aexit__(None, None, None)
        finally:
            self._env = None

    async def _new_env(self):
        try:
            from agent_world_model_env import AWMEnv
        except ImportError as exc:
            raise RuntimeError("AgentWorldModel OpenEnv is not installed. Run examples/awm/setup/install_awm.sh first.") from exc
        env = AWMEnv(base_url=self.base_url)
        await env.__aenter__()
        return env

    def _validate(self, raw_action: str) -> AWMAction:
        return validate_action(parse_action(raw_action), self._tools)

    def _annotate(self, **updates: Any) -> dict[str, Any]:
        result = {
            "awm_protocol_version": AWM_PROTOCOL_VERSION,
            "awm_scenario": self._scenario,
            "awm_task_idx": self._task_idx,
            "vpr_game": "awm",
            "step": self._step,
            "max_steps": self.max_steps,
            "observation": self._last_observation,
            "chat": list(self._chat),
            "tools": openai_tools(self._tools),
            "tool_schema_hash": tool_schema_hash(self._tools),
            "tool_calling": 0,
            "terminal_success": None,
            "protocol_reward": 0.0,
            "terminal_label": None,
            "terminal_reward": None,
            "terminal_outcome_valid": False,
            "outcome_train_mask": True,
            **self._last_info,
        }
        result.update(updates)
        return result

    def _observation_info(self) -> tuple[str, dict[str, Any]]:
        return self._last_observation, self._annotate()

    async def reset(self, *, scenario: str, task_idx: int, seed: int | None = None):
        await self._close_env()
        actual_seed = self.seed if seed is None else int(seed)
        self._actual_seed = actual_seed
        self._rng.seed(actual_seed)
        self._env = await self._new_env()
        self._terminal_result = None
        reset_result = await self._env.reset(
            scenario=str(scenario),
            task_idx=int(task_idx),
            seed=actual_seed,
            llm_base_url=self.terminal_judge_api_base or None,
            llm_api_key=self.terminal_judge_api_key or None,
            llm_model=self.terminal_judge_model or None,
        )
        reset_payload = _observation_dict(reset_result)
        if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
            await self._close_env()
            raise RuntimeError(f"AWM reset failed for {scenario}[{task_idx}]: {reset_payload.get('error') or reset_payload}")
        tools = await self._env.list_tools(use_cache=False)
        self._scenario = str(reset_payload.get("scenario") or scenario)
        self._task_idx = int(reset_payload.get("task_idx", task_idx))
        self._task = str(reset_payload.get("task") or "")
        self._tools = normalize_tools(tools)
        self._chat = build_native_chat(self._task)
        self._step = 0
        self._done = False
        self._last_observation = self._task
        self._prepared_supervision = None
        item_task_id = task_id(self._scenario, self._task_idx)
        self._last_info = {
            "native_direct_tools": True,
            "awm_task_id": item_task_id,
        }
        return self._observation_info()

    async def _call_tool(self, action: AWMAction) -> tuple[str, dict[str, Any]]:
        from openenv.core.env_server.mcp_types import CallToolAction

        result = await self._env.step(CallToolAction(tool_name=action.name or "", arguments=action.arguments or {}))
        return _tool_response_text(result), _observation_dict(result)

    async def _classify_runtime_failure(
        self,
        *,
        phase: str,
        payload: Mapping[str, Any],
        action: AWMAction | None = None,
        final_answer: str | None = None,
    ) -> dict[str, Any]:
        if not infrastructure_error(
            payload,
            phase=phase,
        ):
            return {"status": "normal", "payload": dict(payload)}

        signature = deterministic_error_signature(
            payload,
            phase=phase,
            tool_name=(action.name if action is not None else phase),
        )
        verdict: dict[str, Any] | None = None
        judge_error: str | None = None
        status = "masked"
        if self.runtime_judge_enabled and self.oracle_actor is not None and action is not None and judgeable_tool_error(payload, phase=phase):
            try:
                raw_verdict = await self.oracle_actor.classify_runtime_failure.remote(
                    scenario=self._scenario,
                    task_idx=self._task_idx,
                    task=self._task,
                    failed_action=action.to_dict(),
                    payload=dict(payload),
                )
                if not isinstance(raw_verdict, Mapping):
                    raise TypeError("runtime judge verdict must be an object")
                verdict = dict(raw_verdict)
                confidence = verdict.get("classification_confidence")
                if isinstance(confidence, bool) or not isinstance(confidence, int):
                    raise TypeError("runtime judge confidence must be an integer")
                if verdict.get("error_class") == "policy_execution_error" and confidence >= self.runtime_judge_confidence_threshold:
                    status = "policy_penalized_continued" if verdict.get("post_error_state") == "unchanged" else "policy_penalized_terminated"
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: {exc}"

        failing_action = action.to_dict() if action is not None else {"kind": phase, "final_answer": final_answer}
        record = {
            "task_id": task_id(self._scenario, self._task_idx),
            "scenario": self._scenario,
            "task_idx": self._task_idx,
            "seed": self._actual_seed,
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "signature": signature,
            "failing_action": failing_action,
            "payload": dict(payload),
            "runtime_judge": verdict,
            "runtime_judge_error": judge_error,
            "status": status,
        }
        if self.runtime_recorder is not None:
            await self.runtime_recorder.record.remote(record)
        return {
            "status": status,
            "signature": signature,
            "payload": dict(payload),
            "runtime_policy_error": status.startswith("policy_penalized_"),
            "runtime_judge": verdict,
            "runtime_judge_error": judge_error,
        }

    async def _verify_and_done(self, final_answer: str | None) -> tuple[float, dict[str, Any], dict[str, Any]]:
        from openenv.core.env_server.mcp_types import CallToolAction

        if self._terminal_result is not None:
            return self._terminal_result
        try:
            verify_result = await self._env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={
                        "verifier_mode": self.verifier_mode,
                        "final_answer": final_answer,
                    },
                )
            )
            verify_payload = _observation_dict(verify_result)
            reward = float(getattr(verify_result, "reward", 0.0) or 0.0)
        except Exception as exc:
            verify_payload = {
                "reward_type": "runtime_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
            reward = 0.0
        runtime = {"status": "normal"}
        try:
            await self._env.step(CallToolAction(tool_name="done", arguments={}))
        except Exception as exc:
            runtime["done_error"] = f"{type(exc).__name__}: {exc}"
        self._terminal_result = (reward, verify_payload, runtime)
        return self._terminal_result

    @staticmethod
    def _terminal_metadata(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        label = str(payload.get("reward_type") or "")
        rewards = {"complete": 1.0, "incomplete": 0.1, "agent_error": 0.0}
        valid = label in rewards
        verify_result = payload.get("verify_result")
        judge = verify_result.get("llm_judge") if isinstance(verify_result, Mapping) else None
        judge_error = payload.get("error")
        if isinstance(verify_result, Mapping):
            judge_error = verify_result.get("llm_judge_error") or (judge.get("error") if isinstance(judge, Mapping) else None) or verify_result.get("error") or verify_result.get("error_message") or judge_error
        return {
            "awm_reward_type": label or None,
            "awm_verify_result": verify_result,
            "terminal_label": label or None,
            "terminal_reward": rewards[label] if valid else None,
            "terminal_outcome_valid": valid,
            "outcome_train_mask": valid,
            "terminal_success": (label == "complete") if valid else None,
            "terminal_judge_result": judge if isinstance(judge, Mapping) else None,
            "terminal_judge_error": judge_error,
        }

    async def _execute(self, raw_action: str, action: AWMAction):
        self._step += 1
        protocol_reward = 0.0
        terminal_success: bool | None = None
        terminal_reason: str | None = None
        terminal_metadata: dict[str, Any] = {}
        environment_payload: dict[str, Any] = {}
        runtime: dict[str, Any] = {"status": "normal"}
        action_runtime: dict[str, Any] = runtime
        if action.kind == "tool":
            try:
                response, environment_payload = await self._call_tool(action)
            except Exception as exc:
                environment_payload = {
                    "reward_type": "runtime_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                response = json.dumps(
                    {"error": environment_payload["error"]},
                    ensure_ascii=False,
                )
            runtime = await self._classify_runtime_failure(
                phase="tool",
                payload=environment_payload,
                action=action,
            )
            action_runtime = runtime
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=response,
                history_window=self.history_window,
                tool_call_id=f"call_{self._step}",
            )
            self._last_observation = f"Tool response:\n{response}"
        elif action.kind == "message":
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=None,
                history_window=self.history_window,
            )
            self._last_observation = action.content or ""
            protocol_reward, environment_payload, runtime = await self._verify_and_done(action.content)
            self._done = True
            terminal_metadata = self._terminal_metadata(environment_payload)
            protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
            terminal_success = terminal_metadata["terminal_success"]
            terminal_reason = "final_response"
            runtime = action_runtime
        else:
            error = action.error or "invalid action"
            response = json.dumps({"error": error}, ensure_ascii=False)
            self._chat = append_exchange(
                self._chat,
                action=action,
                raw_action=raw_action,
                tool_response=response,
                history_window=self.history_window,
            )
            self._last_observation = f"Invalid action: {error}"

        if runtime["status"] in {"masked", "policy_penalized_terminated"}:
            protocol_reward, terminal_payload, _ = await self._verify_and_done(None)
            terminal_metadata = self._terminal_metadata(terminal_payload)
            protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
            environment_payload = terminal_payload
            self._done = True
            terminal_success = terminal_metadata["terminal_success"]
            terminal_reason = f"runtime_{runtime['status']}"
        if not self._done and self._step >= self.max_steps:
            protocol_reward, environment_payload, _ = await self._verify_and_done(None)
            terminal_metadata = self._terminal_metadata(environment_payload)
            protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
            self._done = True
            terminal_success = terminal_metadata["terminal_success"]
            terminal_reason = "decision_limit"
            runtime = action_runtime

        verdict = runtime.get("runtime_judge")
        self._last_info = {
            "protocol_reward": protocol_reward,
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
            **terminal_metadata,
            "runtime_train_mask": runtime["status"] != "masked",
            "runtime_failure": runtime["status"] == "masked",
            "runtime_error_signature": runtime.get("signature"),
            "runtime_policy_error": bool(runtime.get("runtime_policy_error", False)),
            "runtime_policy_continued": (runtime["status"] == "policy_penalized_continued"),
            "runtime_policy_terminated": (runtime["status"] == "policy_penalized_terminated"),
            "runtime_judge_verdict": verdict,
            "runtime_judge_error": runtime.get("runtime_judge_error"),
            "runtime_judge_error_class": (verdict.get("error_class") if isinstance(verdict, Mapping) else None),
            "runtime_judge_confidence": (verdict.get("classification_confidence") if isinstance(verdict, Mapping) else None),
            "runtime_judge_post_error_state": (verdict.get("post_error_state") if isinstance(verdict, Mapping) else None),
        }
        return protocol_reward, self._done

    async def step(self, raw_action: str):
        if self._done:
            info = self._annotate(
                raw_action=raw_action,
                parsed_action="",
                parse_ok=True,
                illegal_action=False,
                is_action_valid=1,
                terminal_reason="already_done",
            )
            return self._last_observation, 0.0, True, info
        action = self._validate(raw_action)
        protocol_reward, done = await self._execute(raw_action, action)
        if self.reward_mode == "outcome":
            reward = protocol_reward if self._last_info.get("outcome_train_mask", False) else 0.0
        elif self._last_info.get("runtime_policy_error", False):
            reward = -1.0
        else:
            reward = 0.0
        info = self._annotate(
            raw_action=raw_action,
            parsed_action=canonical_action(action),
            action_kind=action.kind,
            parse_ok=action.kind != "invalid",
            illegal_action=action.kind == "invalid",
            is_action_valid=int(action.kind != "invalid"),
            semantic_train_mask=True,
            tool_calling=int(action.kind == "tool"),
        )
        return self._last_observation, reward, done, info

    async def terminate_context_overflow(self, diagnostics: Mapping[str, Any]):
        """End one oversized state without treating it as an action or task defect."""
        self._prepared_supervision = None
        protocol_reward, terminal_payload, _ = await self._verify_and_done(None)
        terminal_metadata = self._terminal_metadata(terminal_payload)
        protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
        self._done = True
        self._last_info = {
            "action_kind": "context_overflow",
            "protocol_reward": protocol_reward,
            **terminal_metadata,
            "semantic_train_mask": False,
            "runtime_train_mask": False,
            "runtime_failure": False,
            "state_group_advanced": False,
            "terminal_reason": "context_budget_exceeded",
            "context_overflow": True,
            **dict(diagnostics),
        }
        await self._close_env()
        return self._annotate()

    def _set_visible_chat(
        self,
        visible_chat: list[dict[str, Any]] | None,
    ) -> None:
        if visible_chat is None:
            return
        if len(visible_chat) < 2 or visible_chat[:2] != self._chat[:2]:
            raise ValueError("AWM visible chat must preserve the exact system/task prefix")
        self._chat = [dict(message) for message in visible_chat]

    async def prepare_state_group(
        self,
        visible_chat: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Query and freeze K=3 teacher actions before student generation."""
        if self.oracle_actor is None:
            raise RuntimeError("state-group AWM rollout requires an oracle actor")
        if self._done:
            raise RuntimeError("cannot prepare an AWM state group after termination")
        self._set_visible_chat(visible_chat)
        fingerprint = state_fingerprint(
            self._scenario,
            self._task_idx,
            self._chat,
            self._tools,
        )
        teacher_samples: list[dict[str, Any]] = []
        teacher_actions: list[AWMAction] = []
        try:
            teacher_samples = await self.oracle_actor.sample_multiset.remote(
                state_fingerprint=fingerprint,
                messages=build_expert_messages(self._chat),
                tools=openai_tools(self._tools),
            )
            if len(teacher_samples) != 3:
                raise RuntimeError(f"teacher returned {len(teacher_samples)} samples instead of 3")
            teacher_actions = validate_teacher_multiset(teacher_samples, self._tools)
            if not teacher_actions:
                raise RuntimeError("teacher multiset has no schema-valid tool or message action")
        except Exception as exc:
            self._prepared_supervision = None
            invalid_count = len(teacher_samples) - len(teacher_actions)
            protocol_reward, terminal_payload, _ = await self._verify_and_done(None)
            terminal_metadata = self._terminal_metadata(terminal_payload)
            protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
            self._done = True
            return False, self._annotate(
                action_kind="teacher_failure",
                protocol_reward=protocol_reward,
                **terminal_metadata,
                semantic_train_mask=False,
                teacher_frequency=0,
                teacher_multiset=[],
                teacher_multiset_size=0,
                teacher_sample_count=len(teacher_samples),
                teacher_invalid_sample_count=max(invalid_count, 0),
                teacher_action_kind_disagreement=False,
                teacher_failure=True,
                teacher_error=f"{type(exc).__name__}: {exc}",
                matcher_matrix=[],
                state_fingerprint=fingerprint,
                terminal_reason="teacher_failure",
                state_group_advanced=False,
            )

        teacher_multiset = [action.to_dict() for action in teacher_actions]
        disagreement = len({action.kind for action in teacher_actions}) > 1
        self._prepared_supervision = {
            "state_fingerprint": fingerprint,
            "teacher_samples": teacher_samples,
            "teacher_actions": teacher_actions,
            "teacher_multiset": teacher_multiset,
            "teacher_invalid_sample_count": len(teacher_samples) - len(teacher_actions),
            "teacher_action_kind_disagreement": disagreement,
        }
        return True, self._annotate(
            action_kind="teacher_preflight",
            semantic_train_mask=False,
            teacher_frequency=0,
            teacher_multiset=teacher_multiset,
            teacher_multiset_size=len(teacher_multiset),
            teacher_sample_count=len(teacher_samples),
            teacher_invalid_sample_count=len(teacher_samples) - len(teacher_actions),
            teacher_action_kind_disagreement=disagreement,
            teacher_failure=False,
            teacher_error=None,
            matcher_matrix=[],
            state_fingerprint=fingerprint,
            terminal_success=None,
            terminal_reason=None,
            state_group_advanced=False,
        )

    @staticmethod
    def _frequency_sensitive_group(scored) -> bool:
        eligible = [item for item in scored if item.semantic_train_mask]
        if len(eligible) < 2:
            return False
        frequency_rewards = np.asarray([float(item.reward) for item in eligible], dtype=np.float64)
        any_match_rewards = np.where(
            frequency_rewards > 0,
            1.0,
            frequency_rewards,
        )

        def normalized(values):
            if np.ptp(values) <= 1e-8:
                return np.zeros_like(values)
            return (values - values.mean()) / (values.std(ddof=1) + 1e-6)

        advantage_changed = not np.allclose(
            normalized(frequency_rewards),
            normalized(any_match_rewards),
            atol=1e-6,
        )
        frequency_scores = np.asarray([float(item.selection_score) for item in scored], dtype=np.float64)
        any_match_scores = np.asarray(
            [-1.0 if item.action.kind == "invalid" else (1.0 if float(item.reward or 0.0) > 0 else 0.0) for item in scored],
            dtype=np.float64,
        )
        argmax_changed = set(np.flatnonzero(frequency_scores == frequency_scores.max())) != set(np.flatnonzero(any_match_scores == any_match_scores.max()))
        return bool(advantage_changed or argmax_changed)

    async def _matcher_failure_group(
        self,
        *,
        raw_actions: Sequence[str],
        candidates: Sequence[AWMAction],
        prepared: Mapping[str, Any],
        fingerprint: str,
        error: Exception,
    ):
        """Mask an unsupervised group without selecting or executing a candidate."""
        error_text = f"{type(error).__name__}: {error}"
        teacher_samples = prepared["teacher_samples"]
        teacher_multiset = prepared["teacher_multiset"]
        protocol_reward, terminal_payload, _ = await self._verify_and_done(None)
        terminal_metadata = self._terminal_metadata(terminal_payload)
        protocol_reward = float(terminal_metadata["terminal_reward"] or 0.0)
        self._done = True
        candidate_results = []
        for raw, action in zip(raw_actions, candidates, strict=True):
            info = self._annotate(
                raw_action=raw,
                parsed_action=canonical_action(action),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                move_optimal=False,
                legal_non_oracle=False,
                semantic_train_mask=False,
                teacher_frequency=0,
                teacher_multiset=teacher_multiset,
                teacher_multiset_size=len(teacher_multiset),
                teacher_sample_count=len(teacher_samples),
                teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=False,
                teacher_failure=False,
                teacher_error=None,
                matcher_failure=True,
                matcher_error=error_text,
                matcher_matrix=[],
                state_fingerprint=fingerprint,
                protocol_reward=0.0,
                terminal_success=None,
                terminal_reason=None,
                state_group_selection_type="none",
                state_group_random_select_prob=0.0,
                state_group_advanced=False,
            )
            candidate_results.append((self._last_observation, 0.0, False, info))
        failure_info = self._annotate(
            action_kind="matcher_failure",
            protocol_reward=protocol_reward,
            **terminal_metadata,
            semantic_train_mask=False,
            teacher_failure=False,
            matcher_failure=True,
            matcher_error=error_text,
            teacher_sample_count=len(teacher_samples),
            teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
            teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
            state_fingerprint=fingerprint,
            terminal_reason="matcher_failure",
            state_group_selection_type="none",
            state_group_advanced=False,
        )
        return candidate_results, -1, self._last_observation, 0.0, True, failure_info

    async def step_candidate_group(
        self,
        raw_actions: list[str],
        visible_chat: list[dict[str, Any]] | None = None,
    ):
        if self._done:
            raise RuntimeError("cannot score an AWM candidate group after termination")
        self._set_visible_chat(visible_chat)
        candidates = [self._validate(raw) for raw in raw_actions]
        fingerprint = state_fingerprint(
            self._scenario,
            self._task_idx,
            self._chat,
            self._tools,
        )
        prepared = self._prepared_supervision
        self._prepared_supervision = None
        if prepared is None or prepared["state_fingerprint"] != fingerprint:
            raise RuntimeError("AWM student candidates require matching teacher-first preflight")

        teacher_samples = prepared["teacher_samples"]
        teacher_actions = prepared["teacher_actions"]
        teacher_multiset = prepared["teacher_multiset"]
        message_match_counts: dict[int, int] = {}
        matcher_matrix: list[list[bool]] = []
        teacher_messages = [action.content or "" for action in teacher_actions if action.kind == "message"]
        message_positions = [index for index, action in enumerate(candidates) if action.kind == "message"]
        if teacher_messages and message_positions:
            try:
                matched = await self.oracle_actor.match_message_pairs.remote(
                    teacher_messages,
                    [candidates[index].content or "" for index in message_positions],
                )
                if not isinstance(matched, Mapping):
                    raise TypeError("matcher result must be an object")
                counts = matched.get("counts")
                matcher_matrix = matched.get("matrix")
                if not isinstance(counts, list) or len(counts) != len(message_positions):
                    raise ValueError("matcher returned the wrong number of candidate counts")
                if not isinstance(matcher_matrix, list) or len(matcher_matrix) != len(message_positions):
                    raise ValueError("matcher returned the wrong number of matrix rows")
                for count, row in zip(counts, matcher_matrix, strict=True):
                    if not isinstance(row, list) or len(row) != len(teacher_messages) or any(not isinstance(value, bool) for value in row) or isinstance(count, bool) or not isinstance(count, int) or count != sum(row):
                        raise ValueError("matcher returned an invalid pairwise Boolean matrix")
            except Exception as exc:
                return await self._matcher_failure_group(
                    raw_actions=raw_actions,
                    candidates=candidates,
                    prepared=prepared,
                    fingerprint=fingerprint,
                    error=exc,
                )
            message_match_counts = dict(zip(message_positions, counts, strict=True))
        scored = score_candidates(
            candidates,
            teacher_actions,
            message_match_counts=message_match_counts,
        )
        frequency_sensitive = self._frequency_sensitive_group(scored)

        selected_index = select_uniform_argmax([item.selection_score for item in scored], self._rng)
        selected_action = candidates[selected_index]
        protocol_reward, done = await self._execute(raw_actions[selected_index], selected_action)
        runtime_train_mask = bool(self._last_info.get("runtime_train_mask", True))
        penalized_action = canonical_action(selected_action) if self._last_info.get("runtime_policy_error", False) else None
        candidate_results = []
        for index, (raw, action, item) in enumerate(zip(raw_actions, candidates, scored, strict=True)):
            selected = index == selected_index
            runtime_policy_penalty = penalized_action is not None and canonical_action(action) == penalized_action
            reward = -1.0 if runtime_policy_penalty else float(item.reward or 0.0)
            info = self._annotate(
                raw_action=raw,
                parsed_action=canonical_action(action),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=bool(action.kind == "invalid" or runtime_policy_penalty),
                is_action_valid=int(action.kind != "invalid" and not runtime_policy_penalty),
                move_optimal=bool(item.teacher_frequency > 0 and not runtime_policy_penalty),
                legal_non_oracle=bool(action.kind != "invalid" and item.teacher_frequency == 0 and not runtime_policy_penalty),
                semantic_train_mask=bool(item.semantic_train_mask and runtime_train_mask),
                tool_calling=int(action.kind == "tool"),
                runtime_policy_penalty=runtime_policy_penalty,
                teacher_frequency=item.teacher_frequency,
                teacher_multiset=teacher_multiset,
                teacher_multiset_size=len(teacher_multiset),
                teacher_sample_count=len(teacher_samples),
                teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=frequency_sensitive,
                teacher_failure=False,
                teacher_error=None,
                matcher_matrix=matcher_matrix,
                state_fingerprint=fingerprint,
                protocol_reward=protocol_reward if selected else 0.0,
                awm_reward_type=(self._last_info.get("awm_reward_type") if selected and done else None),
                awm_verify_result=(self._last_info.get("awm_verify_result") if selected and done else None),
                terminal_label=(self._last_info.get("terminal_label") if selected and done else None),
                terminal_reward=(self._last_info.get("terminal_reward") if selected and done else None),
                terminal_outcome_valid=bool(selected and done and self._last_info.get("terminal_outcome_valid", False)),
                outcome_train_mask=True,
                terminal_success=(self._last_info.get("terminal_success") if selected and done else None),
                terminal_reason=(self._last_info.get("terminal_reason") if selected and done else None),
                terminal_judge_result=(self._last_info.get("terminal_judge_result") if selected and done else None),
                terminal_judge_error=(self._last_info.get("terminal_judge_error") if selected and done else None),
                state_group_selection_type="uniform_argmax",
                state_group_random_select_prob=0.0,
                state_group_advanced=bool(selected and runtime_train_mask),
            )
            candidate_results.append((self._last_observation, reward, bool(done and selected), info))
        selected_info = candidate_results[selected_index][3]
        return (
            candidate_results,
            selected_index,
            self._last_observation,
            (float(candidate_results[selected_index][1]) if runtime_train_mask else 0.0),
            done,
            selected_info,
        )

    def current_observation_info(self):
        return self._observation_info()

    async def close(self):
        await self._close_env()


class AWMVectorEnv:
    def __init__(self, workers, seeds, runtime_recorder=None):
        if len(workers) != len(seeds):
            raise ValueError("AWM workers and seeds must align")
        self.workers = workers
        self.seeds = seeds
        self.runtime_recorder = runtime_recorder
        self._episode = 0

    def _validate_rows(self, kwargs):
        if kwargs is None or len(kwargs) != len(self.workers):
            raise ValueError(f"expected {len(self.workers)} AWM env kwargs")
        for row in kwargs:
            if not row.get("scenario"):
                raise ValueError("AWM row is missing scenario")
            if "task_idx" not in row:
                raise ValueError("AWM row is missing task_idx")

    def reset(self, kwargs=None):
        self._validate_rows(kwargs)
        offset = self._episode * 100003
        self._episode += 1
        results = ray.get(
            [
                worker.reset.remote(
                    scenario=row["scenario"],
                    task_idx=int(row["task_idx"]),
                    seed=seed + offset,
                )
                for worker, row, seed in zip(self.workers, kwargs, self.seeds, strict=True)
            ]
        )
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions):
        results = ray.get([worker.step.remote(action) for worker, action in zip(self.workers, actions, strict=True)])
        return (
            [result[0] for result in results],
            np.asarray([result[1] for result in results], dtype=np.float32),
            np.asarray([result[2] for result in results], dtype=bool),
            [result[3] for result in results],
        )

    def prepare_state_groups(
        self,
        *,
        active_indices,
        visible_chats,
    ):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(visible_chats):
            raise ValueError("active_indices must align with AWM visible chats")
        return ray.get([self.workers[index].prepare_state_group.remote(visible_chat) for index, visible_chat in zip(indices, visible_chats, strict=True)])

    def terminate_context_overflows(self, *, active_indices, diagnostics):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(diagnostics):
            raise ValueError("active_indices must align with AWM context diagnostics")
        return ray.get([self.workers[index].terminate_context_overflow.remote(item) for index, item in zip(indices, diagnostics, strict=True)])

    def step_candidate_groups(
        self,
        candidate_action_groups,
        active_indices=None,
        visible_chats=None,
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        indices = [int(index) for index in active_indices]
        if len(indices) != len(candidate_action_groups):
            raise ValueError("active_indices must align with AWM candidate groups")
        if visible_chats is None:
            visible_chats = [None] * len(candidate_action_groups)
        if len(visible_chats) != len(candidate_action_groups):
            raise ValueError("visible_chats must align with AWM candidate groups")
        results = ray.get([self.workers[index].step_candidate_group.remote(group, visible_chat=visible_chat) for index, group, visible_chat in zip(indices, candidate_action_groups, visible_chats, strict=True)])
        return (
            [result[0] for result in results],
            np.asarray([result[1] for result in results], dtype=np.int32),
            [result[2] for result in results],
            np.asarray([result[3] for result in results], dtype=np.float32),
            np.asarray([result[4] for result in results], dtype=bool),
            [result[5] for result in results],
        )

    def close(self):
        ray.get([worker.close.remote() for worker in self.workers])
        for worker in self.workers:
            ray.kill(worker)
        if self.runtime_recorder is not None:
            ray.kill(self.runtime_recorder)


def build_awm_envs(
    *,
    seed: int,
    count: int,
    env_config,
    is_train: bool,
    group_n: int,
    oracle_actor=None,
):
    awm = env_config.awm
    max_steps = int(awm.train_max_steps if is_train else awm.eval_max_steps)
    worker_options = dict(getattr(env_config, "resources_per_worker", {}) or {})
    worker_factory = AWMWorker.options(**worker_options) if worker_options else AWMWorker

    reward_mode = str(awm.reward_mode)
    runtime_recorder = None
    runtime_config = getattr(awm, "runtime_failures", None)
    judge_config = getattr(runtime_config, "judge", None) if runtime_config is not None else None
    terminal_config = awm.terminal_judge
    runtime_judge_enabled = bool(is_train and runtime_config is not None and bool(getattr(runtime_config, "enabled", False)) and judge_config is not None and judge_config.enabled)
    if is_train and runtime_config is not None and bool(getattr(runtime_config, "enabled", False)):
        runtime_recorder = AWMRuntimeFailureRecorder.remote(str(runtime_config.path))
    workers = []
    seeds = []
    for index in range(int(count) * int(group_n)):
        worker_seed = int(seed) + index
        workers.append(
            worker_factory.remote(
                base_url=str(awm.base_url),
                max_steps=max_steps,
                history_window=int(awm.history_window),
                verifier_mode=str(awm.verifier_mode),
                reward_mode=reward_mode,
                oracle_actor=oracle_actor,
                runtime_recorder=runtime_recorder,
                seed=worker_seed,
                runtime_judge_enabled=runtime_judge_enabled,
                runtime_judge_confidence_threshold=int(getattr(judge_config, "confidence_threshold", 80)),
                terminal_judge_api_base=str(terminal_config.api_base),
                terminal_judge_api_key_env=str(terminal_config.api_key_env),
                terminal_judge_model=str(terminal_config.model),
            )
        )
        seeds.append(worker_seed)
    return AWMVectorEnv(workers, seeds, runtime_recorder=runtime_recorder)
