"""Ray workers for AgentWorldModel semantic and outcome training."""

from __future__ import annotations

import json
import random
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
from .oracle import build_expert_messages

AWM_OPENENV_COMMIT = "5298e0d91c6cd55d5f3a81259d5b2a9a1e05eff0"
AWM_DATASET_REVISION = "dde80a0283fe781bdc51656bce57063dc5650213"
AWM_DATASET_NAME = "Snowflake/AgentWorldModel-1K"
AWM_PROTOCOL_VERSION = 7


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
        seed: int = 0,
    ):
        if reward_mode not in {"semantic", "outcome"}:
            raise ValueError(f"unsupported AWM reward_mode: {reward_mode}")
        if verifier_mode not in {"code", "sql"}:
            raise ValueError(f"unsupported AWM verifier_mode: {verifier_mode}")
        if int(history_window) != 3:
            raise ValueError("AWM protocol requires history_window=3")
        self.base_url = str(base_url)
        self.max_steps = int(max_steps)
        self.history_window = int(history_window)
        self.verifier_mode = verifier_mode
        self.reward_mode = reward_mode
        self.oracle_actor = oracle_actor
        self.seed = int(seed)
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
            raise RuntimeError("AgentWorldModel OpenEnv is not installed. Run examples/awm/scripts/install_awm.sh first.") from exc
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
            **self._last_info,
        }
        result.update(updates)
        return result

    def _observation_info(self) -> tuple[str, dict[str, Any]]:
        return self._last_observation, self._annotate()

    async def reset(self, *, scenario: str, task_idx: int, seed: int | None = None):
        await self._close_env()
        actual_seed = self.seed if seed is None else int(seed)
        self._rng.seed(actual_seed)
        self._env = await self._new_env()
        reset_result = await self._env.reset(
            scenario=str(scenario),
            task_idx=int(task_idx),
            seed=actual_seed,
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
        self._last_info = {
            "native_direct_tools": True,
        }
        return self._observation_info()

    async def _call_tool(self, action: AWMAction) -> tuple[str, dict[str, Any]]:
        from openenv.core.env_server.mcp_types import CallToolAction

        result = await self._env.step(CallToolAction(tool_name=action.name or "", arguments=action.arguments or {}))
        return _tool_response_text(result), _observation_dict(result)

    async def _verify_and_done(self, final_answer: str | None) -> tuple[float, dict[str, Any]]:
        from openenv.core.env_server.mcp_types import CallToolAction

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
        await self._env.step(CallToolAction(tool_name="done", arguments={}))
        return reward, verify_payload

    async def _execute(self, raw_action: str, action: AWMAction):
        self._step += 1
        protocol_reward = 0.0
        terminal_success: bool | None = None
        terminal_reason: str | None = None
        environment_payload: dict[str, Any] = {}
        if action.kind == "tool":
            response, environment_payload = await self._call_tool(action)
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
            protocol_reward, environment_payload = await self._verify_and_done(action.content)
            self._done = True
            terminal_success = environment_payload.get("reward_type") == "complete"
            terminal_reason = "final_response"
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

        if not self._done and self._step >= self.max_steps:
            protocol_reward, environment_payload = await self._verify_and_done(None)
            self._done = True
            terminal_success = environment_payload.get("reward_type") == "complete"
            terminal_reason = "decision_limit"
        self._last_info = {
            "awm_reward_type": environment_payload.get("reward_type"),
            "awm_verify_result": environment_payload.get("verify_result"),
            "protocol_reward": protocol_reward,
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
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
        reward = protocol_reward if self.reward_mode == "outcome" else 0.0
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
            return False, self._annotate(
                action_kind="teacher_failure",
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
                terminal_success=None,
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

    def _matcher_failure_group(
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
            semantic_train_mask=False,
            teacher_failure=False,
            matcher_failure=True,
            matcher_error=error_text,
            teacher_sample_count=len(teacher_samples),
            teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
            teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
            state_fingerprint=fingerprint,
            terminal_success=None,
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
                return self._matcher_failure_group(
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
        candidate_results = []
        for index, (raw, action, item) in enumerate(zip(raw_actions, candidates, scored, strict=True)):
            selected = index == selected_index
            reward = float(item.reward) if item.reward is not None else 0.0
            info = self._annotate(
                raw_action=raw,
                parsed_action=canonical_action(action),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                move_optimal=bool(item.teacher_frequency > 0),
                legal_non_oracle=bool(action.kind != "invalid" and item.teacher_frequency == 0),
                semantic_train_mask=bool(item.semantic_train_mask),
                tool_calling=int(action.kind == "tool"),
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
                terminal_success=(self._last_info.get("terminal_success") if selected and done else None),
                terminal_reason=(self._last_info.get("terminal_reason") if selected and done else None),
                state_group_selection_type="uniform_argmax",
                state_group_random_select_prob=0.0,
                state_group_advanced=selected,
            )
            candidate_results.append((self._last_observation, reward, bool(done and selected), info))
        selected_info = candidate_results[selected_index][3]
        return (
            candidate_results,
            selected_index,
            self._last_observation,
            float(scored[selected_index].reward or 0.0),
            done,
            selected_info,
        )

    def current_observation_info(self):
        return self._observation_info()

    async def close(self):
        await self._close_env()


class AWMVectorEnv:
    def __init__(self, workers, seeds):
        if len(workers) != len(seeds):
            raise ValueError("AWM workers and seeds must align")
        self.workers = workers
        self.seeds = seeds
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
    reward_mode = str(awm.reward_mode)
    workers = []
    seeds = []
    for index in range(int(count) * int(group_n)):
        worker_seed = int(seed) + index
        workers.append(
            AWMWorker.remote(
                base_url=str(awm.base_url),
                max_steps=max_steps,
                history_window=int(awm.history_window),
                verifier_mode=str(awm.verifier_mode),
                reward_mode=reward_mode,
                oracle_actor=oracle_actor,
                seed=worker_seed,
            )
        )
        seeds.append(worker_seed)
    return AWMVectorEnv(workers, seeds)
