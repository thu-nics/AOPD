import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    build_native_chat,
    canonical_action,
    normalize_tools,
)
from agent_system.environments.env_package.awm.runtime.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime.manager import (
    AWMEnvironmentManager,
    awm_projection,
)
from agent_system.multi_turn_rollout.rollout_loop import (
    _awm_preflight_failure_summary,
)


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Oracle:
    def __init__(self, *, samples=None, sample_error=None, matcher_error=None):
        def sample(**kwargs):
            if sample_error is not None:
                raise sample_error
            return samples

        def match(teacher_messages, candidate_messages):
            if matcher_error is not None:
                raise matcher_error
            return {
                "counts": [0] * len(candidate_messages),
                "matrix": [[False] * len(teacher_messages) for _ in candidate_messages],
            }

        self.sample_multiset = _RemoteMethod(sample)
        self.match_message_pairs = _RemoteMethod(match)


def _worker(oracle, **worker_kwargs):
    worker_class = AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        max_history_exchanges=3,
        verifier_mode="sql",
        reward_mode="semantic",
        oracle_actor=oracle,
        **worker_kwargs,
    )
    worker._scenario = "scenario"
    worker._task_idx = 0
    worker._task = "Finish the task"
    worker._tools = normalize_tools(
        [
            {
                "name": "lookup",
                "description": "Lookup one record.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}},
                    "required": ["item_id"],
                },
            }
        ]
    )
    worker._chat = build_native_chat(worker._task)
    return worker


def test_teacher_failure_does_not_generate_or_advance_state():
    worker = _worker(_Oracle(sample_error=RuntimeError("API unavailable")))
    original_chat = list(worker._chat)

    ready, info = asyncio.run(worker.prepare_teacher_supervision())

    assert ready is False
    assert info["teacher_failure"] is True
    assert info["state_group_advanced"] is False
    assert worker._step == 0
    assert worker._chat == original_chat
    assert worker._prepared_teacher_supervision is None


def test_preflight_failure_summary_groups_environment_and_exact_errors():
    summary = _awm_preflight_failure_summary(
        [
            [
                {
                    "agentic_env_family": "awm",
                    "action_kind": "teacher_failure",
                    "teacher_error": "RuntimeError: provider unavailable",
                }
            ],
            [
                {
                    "agentic_env_family": "envscaler",
                    "action_kind": "teacher_failure",
                    "teacher_error": "RuntimeError: provider unavailable",
                }
            ],
            [
                {
                    "agentic_env_family": "awm",
                    "action_kind": "context_overflow",
                }
            ],
            [],
        ]
    )

    assert summary == {
        "failed_states": 3,
        "by_environment": {"awm": 2, "envscaler": 1},
        "by_failure_kind": {
            "context_overflow": 1,
            "teacher_failure": 2,
        },
        "teacher_errors": [{"count": 2, "error": "RuntimeError: provider unavailable"}],
    }


def test_matcher_failure_happens_before_environment_advancement():
    message = AWMAction(kind="message", content="The task is complete.")
    samples = [{"sample_index": index, "action": message.to_dict()} for index in range(3)]
    worker = _worker(
        _Oracle(
            samples=samples,
            matcher_error=RuntimeError("matcher unavailable"),
        )
    )
    original_chat = list(worker._chat)
    ready, _ = asyncio.run(worker.prepare_teacher_supervision())
    assert ready is True

    result = asyncio.run(worker.step_candidate_group(["candidate"] * 4))
    candidate_results, selected_index, _, reward, rollout_done, failure_info = result

    assert selected_index == -1
    assert reward == 0.0
    assert rollout_done is True
    assert failure_info["matcher_failure"] is True
    assert failure_info["state_group_advanced"] is False
    assert all(item[3]["semantic_train_mask"] is False for item in candidate_results)
    assert all(item[3]["state_group_advanced"] is False for item in candidate_results)
    assert worker._step == 0
    assert worker._chat == original_chat


def test_only_selected_candidate_carries_terminal_judge_metadata():
    action = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    worker = _worker(_Oracle(samples=samples))
    ready, _ = asyncio.run(worker.prepare_teacher_supervision())
    assert ready is True

    async def terminate(_raw_action, _action):
        worker._last_info = {
            "protocol_reward": 1.0,
            "terminal_success": True,
            "terminal_reason": "final_response",
            "awm_reward_type": "complete",
            "awm_verify_result": {"result": "ok"},
            "terminal_label": "complete",
            "terminal_reward": 1.0,
            "terminal_outcome_valid": True,
            "outcome_train_mask": True,
            "terminal_judge_result": {"classification": "complete"},
            "terminal_judge_error": None,
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": False,
        }
        worker._done = True
        return 1.0, True

    worker._execute = terminate
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>'
    candidate_results, selected_index, *_ = asyncio.run(worker.step_candidate_group([raw] * 4))

    for index, (_, _, done, info) in enumerate(candidate_results):
        if index == selected_index:
            assert done is True
            assert info["terminal_label"] == "complete"
            assert info["terminal_judge_result"] == {"classification": "complete"}
        else:
            assert done is False
            assert info["terminal_label"] is None
            assert info["terminal_reward"] is None
            assert info["terminal_outcome_valid"] is False
            assert info["terminal_judge_result"] is None


def test_third_identical_no_progress_call_caps_positive_semantic_reward():
    action = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    worker = _worker(
        _Oracle(samples=samples),
        repeat_reward_cap_enabled=True,
        repeat_reward_cap_min_streak=3,
        repeat_reward_cap_value=0.0,
    )
    canonical = canonical_action(action)
    for _ in range(2):
        worker._no_progress.record(
            action_kind="tool",
            canonical_action=canonical,
            observation="same observation",
        )
    worker._last_selected_canonical_action = canonical
    ready, _ = asyncio.run(worker.prepare_teacher_supervision())
    assert ready is True

    async def execute(_raw_action, _action):
        worker._last_observation = "same observation"
        worker._last_info = {
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": False,
            "terminal_success": None,
            "terminal_reason": None,
        }
        return 0.0, False

    worker._execute = execute
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>'
    candidate_results, selected_index, *_ = asyncio.run(worker.step_candidate_group([raw] * 4))

    assert selected_index in range(4)
    for _, reward, done, info in candidate_results:
        assert reward == 0.0
        assert done is False
        assert info["raw_semantic_reward"] > 0.0
        assert info["repeat_reward_capped"] is True
        assert info["prospective_no_progress_repeat"] is True
        assert info["semantic_train_mask"] is True


def test_fourth_identical_no_progress_call_terminates_after_trainable_group():
    action = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    worker = _worker(
        _Oracle(samples=samples),
        repeat_reward_cap_enabled=True,
        repeat_reward_cap_min_streak=3,
        repeat_reward_cap_value=0.0,
        repeat_termination_enabled=True,
        repeat_termination_max_streak=4,
    )
    canonical = canonical_action(action)
    for _ in range(3):
        worker._no_progress.record(
            action_kind="tool",
            canonical_action=canonical,
            observation="same observation",
        )
    worker._last_selected_canonical_action = canonical
    ready, _ = asyncio.run(worker.prepare_teacher_supervision())
    assert ready is True

    async def execute(_raw_action, _action):
        worker._last_observation = "same observation"
        worker._last_info = {
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": False,
            "terminal_success": None,
            "terminal_reason": None,
        }
        return 0.0, False

    async def verify(_final_answer):
        return 0.0, {"reward_type": "incomplete"}, {"status": "normal"}

    worker._execute = execute
    worker._verify_and_done = verify
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>'
    candidate_results, selected_index, _, _, rollout_done, failure_info = asyncio.run(worker.step_candidate_group([raw] * 4))

    assert rollout_done is True
    assert failure_info["terminal_reason"] == "no_progress_repeat_limit"
    assert worker._no_progress.repeat_streak == 4
    for index, (_, reward, done, info) in enumerate(candidate_results):
        assert reward == 0.0
        assert info["semantic_train_mask"] is True
        if index == selected_index:
            assert done is True
            assert info["terminal_reason"] == "no_progress_repeat_limit"
        else:
            assert done is False
            assert info["terminal_reason"] is None


def test_teacher_and_candidates_share_truncated_view_without_losing_history():
    action = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    worker = _worker(_Oracle(samples=samples))
    worker._chat.extend(
        [
            {"role": "assistant", "content": "old action"},
            {"role": "user", "content": "old result"},
            {"role": "assistant", "content": "new action"},
            {"role": "user", "content": "new result"},
        ]
    )
    logical_chat = list(worker._chat)
    visible_chat = [*worker._chat[:2], *worker._chat[-2:]]

    ready, _ = asyncio.run(worker.prepare_teacher_supervision(visible_chat))
    assert ready is True

    async def execute(_raw_action, _action):
        worker._last_info = {
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": False,
            "terminal_success": None,
            "terminal_reason": None,
        }
        return 0.0, False

    worker._execute = execute
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>'
    result = asyncio.run(worker.step_candidate_group([raw] * 4, visible_chat=visible_chat))

    assert result[1] in range(4)
    assert worker._chat == logical_chat


def test_awm_rejects_privileged_teacher_context_explicitly():
    worker_class = AWMWorker.__ray_metadata__.modified_class
    with pytest.raises(ValueError, match="does not expose"):
        worker_class(
            base_url="unused",
            max_steps=20,
            verifier_mode="sql",
            reward_mode="semantic",
            use_privileged_teacher_context=True,
        )


class _ScheduleVector:
    def __init__(self):
        self.schedule_step = None

    def reset(self, *, kwargs, schedule_step):
        self.schedule_step = schedule_step
        infos = [{"observation": "task", "chat": [], "tools": []} for _ in kwargs]
        return ["task"] * len(kwargs), infos


def test_manager_enforces_task_level_resume_coordinates():
    vector = _ScheduleVector()
    config = SimpleNamespace(
        env=SimpleNamespace(
            rollout=SimpleNamespace(current_step=6),
        )
    )
    manager = AWMEnvironmentManager(vector, awm_projection, config)
    rows = np.asarray(
        [{"schedule_step": 5, "schedule_slot": slot} for slot in range(2)],
        dtype=object,
    )

    observations, _ = manager.reset(rows)

    assert vector.schedule_step == 5
    assert observations["text"] == ["task", "task"]
    with pytest.raises(RuntimeError, match="requires schedule_step=5"):
        manager.reset([{"schedule_step": 4, "schedule_slot": slot} for slot in range(2)])
    with pytest.raises(RuntimeError, match="ordered zero-based"):
        manager.reset(
            [
                {"schedule_step": 5, "schedule_slot": 1},
                {"schedule_step": 5, "schedule_slot": 0},
            ]
        )
