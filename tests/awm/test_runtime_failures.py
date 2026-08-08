import asyncio
import json

import pytest

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    build_native_chat,
    normalize_tools,
    state_fingerprint,
)
from agent_system.environments.env_package.awm.runtime.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime.failures import (
    RUNTIME_FAILURE_PROTOCOL_VERSION,
    AWMRuntimeFailureRecorder,
    deterministic_error_signature,
    infrastructure_error,
    judgeable_tool_error,
)


def _worker():
    worker_class = AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        history_window=6,
        verifier_mode="code",
        reward_mode="semantic",
    )
    worker._scenario = "scenario"
    worker._task_idx = 2
    worker._task = "Finish the task"
    worker._actual_seed = 17
    worker._tools = normalize_tools(
        [
            {
                "name": name,
                "description": name,
                "inputSchema": {"type": "object", "properties": {}},
            }
            for name in ("lookup", "broken")
        ]
    )
    worker._chat = build_native_chat(worker._task)
    return worker


class _RemoteVerdict:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)

        async def resolve():
            return dict(self.verdict)

        return resolve()


def _enable_judge(worker, verdict):
    remote = _RemoteVerdict(verdict)
    worker.runtime_judge_enabled = True
    worker.oracle_actor = type("FakeActor", (), {"classify_runtime_failure": remote})()
    return remote


def test_error_signature_accepts_only_strong_infrastructure_evidence():
    server_error = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    assert deterministic_error_signature(server_error, phase="tool", tool_name="broken")
    assert judgeable_tool_error(server_error, phase="tool")
    assert not judgeable_tool_error(server_error, phase="verify")
    route_collision = {
        "reward_type": "server_error",
        "error": 'Status code: 422. Response: {"detail":[{"loc":["path","item_id"],"input":"search-by-name"}]}',
    }
    assert deterministic_error_signature(route_collision, phase="tool", tool_name="broken")
    ordinary_bad_argument = {
        "reward_type": "invalid_args",
        "error": "Input should be a valid integer",
    }
    assert deterministic_error_signature(ordinary_bad_argument, phase="tool", tool_name="broken") is None
    assert not infrastructure_error(ordinary_bad_argument, phase="tool")
    ordinary_http_4xx = {
        "reward_type": "server_error",
        "error": "Error calling lookup. Status code: 404. Response: Not Found",
    }
    assert deterministic_error_signature(ordinary_http_4xx, phase="tool", tool_name="lookup") is None
    assert not infrastructure_error(ordinary_http_4xx, phase="tool")
    assert infrastructure_error(
        {"reward_type": "runtime_exception", "error": "done failed"},
        phase="done",
    )


def test_done_failure_uses_the_same_current_group_mask():
    worker = _worker()
    result = asyncio.run(
        worker._classify_runtime_failure(
            phase="done",
            payload={"reward_type": "runtime_exception", "error": "done failed"},
        )
    )
    assert result["status"] == "masked"
    assert result["signature"] is None


def test_strong_runtime_failure_is_directly_masked_without_replay():
    worker = _worker()
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    result = asyncio.run(
        worker._classify_runtime_failure(
            phase="tool",
            payload=payload,
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )
    assert result["status"] == "masked"
    assert result["signature"] == deterministic_error_signature(payload, phase="tool", tool_name="broken")


@pytest.mark.parametrize(
    ("verdict", "expected_status", "expected_policy"),
    [
        (
            {
                "error_class": "policy_execution_error",
                "classification_confidence": 95,
                "post_error_state": "unchanged",
                "rationale": "CHECK failed before commit",
            },
            "policy_penalized_continued",
            True,
        ),
        (
            {
                "error_class": "policy_execution_error",
                "classification_confidence": 79,
                "post_error_state": "unchanged",
                "rationale": "low confidence",
            },
            "masked",
            False,
        ),
        (
            {
                "error_class": "policy_execution_error",
                "classification_confidence": 95,
                "post_error_state": "possibly_mutated",
                "rationale": "a prior commit may have succeeded",
            },
            "policy_penalized_terminated",
            True,
        ),
        (
            {
                "error_class": "infrastructure_error",
                "classification_confidence": 95,
                "post_error_state": "unchanged",
                "rationale": "route defect",
            },
            "masked",
            False,
        ),
        (
            {
                "error_class": "uncertain",
                "classification_confidence": 80,
                "post_error_state": "unknown",
                "rationale": "insufficient evidence",
            },
            "masked",
            False,
        ),
    ],
)
def test_5xx_judge_only_penalizes_high_confidence_policy_errors(
    verdict,
    expected_status,
    expected_policy,
):
    worker = _worker()
    remote = _enable_judge(worker, verdict)
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }

    result = asyncio.run(
        worker._classify_runtime_failure(
            phase="tool",
            payload=payload,
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert result["status"] == expected_status
    assert result["runtime_policy_error"] is expected_policy
    assert result["runtime_judge"]["rationale"] == verdict["rationale"]
    assert len(remote.calls) == 1


def test_policy_error_with_unchanged_state_keeps_error_observation_and_continues():
    worker = _worker()
    _enable_judge(
        worker,
        {
            "error_class": "policy_execution_error",
            "classification_confidence": 95,
            "post_error_state": "unchanged",
            "rationale": "CHECK failed before commit",
        },
    )
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }

    async def call_tool(action):
        return '{"error":"HTTP 500"}', payload

    worker._call_tool = call_tool
    _, done = asyncio.run(
        worker._execute(
            '<tool_call>{"name":"broken","arguments":{}}</tool_call>',
            AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert done is False
    assert worker._last_info["runtime_train_mask"] is True
    assert worker._last_info["runtime_policy_error"] is True
    assert worker._last_info["runtime_policy_continued"] is True
    assert worker._chat[-1]["role"] == "tool"
    assert "HTTP 500" in worker._chat[-1]["content"]


def test_policy_error_with_possible_mutation_trains_group_but_terminates():
    worker = _worker()
    _enable_judge(
        worker,
        {
            "error_class": "policy_execution_error",
            "classification_confidence": 95,
            "post_error_state": "possibly_mutated",
            "rationale": "a write may have committed before serialization failed",
        },
    )
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }

    async def call_tool(action):
        return '{"error":"HTTP 500"}', payload

    worker._call_tool = call_tool
    _, done = asyncio.run(
        worker._execute(
            '<tool_call>{"name":"broken","arguments":{}}</tool_call>',
            AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert done is True
    assert worker._last_info["runtime_train_mask"] is True
    assert worker._last_info["runtime_policy_error"] is True
    assert worker._last_info["runtime_policy_terminated"] is True
    assert worker._last_info["terminal_reason"] == "runtime_policy_penalized_terminated"


def test_runtime_judge_exception_fails_closed_to_group_mask():
    worker = _worker()

    class FailingRemote:
        def remote(self, **kwargs):
            async def fail():
                raise RuntimeError("judge unavailable")

            return fail()

    worker.runtime_judge_enabled = True
    worker.oracle_actor = type(
        "FailingActor",
        (),
        {"classify_runtime_failure": FailingRemote()},
    )()
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }

    result = asyncio.run(
        worker._classify_runtime_failure(
            phase="tool",
            payload=payload,
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert result["status"] == "masked"
    assert result["runtime_policy_error"] is False
    assert result["runtime_judge"] is None
    assert result["runtime_judge_error"] == "RuntimeError: judge unavailable"


def test_runtime_failure_masks_only_current_group_and_does_not_advance():
    worker = _worker()
    action = AWMAction(kind="tool", name="broken", arguments={})
    teacher_samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    fingerprint = state_fingerprint(worker._scenario, worker._task_idx, worker._chat, worker._tools)
    worker._prepared_supervision = {
        "state_fingerprint": fingerprint,
        "teacher_samples": teacher_samples,
        "teacher_actions": [action] * 3,
        "teacher_multiset": [action.to_dict()] * 3,
        "teacher_invalid_sample_count": 0,
        "teacher_action_kind_disagreement": False,
    }

    async def execute(raw_action, selected_action):
        worker._done = True
        worker._last_info = {
            "runtime_train_mask": False,
            "runtime_failure": True,
            "terminal_success": None,
            "terminal_reason": "runtime_masked",
        }
        return 0.0, True

    worker._execute = execute
    raw = '<tool_call>{"name":"broken","arguments":{}}</tool_call>'
    candidate_results, selected_index, _, reward, done, selected_info = asyncio.run(worker.step_candidate_group([raw] * 4))
    assert selected_index in range(4)
    assert reward == 0.0
    assert done is True
    assert selected_info["runtime_failure"] is True
    assert selected_info["state_group_advanced"] is False
    assert all(item[3]["semantic_train_mask"] is False for item in candidate_results)
    assert all(item[3]["runtime_train_mask"] is False for item in candidate_results)


def test_policy_execution_error_overrides_all_identical_candidate_rewards():
    worker = _worker()
    failed_action = AWMAction(kind="tool", name="broken", arguments={})
    teacher_samples = [{"sample_index": index, "action": failed_action.to_dict()} for index in range(3)]
    fingerprint = state_fingerprint(
        worker._scenario,
        worker._task_idx,
        worker._chat,
        worker._tools,
    )
    worker._prepared_supervision = {
        "state_fingerprint": fingerprint,
        "teacher_samples": teacher_samples,
        "teacher_actions": [failed_action] * 3,
        "teacher_multiset": [failed_action.to_dict()] * 3,
        "teacher_invalid_sample_count": 0,
        "teacher_action_kind_disagreement": False,
    }

    async def execute(raw_action, selected_action):
        assert selected_action == failed_action
        worker._last_info = {
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": True,
            "runtime_policy_continued": True,
            "terminal_success": None,
            "terminal_reason": None,
        }
        return 0.0, False

    worker._execute = execute
    broken = '<tool_call>{"name":"broken","arguments":{}}</tool_call>'
    lookup = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    malformed = "<tool_call>{bad json}</tool_call>"
    candidate_results, selected_index, _, reward, done, _ = asyncio.run(
        worker.step_candidate_group(
            [broken, broken, lookup, malformed],
        )
    )

    assert selected_index in {0, 1}
    assert [item[1] for item in candidate_results] == [-1.0, -1.0, 0.0, -1.0]
    assert [item[3]["runtime_policy_penalty"] for item in candidate_results] == [True, True, False, False]
    assert all(item[3]["semantic_train_mask"] for item in candidate_results)
    assert reward == -1.0
    assert done is False


def test_context_overflow_terminates_only_the_state_without_runtime_failure():
    worker = _worker()
    info = asyncio.run(
        worker.terminate_context_overflow(
            {
                "context_prompt_tokens": 28050,
                "context_max_prompt_tokens": 27904,
                "context_excess_tokens": 146,
                "context_overflow_component": "newest_complete_exchange",
            }
        )
    )
    assert worker._done is True
    assert info["action_kind"] == "context_overflow"
    assert info["terminal_reason"] == "context_budget_exceeded"
    assert info["semantic_train_mask"] is False
    assert info["runtime_train_mask"] is False
    assert info["runtime_failure"] is False


def test_recorder_repairs_torn_tail_and_never_deduplicates_tasks(tmp_path):
    recorder_class = AWMRuntimeFailureRecorder.__ray_metadata__.modified_class
    path = tmp_path / "runtime_failures.jsonl"
    first = {"protocol_version": RUNTIME_FAILURE_PROTOCOL_VERSION, "task_id": "scenario:0", "status": "masked"}
    path.write_text(json.dumps(first) + '\n{"task_id":')
    recorder = recorder_class(str(path))
    recorder.record({"task_id": "scenario:0", "status": "policy_penalized_continued"})
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [record["status"] for record in records] == [
        "masked",
        "policy_penalized_continued",
    ]
    assert recorder.stats()["runtime_failure_records"] == 2
    assert recorder.stats()["runtime_failure_masked"] == 1
    assert recorder.stats()["runtime_failure_policy_penalized_continued"] == 1


def test_recorder_normalizes_valid_final_record_without_newline(tmp_path):
    recorder_class = AWMRuntimeFailureRecorder.__ray_metadata__.modified_class
    path = tmp_path / "runtime_failures.jsonl"
    first = {"protocol_version": RUNTIME_FAILURE_PROTOCOL_VERSION, "task_id": "scenario:0", "status": "masked"}
    path.write_text(json.dumps(first))

    recorder = recorder_class(str(path))
    recorder.record({"task_id": "scenario:1", "status": "masked"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["task_id"] for record in records] == ["scenario:0", "scenario:1"]


def test_recorder_rejects_newline_terminated_invalid_json(tmp_path):
    recorder_class = AWMRuntimeFailureRecorder.__ray_metadata__.modified_class
    path = tmp_path / "runtime_failures.jsonl"
    path.write_text('{"task_id":\n')

    with pytest.raises(RuntimeError, match="invalid AWM runtime-failure JSONL record 1"):
        recorder_class(str(path))
