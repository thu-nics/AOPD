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
    AWMRuntimeFailureRecorder,
    deterministic_error_signature,
    infrastructure_error,
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


def test_error_signature_accepts_only_strong_infrastructure_evidence():
    server_error = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    assert deterministic_error_signature(server_error, phase="tool", tool_name="broken")
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
    first = {"protocol_version": 2, "task_id": "scenario:0", "status": "masked"}
    path.write_text(json.dumps(first) + '\n{"task_id":')
    recorder = recorder_class(str(path))
    recorder.record({"task_id": "scenario:0", "status": "masked"})
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [record["status"] for record in records] == ["masked", "masked"]
    assert recorder.stats() == {"runtime_failure_records": 2}


def test_recorder_normalizes_valid_final_record_without_newline(tmp_path):
    recorder_class = AWMRuntimeFailureRecorder.__ray_metadata__.modified_class
    path = tmp_path / "runtime_failures.jsonl"
    first = {"protocol_version": 2, "task_id": "scenario:0", "status": "masked"}
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
