import asyncio
import json

from agent_system.environments.env_package.awm.actions import (
    AWMAction,
    build_native_chat,
    normalize_tools,
    state_fingerprint,
)
from agent_system.environments.env_package.awm.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime_failures import (
    AWMRuntimeFailureRecorder,
    deterministic_error_signature,
    infrastructure_error,
    replay_observation_signature,
)


class _Result:
    def __init__(self, payload, reward=0.0):
        self.observation = payload
        self.reward = reward


class _ReplayEnv:
    def __init__(self, tools, failing_payload, prefix_payload=None):
        self.tools = tools
        self.failing_payload = failing_payload
        self.prefix_payload = prefix_payload or {
            "reward_type": "tool_call_ok",
            "tool_result": {"id": 1},
        }
        self.closed = False

    async def reset(self, **kwargs):
        return _Result(
            {
                "reward_type": "reset_ok",
                "scenario": kwargs["scenario"],
                "task_idx": kwargs["task_idx"],
            }
        )

    async def list_tools(self, use_cache=False):
        assert use_cache is False
        return self.tools

    async def step(self, action):
        if action.tool_name == "lookup":
            return _Result(self.prefix_payload)
        assert action.tool_name == "broken"
        return _Result(self.failing_payload)

    async def __aexit__(self, *args):
        self.closed = True


def _worker():
    worker_class = AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        history_window=3,
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
    lookup_payload = {"reward_type": "tool_call_ok", "tool_result": {"id": 1}}
    worker._executed_tool_trace = [
        {
            "action": AWMAction(kind="tool", name="lookup", arguments={}).to_dict(),
            "reward_type": "tool_call_ok",
            "observation_signature": replay_observation_signature(lookup_payload),
        }
    ]
    return worker


def test_error_signature_accepts_only_strong_infrastructure_evidence():
    server_error = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    assert deterministic_error_signature(server_error, phase="tool", tool_name="broken")
    route_collision = {
        "reward_type": "server_error",
        "error": ('Status code: 422. Response: {"detail":[{"loc":["path","item_id"],"input":"search-by-name"}]}'),
    }
    assert deterministic_error_signature(route_collision, phase="tool", tool_name="broken")
    ordinary_bad_argument = {
        "reward_type": "invalid_args",
        "error": "Input should be a valid integer",
    }
    assert (
        deterministic_error_signature(
            ordinary_bad_argument,
            phase="tool",
            tool_name="broken",
        )
        is None
    )
    assert not infrastructure_error(ordinary_bad_argument, phase="tool")
    ordinary_http_4xx = {
        "reward_type": "server_error",
        "error": "Error calling lookup. Status code: 404. Response: Not Found",
    }
    assert (
        deterministic_error_signature(
            ordinary_http_4xx,
            phase="tool",
            tool_name="lookup",
        )
        is None
    )
    assert not infrastructure_error(ordinary_http_4xx, phase="tool")


def test_exact_prefix_replay_confirms_same_environment_error():
    worker = _worker()
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    replay_env = _ReplayEnv(worker._tools, payload)

    async def new_env():
        return replay_env

    worker._new_env = new_env
    signature = deterministic_error_signature(payload, phase="tool", tool_name="broken")
    result = asyncio.run(
        worker._replay_runtime_failure(
            phase="tool",
            expected_signature=signature,
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert result["status"] == "confirmed"
    assert result["signature"] == signature
    assert replay_env.closed is True


def test_prefix_observation_drift_remains_pending():
    worker = _worker()
    payload = {
        "reward_type": "server_error",
        "error": "Error calling broken. Status code: 500. Response: Internal Server Error",
    }
    replay_env = _ReplayEnv(
        worker._tools,
        payload,
        prefix_payload={"reward_type": "tool_call_ok", "tool_result": {"id": 2}},
    )

    async def new_env():
        return replay_env

    worker._new_env = new_env
    result = asyncio.run(
        worker._replay_runtime_failure(
            phase="tool",
            expected_signature=deterministic_error_signature(
                payload,
                phase="tool",
                tool_name="broken",
            ),
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert result["status"] == "pending"
    assert result["error"] == "replay prefix observation changed"
    assert replay_env.closed is True


def test_runtime_failure_masks_only_current_group_and_does_not_advance():
    worker = _worker()
    action = AWMAction(kind="tool", name="broken", arguments={})
    teacher_samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    fingerprint = state_fingerprint(
        worker._scenario,
        worker._task_idx,
        worker._chat,
        worker._tools,
    )
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
            "runtime_failure_confirmed": True,
            "runtime_infrastructure_pending": False,
            "runtime_replay_status": "confirmed",
            "terminal_success": None,
            "terminal_reason": "runtime_confirmed",
        }
        return 0.0, True

    worker._execute = execute
    raw = '<tool_call>{"name":"broken","arguments":{}}</tool_call>'
    result = asyncio.run(worker.step_candidate_group([raw] * 4))
    candidate_results, selected_index, _, reward, done, selected_info = result

    assert selected_index in range(4)
    assert reward == 0.0
    assert done is True
    assert selected_info["runtime_failure_confirmed"] is True
    assert selected_info["state_group_advanced"] is False
    assert all(item[3]["semantic_train_mask"] is False for item in candidate_results)
    assert all(item[3]["runtime_train_mask"] is False for item in candidate_results)
    assert all(item[3]["state_group_advanced"] is False for item in candidate_results)


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
    assert info["terminal_success"] is None
    assert info["semantic_train_mask"] is False
    assert info["runtime_train_mask"] is False
    assert info["runtime_failure"] is False
    assert info["context_excess_tokens"] == 146


def test_recorder_repairs_torn_tail_and_never_deduplicates_tasks(tmp_path):
    recorder_class = AWMRuntimeFailureRecorder.__ray_metadata__.modified_class
    path = tmp_path / "runtime_failures.jsonl"
    first = {
        "protocol_version": 1,
        "task_id": "scenario:0",
        "replay_status": "confirmed",
    }
    path.write_text(json.dumps(first) + '\n{"task_id":')

    recorder = recorder_class(str(path))
    recorder.record(
        {
            "task_id": "scenario:0",
            "replay_status": "pending",
        }
    )

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [record["replay_status"] for record in records] == [
        "confirmed",
        "pending",
    ]
    assert recorder.stats() == {"runtime_failure_records": 2}
