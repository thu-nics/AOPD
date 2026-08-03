import asyncio
import json

from agent_system.environments.env_package.awm.actions import AWMAction, normalize_tools
from agent_system.environments.env_package.awm.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime_quarantine import (
    AWMRuntimeQuarantineRegistry,
    deterministic_error_signature,
    infrastructure_error,
    replay_observation_signature,
)
from agent_system.multi_turn_rollout.rollout_loop import (
    _mask_awm_runtime_trajectories,
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
    lookup_payload = {"reward_type": "tool_call_ok", "tool_result": {"id": 1}}
    worker._executed_tool_trace = [
        {
            "action": AWMAction(kind="tool", name="lookup", arguments={}).to_dict(),
            "reward_type": "tool_call_ok",
            "observation_signature": replay_observation_signature(lookup_payload),
        }
    ]
    return worker


def test_error_signature_accepts_only_strong_deterministic_evidence():
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
    assert deterministic_error_signature(ordinary_bad_argument, phase="tool", tool_name="broken") is None
    assert not infrastructure_error(ordinary_bad_argument, phase="tool")
    ordinary_http_4xx = {
        "reward_type": "server_error",
        "error": "Error calling lookup. Status code: 404. Response: Not Found",
    }
    assert deterministic_error_signature(ordinary_http_4xx, phase="tool", tool_name="lookup") is None
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

    assert result["status"] == "quarantine"
    assert result["signature"] == signature
    assert replay_env.closed is True


def test_prefix_observation_drift_cannot_quarantine_task():
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
            expected_signature=deterministic_error_signature(payload, phase="tool", tool_name="broken"),
            action=AWMAction(kind="tool", name="broken", arguments={}),
        )
    )

    assert result["status"] == "pending"
    assert result["error"] == "replay prefix observation changed"
    assert replay_env.closed is True


def test_runtime_failure_masks_all_prior_rows_from_same_reset():
    trajectories = [
        [
            {"semantic_train_mask": True, "runtime_train_mask": True},
            {"semantic_train_mask": True, "runtime_train_mask": True},
        ],
        [{"semantic_train_mask": True, "runtime_train_mask": True}],
    ]
    infos = [
        [{}, {"runtime_quarantine": True}],
        [{"runtime_infrastructure_pending": False}],
    ]

    assert _mask_awm_runtime_trajectories(trajectories, infos) == 1
    assert all(not row["semantic_train_mask"] for row in trajectories[0])
    assert all(not row["runtime_train_mask"] for row in trajectories[0])
    assert all(row["runtime_trajectory_masked"] for row in trajectories[0])
    assert trajectories[1][0]["semantic_train_mask"] is True
    assert trajectories[1][0]["runtime_trajectory_masked"] is False


def test_registry_repairs_torn_tail_before_appending(tmp_path):
    registry_class = AWMRuntimeQuarantineRegistry.__ray_metadata__.modified_class
    path = tmp_path / "runtime_quarantine.jsonl"
    first = {
        "protocol_version": 1,
        "task_id": "scenario:0",
        "signature": "signature-0",
    }
    path.write_text(json.dumps(first) + '\n{"task_id":')

    registry = registry_class(str(path))
    assert registry.is_quarantined("scenario:0")
    assert registry.record(
        {
            "task_id": "scenario:1",
            "signature": "signature-1",
        }
    )

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [record["task_id"] for record in records] == [
        "scenario:0",
        "scenario:1",
    ]
