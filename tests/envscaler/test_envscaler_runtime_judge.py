import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    canonical_action,
    normalize_tools,
    state_fingerprint,
)
from agent_system.environments.env_package.awm.runtime.oracle import (
    DeepSeekAWMOracleClient,
)
from agent_system.environments.env_package.envscaler.runtime import (
    EnvScalerWorker,
)
from agent_system.environments.env_package.envscaler.runtime_judge import (
    ENVSCALER_RUNTIME_JUDGE_INSTRUCTION,
    ENVSCALER_RUNTIME_JUDGE_SCOPE,
    build_envscaler_runtime_judge_evidence,
)


def _deepseek_response(verdict):
    return {
        "choices": [{"message": {"content": json.dumps(verdict)}}],
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp-envscaler-runtime",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
        },
    }


def _empty_awm_evidence(tmp_path):
    data_dir = tmp_path / "awm"
    data_dir.mkdir()
    (data_dir / "gen_envs.jsonl").write_text("", encoding="utf-8")
    (data_dir / "gen_db.jsonl").write_text("", encoding="utf-8")
    return data_dir


def _evidence():
    return build_envscaler_runtime_judge_evidence(
        source_identity={"commit": "pinned"},
        task={
            "env_id": "inventory",
            "task_id": "inventory:1",
            "task": "Update an existing item",
            "checklist_with_func": [],
        },
        environment={"env_class_code": ("class Inventory:\n    def update(self, item_id):\n        return self.items[item_id]\n")},
        visible_chat=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "update the item"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "update",
                    "parameters": {
                        "type": "object",
                        "properties": {"item_id": {"type": "integer"}},
                        "required": ["item_id"],
                    },
                },
            }
        ],
        failed_action={
            "kind": "tool",
            "name": "update",
            "arguments": {"item_id": 999},
        },
        exception_type="KeyError",
        exception_message="999",
        traceback_text="Traceback: KeyError: 999",
        state_before={"items": {1: "known"}, "tags": {"a", "b"}},
        state_at_exception={"items": {1: "known"}},
    )


@pytest.mark.parametrize("response_model", ["deepseek-v4-flash", "deepseek-flash"])
def test_envscaler_runtime_judge_is_cache_first_and_persisted(tmp_path, response_model):
    data_dir = _empty_awm_evidence(tmp_path)
    cache_path = tmp_path / "runtime_judge.jsonl"
    payloads = []
    verdict = {
        "error_class": "policy_execution_error",
        "classification_confidence": 95,
        "post_error_state": "unchanged",
        "rationale": "The requested item ID is absent from current state.",
    }

    def request(payload):
        payloads.append(payload)
        return {**_deepseek_response(verdict), "model": response_model}

    client = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_cache_path=str(cache_path),
        request_fn=request,
    )
    first = client.classify_envscaler_runtime_failure(evidence=_evidence())
    second = client.classify_envscaler_runtime_failure(evidence=_evidence())

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(payloads) == 1
    assert payloads[0]["messages"][0] == {
        "role": "system",
        "content": ENVSCALER_RUNTIME_JUDGE_INSTRUCTION,
    }
    assert payloads[0]["thinking"] == {"type": "enabled"}
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    assert record["judge_scope"] == ENVSCALER_RUNTIME_JUDGE_SCOPE
    assert record["verdict"] == verdict

    def fail_on_request(_payload):
        raise AssertionError("persisted EnvScaler verdict must be cache-first")

    reloaded = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_cache_path=str(cache_path),
        request_fn=fail_on_request,
    )
    cached = reloaded.classify_envscaler_runtime_failure(evidence=_evidence())
    assert cached["cache_hit"] is True
    assert reloaded.stats()["envscaler_runtime_judge_cache_records_loaded"] == 1


class _RemoteVerdict:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)

        async def result():
            return dict(self.verdict)

        return result()


class _BrokenRuntime:
    def __init__(self):
        self.value = 1

    def update(self, item_id):
        self.value = 99
        raise KeyError(item_id)


def _worker_with_verdict(verdict):
    remote = _RemoteVerdict(verdict)
    actor = SimpleNamespace(classify_envscaler_runtime_failure=remote)
    worker_class = EnvScalerWorker.__ray_metadata__.modified_class
    worker = worker_class(
        oracle_actor=actor,
        runtime_judge_enabled=True,
        runtime_judge_confidence_threshold=80,
    )
    worker._source = SimpleNamespace(identity={"commit": "pinned"})
    worker._task = {
        "task_id": "inventory:1",
        "env_id": "inventory",
        "task": "Update an existing item",
        "checklist_with_func": [],
    }
    worker._task_index = 0
    worker._environment = {"env_class_code": ("class Inventory:\n    def update(self, item_id):\n        self.value = 99\n        raise KeyError(item_id)\n")}
    worker._runtime = _BrokenRuntime()
    worker._initial_state = {"value": 1}
    worker._chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "update the item"},
    ]
    worker._tools = normalize_tools(
        [
            {
                "name": "update",
                "description": "Update one item.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}},
                    "required": ["item_id"],
                },
            }
        ]
    )
    worker._last_observation = "update the item"
    worker._last_info = {}
    return worker, remote


def test_policy_execution_error_restores_state_penalizes_and_continues():
    worker, remote = _worker_with_verdict(
        {
            "error_class": "policy_execution_error",
            "classification_confidence": 95,
            "post_error_state": "unchanged",
            "rationale": "The ID is absent from current state.",
        }
    )
    action = AWMAction(
        kind="tool",
        name="update",
        arguments={"item_id": 999},
    )

    done = asyncio.run(worker._execute("raw", action))

    assert done is False
    assert worker._runtime.value == 1
    assert worker._last_info["runtime_train_mask"] is True
    assert worker._last_info["runtime_policy_error"] is True
    assert worker._last_info["runtime_policy_continued"] is True
    assert len(remote.calls) == 1


def test_policy_execution_error_overrides_identical_candidate_rewards():
    worker, _ = _worker_with_verdict(
        {
            "error_class": "policy_execution_error",
            "classification_confidence": 95,
            "post_error_state": "unchanged",
            "rationale": "The ID is absent from current state.",
        }
    )
    failed_action = AWMAction(
        kind="tool",
        name="update",
        arguments={"item_id": 999},
    )
    samples = [{"sample_index": index, "action": failed_action.to_dict()} for index in range(3)]
    worker._prepared_teacher_supervision = {
        "state_fingerprint": state_fingerprint(
            "envscaler:inventory",
            worker._task_index,
            worker._chat,
            worker._tools,
        ),
        "teacher_samples": samples,
        "teacher_actions": [failed_action] * 3,
        "teacher_multiset": [failed_action.to_dict()] * 3,
        "teacher_invalid_sample_count": 0,
        "teacher_action_kind_disagreement": False,
    }
    failed = '<tool_call>{"name":"update","arguments":{"item_id":999}}</tool_call>'
    alternative = '<tool_call>{"name":"update","arguments":{"item_id":1}}</tool_call>'
    malformed = "<tool_call>{bad json}</tool_call>"

    results, selected_index, _, selected_reward, done, _ = asyncio.run(
        worker.step_candidate_group(
            [failed, failed, alternative, malformed],
        )
    )

    assert selected_index in {0, 1}
    assert [item[1] for item in results] == [-1.0, -1.0, 0.0, -1.0]
    assert [item[3]["runtime_policy_penalty"] for item in results] == [True, True, False, False]
    assert canonical_action(failed_action) == results[0][3]["parsed_action"]
    assert selected_reward == -1.0
    assert done is False


def test_infrastructure_or_uncertain_exception_masks_and_terminates():
    worker, _ = _worker_with_verdict(
        {
            "error_class": "infrastructure_error",
            "classification_confidence": 95,
            "post_error_state": "unchanged",
            "rationale": "Generated implementation references a missing field.",
        }
    )
    action = AWMAction(
        kind="tool",
        name="update",
        arguments={"item_id": 1},
    )

    done = asyncio.run(worker._execute("raw", action))

    assert done is True
    assert worker._runtime.value == 1
    assert worker._last_info["runtime_train_mask"] is False
    assert worker._last_info["runtime_failure"] is True
    assert worker._last_info["runtime_policy_error"] is False
    assert worker._last_info["terminal_reason"] == "runtime_masked"
