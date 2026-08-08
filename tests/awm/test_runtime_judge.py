import json

import pytest

from agent_system.environments.env_package.awm.runtime.judge import (
    RUNTIME_JUDGE_INSTRUCTION,
    RUNTIME_JUDGE_PROMPT_HASH,
    RuntimeJudgeEvidenceStore,
    runtime_judge_decoding_config,
    validate_runtime_judge_verdict,
)
from agent_system.environments.env_package.awm.runtime.oracle import (
    DeepSeekAWMOracleClient,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _evidence_files(tmp_path):
    data_dir = tmp_path / "awm"
    data_dir.mkdir()
    source = """
class Widget(Base):
    __tablename__ = "widgets"

@app.post("/api/widgets", operation_id="create_widget")
async def create_widget(body):
    widget = Widget(name=body.name)
    session.add(widget)
    session.commit()
    return widget

@app.get("/api/widgets/{widget_id}", operation_id="get_widget")
async def get_widget(widget_id):
    return session.query(Widget).get(widget_id)
"""
    _write_jsonl(
        data_dir / "gen_envs.jsonl",
        [{"scenario": "widgets", "full_code": source}],
    )
    _write_jsonl(
        data_dir / "gen_db.jsonl",
        [
            {
                "scenario": "widgets",
                "db_schema": {
                    "tables": [
                        {
                            "name": "widgets",
                            "ddl": "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);",
                        }
                    ]
                },
            }
        ],
    )
    trials_path = tmp_path / "trials.jsonl"
    _write_jsonl(
        trials_path,
        [
            {
                "task_id": "widgets:3",
                "result": {
                    "trajectory": [
                        {
                            "action_kind": "tool",
                            "parsed_action": json.dumps(
                                {
                                    "kind": "tool",
                                    "name": "create_widget",
                                    "arguments": {"name": "working"},
                                }
                            ),
                        }
                    ]
                },
            }
        ],
    )
    return data_dir, trials_path


def _deepseek_response(content):
    return {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp-runtime-judge",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def test_evidence_store_extracts_endpoint_routes_ddl_and_success_reference(tmp_path):
    data_dir, trials_path = _evidence_files(tmp_path)
    store = RuntimeJudgeEvidenceStore(
        data_dir=data_dir,
        reference_trials_path=trials_path,
    )

    evidence = store.build(
        scenario="widgets",
        task_idx=3,
        task="Create a widget",
        failed_action={
            "kind": "tool",
            "name": "create_widget",
            "arguments": {"name": "duplicate"},
        },
        payload={"error": "Status code: 500"},
    )

    assert "async def create_widget" in evidence["failed_endpoint_source"]
    assert evidence["referenced_database_ddl"] == ["CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);"]
    assert evidence["successful_fresh_reset_reference_actions"] == [{"name": "create_widget", "arguments": {"name": "working"}}]
    assert {route["operation_id"] for route in evidence["related_route_registry"]} == {
        "create_widget",
        "get_widget",
    }


def test_trial_index_uses_only_top_level_task_id(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    _write_jsonl(
        trials_path,
        [
            {
                "result": {"task_id": "wrong:999", "trajectory": []},
                "task_id": "widgets:3",
            }
        ],
    )
    assert RuntimeJudgeEvidenceStore._trial_task_offsets(trials_path) == {"widgets:3": 0}


def test_runtime_judge_uses_thinking_max_8k_exact_schema_and_cache(tmp_path):
    data_dir, trials_path = _evidence_files(tmp_path)
    payloads = []
    verdict = {
        "error_class": "policy_execution_error",
        "classification_confidence": 95,
        "post_error_state": "unchanged",
        "rationale": "UNIQUE constraint failed before commit",
    }

    def request(payload):
        payloads.append(payload)
        return _deepseek_response(verdict)

    cache_path = tmp_path / "runtime_judge.jsonl"
    client = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_reference_trials_path=str(trials_path),
        runtime_judge_cache_path=str(cache_path),
        runtime_judge_reasoning_effort="max",
        runtime_judge_max_tokens=8192,
        request_fn=request,
    )
    kwargs = {
        "scenario": "widgets",
        "task_idx": 3,
        "task": "Create a widget",
        "failed_action": {
            "kind": "tool",
            "name": "create_widget",
            "arguments": {"name": "duplicate"},
        },
        "payload": {"error": "Status code: 500"},
    }

    first = client.classify_runtime_failure(**kwargs)
    second = client.classify_runtime_failure(**kwargs)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["error_class"] == "policy_execution_error"
    assert len(payloads) == 1
    request_payload = payloads[0]
    assert request_payload["thinking"] == {"type": "enabled"}
    assert request_payload["reasoning_effort"] == "max"
    assert request_payload["max_tokens"] == 8192
    assert request_payload["response_format"] == {"type": "json_object"}
    assert request_payload["messages"][0] == {
        "role": "system",
        "content": RUNTIME_JUDGE_INSTRUCTION,
    }
    assert "An irrelevant or unnecessary but schema-valid tool call" in request_payload["messages"][0]["content"]
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    assert record["prompt_hash"] == RUNTIME_JUDGE_PROMPT_HASH
    assert record["verdict"] == verdict
    stats = client.stats()
    assert stats["runtime_judge_requests"] == 1
    assert stats["runtime_judge_cache_hits"] == 1
    assert stats["runtime_judge_policy_execution_errors"] == 2

    def fail_on_request(_payload):
        raise AssertionError("a valid persisted verdict must be served cache-first")

    reloaded = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_reference_trials_path=str(trials_path),
        runtime_judge_cache_path=str(cache_path),
        runtime_judge_reasoning_effort="max",
        runtime_judge_max_tokens=8192,
        request_fn=fail_on_request,
    )
    reloaded_verdict = reloaded.classify_runtime_failure(**kwargs)
    assert reloaded_verdict["cache_hit"] is True
    assert reloaded_verdict["error_class"] == "policy_execution_error"
    assert reloaded.stats()["runtime_judge_cache_records_loaded"] == 1
    assert stats["runtime_judge_total_tokens"] == 150


def test_runtime_judge_rejects_extra_fields_and_small_response_budget():
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_runtime_judge_verdict(
            {
                "error_class": "policy_execution_error",
                "classification_confidence": 95,
                "post_error_state": "unchanged",
                "rationale": "constraint failed",
                "safe_to_continue": True,
            }
        )
    with pytest.raises(ValueError, match="max_tokens >= 8192"):
        runtime_judge_decoding_config(max_tokens=4096)
