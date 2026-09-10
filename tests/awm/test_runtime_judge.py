import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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


@pytest.mark.parametrize("response_model", ["deepseek-v4-flash", "deepseek-flash"])
def test_runtime_judge_uses_thinking_low_8k_exact_schema_and_cache(tmp_path, response_model):
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
        return {**_deepseek_response(verdict), "model": response_model}

    cache_path = tmp_path / "runtime_judge.jsonl"
    client = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_reference_trials_path=str(trials_path),
        runtime_judge_cache_path=str(cache_path),
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
    assert request_payload["reasoning_effort"] == "low"
    assert client.reasoning_effort == "max"  # Teacher decoding is independent.
    assert "temperature" not in request_payload
    assert "top_p" not in request_payload
    assert request_payload["max_tokens"] == 8192
    assert request_payload["response_format"] == {"type": "json_object"}
    assert request_payload["messages"][0] == {
        "role": "system",
        "content": RUNTIME_JUDGE_INSTRUCTION,
    }
    assert "An irrelevant or unnecessary but schema-valid tool call" in request_payload["messages"][0]["content"]
    assert "NO REFERENCE WAS PROVIDED" in request_payload["messages"][0]["content"]
    assert "unchanged is not established" in request_payload["messages"][0]["content"]
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


def test_runtime_judge_uses_dashscope_native_decoding():
    assert runtime_judge_decoding_config(provider="dashscope") == {
        "enable_thinking": True,
        "thinking_budget": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def test_runtime_judge_uses_zai_native_decoding():
    assert runtime_judge_decoding_config(provider="zai") == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "max",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_runtime_judge_effort_is_configurable(effort):
    assert runtime_judge_decoding_config(reasoning_effort=effort)["reasoning_effort"] == effort
    with pytest.raises(ValueError, match="reasoning_effort"):
        runtime_judge_decoding_config(reasoning_effort="invalid")


def _judge_client(tmp_path, request, **kwargs):
    data_dir, _ = _evidence_files(tmp_path)
    client = DeepSeekAWMOracleClient(
        runtime_judge_enabled=True,
        runtime_judge_data_dir=str(data_dir),
        runtime_judge_cache_path=str(tmp_path / "runtime_judge.jsonl"),
        request_fn=request,
        **kwargs,
    )
    client.runtime_judge_evidence = SimpleNamespace(build=lambda **_: {"task_id": "widgets:3"})
    return client


def _classify(client, envscaler):
    if envscaler:
        return client.classify_envscaler_runtime_failure(evidence={"task_id": "widgets:3"})
    return client.classify_runtime_failure(scenario="widgets", task_idx=3, task="create", failed_action={}, payload={})


def _uncertain_verdict():
    return {
        "error_class": "uncertain",
        "classification_confidence": 60,
        "post_error_state": "unchanged",
        "rationale": "A read-only lookup failed; the supplied evidence cannot establish the cause.",
    }


@pytest.mark.parametrize("envscaler", [False, True])
@pytest.mark.parametrize("bad_kind", ["empty", "json", "fields", "truncated", "no_choices"])
def test_runtime_judge_retries_format_only_and_accounts_all_usage(tmp_path, envscaler, bad_kind):
    calls = []

    def request(payload):
        calls.append(payload)
        result = _deepseek_response(_uncertain_verdict())
        result["usage"]["prompt_cache_hit_tokens"] = 80
        result["usage"]["completion_tokens_details"] = {"reasoning_tokens": 35}
        if len(calls) == 1:
            if bad_kind == "empty":
                result["choices"][0]["message"]["content"] = None
            elif bad_kind == "json":
                result["choices"][0]["message"]["content"] = "not JSON"
            elif bad_kind == "fields":
                result["choices"][0]["message"]["content"] = json.dumps({"error_class": "uncertain"})
            elif bad_kind == "truncated":
                result["choices"][0]["finish_reason"] = "length"
            else:
                result["choices"] = []
        return result

    client = _judge_client(tmp_path, request)
    assert _classify(client, envscaler)["error_class"] == "uncertain"
    assert _classify(client, envscaler)["cache_hit"] is True
    assert len(calls) == 2  # Valid uncertain is accepted, not retried toward policy.
    assert calls[0] == calls[1]  # Failed reasoning/content never enters a new prompt.
    prefix = "envscaler_runtime_judge" if envscaler else "runtime_judge"
    stats = client.stats()
    assert stats[f"{prefix}_requests"] == 2
    assert stats[f"{prefix}_total_tokens"] == 300
    assert stats[f"{prefix}_failures"] == 0
    record = json.loads((tmp_path / "runtime_judge.jsonl").read_text())
    assert record["usage"]["total_tokens"] == 300
    assert len(record["usage_attempts"]) == 2
    assert all(attempt["prompt_cache_hit_tokens"] == 80 for attempt in record["usage_attempts"])
    assert all(attempt["completion_tokens_details"]["reasoning_tokens"] == 35 for attempt in record["usage_attempts"])
    assert record["protocol_version"] == 2


@pytest.mark.parametrize("envscaler", [False, True])
@pytest.mark.parametrize("format_retries", [0, 1])
def test_runtime_judge_exhaustion_never_caches_invalid_verdict(tmp_path, envscaler, format_retries):
    def request(_):
        return _deepseek_response({"unexpected": "invalid"})

    client = _judge_client(tmp_path, request, runtime_judge_max_format_retries=format_retries)
    for _ in range(2):
        with pytest.raises(ValueError, match="unexpected fields"):
            _classify(client, envscaler)
    prefix = "envscaler_runtime_judge" if envscaler else "runtime_judge"
    assert client.stats()[f"{prefix}_requests"] == 2 * (format_retries + 1)
    assert client.stats()[f"{prefix}_failures"] == 2
    assert not client._runtime_judge_flights
    assert not client._envscaler_runtime_judge_flights
    assert not (tmp_path / "runtime_judge.jsonl").exists()


@pytest.mark.parametrize("envscaler", [False, True])
def test_runtime_judge_identity_error_does_not_format_retry(tmp_path, envscaler):
    calls = []

    def request(payload):
        calls.append(payload)
        return {**_deepseek_response(_uncertain_verdict()), "model": "unapproved-model"}

    client = _judge_client(tmp_path, request)
    with pytest.raises(RuntimeError, match="model"):
        _classify(client, envscaler)
    assert len(calls) == 1
    assert not (tmp_path / "runtime_judge.jsonl").exists()


@pytest.mark.parametrize("envscaler", [False, True])
@pytest.mark.parametrize("obsolete_field", ["protocol_version", "prompt_hash", "decoding_config"])
def test_runtime_judge_old_protocol_prompt_and_decoding_are_not_reused(tmp_path, envscaler, obsolete_field):
    client = _judge_client(tmp_path, lambda _: _deepseek_response(_uncertain_verdict()))
    _classify(client, envscaler)
    path = tmp_path / "runtime_judge.jsonl"
    record = json.loads(path.read_text())
    if obsolete_field == "protocol_version":
        record[obsolete_field] = 1
    elif obsolete_field == "prompt_hash":
        record[obsolete_field] = "old-uncalibrated-prompt"
    else:
        record[obsolete_field]["reasoning_effort"] = "max"
    path.write_text(json.dumps(record) + "\n")
    reloaded = DeepSeekAWMOracleClient(runtime_judge_cache_path=str(path), request_fn=lambda _: None)
    assert reloaded.stats()["runtime_judge_cache_records_loaded"] == 0
    assert reloaded.stats()["envscaler_runtime_judge_cache_records_loaded"] == 0


@pytest.mark.parametrize("envscaler", [False, True])
def test_runtime_judge_singleflight_shares_format_retry(tmp_path, envscaler):
    entered_request = threading.Event()
    entered_waiter = threading.Event()
    release_request = threading.Event()
    calls = []

    def request(payload):
        calls.append(payload)
        if len(calls) == 1:
            entered_request.set()
            assert release_request.wait(timeout=5)
            return _deepseek_response({"malformed": True})
        return _deepseek_response(_uncertain_verdict())

    client = _judge_client(tmp_path, request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(_classify, client, envscaler)
        try:
            assert entered_request.wait(timeout=5)
            flights = client._envscaler_runtime_judge_flights if envscaler else client._runtime_judge_flights
            with client._lock:
                flight = next(iter(flights.values()))
                original_result = flight.result

                def waiting_result(*args, **kwargs):
                    entered_waiter.set()
                    return original_result(*args, **kwargs)

                flight.result = waiting_result
            follower = pool.submit(_classify, client, envscaler)
            assert entered_waiter.wait(timeout=5)
        finally:
            release_request.set()
        assert leader.result(timeout=5)["cache_hit"] is False
        assert follower.result(timeout=5)["cache_hit"] is True
    assert len(calls) == 2
    prefix = "envscaler_runtime_judge" if envscaler else "runtime_judge"
    assert client.stats()[f"{prefix}_cache_singleflight_waits"] == 1
    assert client.stats()[f"{prefix}_requests"] == 2
    assert len((tmp_path / "runtime_judge.jsonl").read_text().splitlines()) == 1
