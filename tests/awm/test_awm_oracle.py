import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    canonical_action,
    validate_action,
)
from agent_system.environments.env_package.awm.runtime.oracle import (
    MATCHER_DECODING_CONFIG,
    MATCHER_PROMPT_HASH,
    ORACLE_PROTOCOL_VERSION,
    TEACHER_PROMPT_HASH,
    TEACHER_PROMPT_REVISION,
    TEACHER_SINGLE_ACTION_INSTRUCTION,
    DeepSeekAWMOracleClient,
    build_teacher_messages,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update",
            "description": "Update",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _response(content, *, model="deepseek-v4-flash", prompt_tokens=0, completion_tokens=0):
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "system_fingerprint": "fp-test",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_teacher_keeps_three_ordered_samples_and_duplicates(tmp_path):
    client = DeepSeekAWMOracleClient(
        cache_path=str(tmp_path / "teacher.jsonl"),
        request_fn=lambda payload: _response("unused"),
    )

    def fake_sample(messages, tools, sample_index, **kwargs):
        item_id = 1 if sample_index < 2 else 2
        action = AWMAction(kind="tool", name="lookup", arguments={"item_id": item_id})
        return {
            "sample_index": sample_index,
            "action": action.to_dict(),
            "raw_content": str(item_id),
            "reasoning_content": "",
        }

    client._sample_once = fake_sample
    samples = client.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
    )
    assert [sample["action"]["arguments"]["item_id"] for sample in samples] == [1, 1, 2]
    assert len(samples) == 3

    cached = client.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "changed"}],
        tools=TOOLS,
    )
    assert cached == samples
    assert client.stats()["teacher_cache_hits"] == 1


def test_teacher_singleflight_preserves_one_shared_multiset(tmp_path):
    client = DeepSeekAWMOracleClient(
        cache_path=str(tmp_path / "teacher.jsonl"),
        request_fn=lambda payload: _response("unused"),
    )
    calls = 0
    calls_lock = threading.Lock()
    callers = threading.Barrier(2)

    def fake_sample(messages, tools, sample_index, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        action = AWMAction(
            kind="tool",
            name="lookup",
            arguments={"item_id": sample_index},
        )
        return {
            "sample_index": sample_index,
            "action": action.to_dict(),
            "raw_content": str(sample_index),
            "reasoning_content": "",
        }

    def invoke():
        callers.wait()
        return client.sample_multiset(
            state_fingerprint="shared",
            messages=[{"role": "user", "content": "task"}],
            tools=TOOLS,
        )

    client._sample_once = fake_sample
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))

    assert results[0] == results[1]
    assert calls == 3
    stats = client.stats()
    assert stats["teacher_cache_lookups"] == 2
    assert stats["teacher_cache_misses"] == 1
    assert stats["teacher_cache_singleflight_waits"] == 1
    assert stats["teacher_cache_generated_sets"] == 1
    record = json.loads((tmp_path / "teacher.jsonl").read_text().strip())
    assert len(record["teacher_samples"]) == 3
    assert record["protocol_version"] == ORACLE_PROTOCOL_VERSION == 13
    assert record["teacher_prompt_revision"] == TEACHER_PROMPT_REVISION
    assert record["teacher_prompt_hash"] == TEACHER_PROMPT_HASH
    assert record["teacher_protocol_config"]["teacher_prompt_hash"] == TEACHER_PROMPT_HASH
    assert record["progress_context"] == {
        "multi_call_fallback_eligible": False,
        "previous_canonical_action": None,
    }
    assert record["teacher_cache_fingerprint"]


def test_cache_load_does_not_count_historical_api_usage(tmp_path):
    cache_path = tmp_path / "teacher.jsonl"
    client = DeepSeekAWMOracleClient(
        cache_path=str(cache_path),
        request_fn=lambda payload: _response("Done", prompt_tokens=11, completion_tokens=7),
    )
    client.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
    )
    legacy_record = json.loads(cache_path.read_text())
    assert legacy_record.pop("provider") == "deepseek"
    cache_path.write_text(json.dumps(legacy_record) + "\n")

    reloaded = DeepSeekAWMOracleClient(
        cache_path=str(cache_path),
        request_fn=lambda payload: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    reloaded.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
    )
    stats = reloaded.stats()
    assert stats["requests"] == 0
    assert stats["teacher_requests"] == 0
    assert stats["teacher_total_tokens"] == 0
    assert stats["teacher_cache_records_loaded"] == 1
    assert stats["teacher_cache_hits"] == 1


def test_teacher_cache_ignores_stale_prompt_protocol(tmp_path):
    cache_path = tmp_path / "teacher.jsonl"
    original = DeepSeekAWMOracleClient(
        cache_path=str(cache_path),
        request_fn=lambda payload: _response("Done"),
    )
    original.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
    )
    record = json.loads(cache_path.read_text())
    record["teacher_protocol_config"]["teacher_prompt_hash"] = "stale"
    cache_path.write_text(json.dumps(record) + "\n")

    calls = 0

    def request(payload):
        nonlocal calls
        calls += 1
        return _response("Done")

    reloaded = DeepSeekAWMOracleClient(
        cache_path=str(cache_path),
        request_fn=request,
    )
    reloaded.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
    )

    assert reloaded.stats()["teacher_cache_records_loaded"] == 0
    assert calls == 3


def test_teacher_cache_is_scoped_to_repeat_fallback_context(tmp_path):
    def request(payload):
        response = _response(None)
        response["choices"] = [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "lookup", "arguments": "{}"}},
                        {"function": {"name": "update", "arguments": "{}"}},
                    ],
                },
            }
        ]
        return response

    client = DeepSeekAWMOracleClient(
        cache_path=str(tmp_path / "teacher.jsonl"),
        request_fn=request,
    )
    previous = canonical_action(AWMAction(kind="tool", name="lookup", arguments={}))
    initial = client.sample_multiset(
        state_fingerprint="same-rendered-state",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
        previous_canonical_action=previous,
        no_progress_repeat_streak=1,
    )
    fallback = client.sample_multiset(
        state_fingerprint="same-rendered-state",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
        previous_canonical_action=previous,
        no_progress_repeat_streak=2,
    )
    cached_fallback = client.sample_multiset(
        state_fingerprint="same-rendered-state",
        messages=[{"role": "user", "content": "task"}],
        tools=TOOLS,
        previous_canonical_action=previous,
        no_progress_repeat_streak=3,
    )

    assert {item["action"]["name"] for item in initial} == {"lookup"}
    assert {item["action"]["name"] for item in fallback} == {"update"}
    assert cached_fallback == fallback
    stats = client.stats()
    assert stats["teacher_cache_generated_sets"] == 2
    assert stats["teacher_cache_hits"] == 1


def test_teacher_request_uses_only_supported_thinking_parameters():
    payloads = []

    def request(payload):
        payloads.append(payload)
        return _response("Done", prompt_tokens=11, completion_tokens=7)

    client = DeepSeekAWMOracleClient(request_fn=request)
    sample = client._sample_once([{"role": "user", "content": "task"}], TOOLS, 0)
    assert payloads[0]["thinking"] == {"type": "enabled"}
    assert payloads[0]["reasoning_effort"] == "max"
    assert "temperature" not in payloads[0]
    assert "top_p" not in payloads[0]
    assert sample["usage"]["total_tokens"] == 18
    assert payloads[0]["tools"] == TOOLS
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[0]["parallel_tool_calls"] is False
    assert client.stats()["teacher_total_tokens"] == 18


def test_dashscope_qwen36_teacher_uses_native_thinking_and_function_calling_parameters():
    payloads = []

    def request(payload):
        payloads.append(payload)
        response = _response(None, model="qwen3.6-flash")
        response["choices"] = [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "private reasoning",
                    "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}],
                },
            }
        ]
        return response

    client = DeepSeekAWMOracleClient(
        provider="dashscope",
        model="qwen3.6-flash",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        reasoning_effort=None,
        thinking_budget=4096,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        request_fn=request,
    )
    sample = client._sample_once([{"role": "user", "content": "task"}], TOOLS, 0)

    assert payloads[0]["enable_thinking"] is True
    assert payloads[0]["thinking_budget"] == 4096
    assert payloads[0]["temperature"] == 0.6
    assert payloads[0]["top_p"] == 0.95
    assert "thinking" not in payloads[0]
    assert "reasoning_effort" not in payloads[0]
    assert payloads[0]["parallel_tool_calls"] is False
    assert sample["action"] == {"kind": "tool", "name": "lookup", "arguments": {}, "content": None, "error": None}


def test_teacher_executes_first_native_call_and_records_truncation():
    def request(payload):
        response = _response(None)
        response["choices"] = [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "reason",
                    "tool_calls": [
                        {"id": "first", "function": {"name": "lookup", "arguments": "{}"}},
                        {"id": "second", "function": {"name": "lookup", "arguments": "{}"}},
                    ],
                },
            }
        ]
        return response

    client = DeepSeekAWMOracleClient(request_fn=request)
    sample = client._sample_once([{"role": "user", "content": "task"}], TOOLS, 0)
    assert sample["action"]["name"] == "lookup"
    assert sample["skipped_tool_calls"] == 1
    assert sample["reasoning_content"] == "reason"
    assert client.stats()["teacher_parallel_calls_truncated"] == 1


def test_teacher_multi_call_fallback_uses_one_nonrepeat_schema_valid_vote():
    def request(payload):
        response = _response(None)
        response["choices"] = [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "repeat", "function": {"name": "lookup", "arguments": "{}"}},
                        {"id": "advance", "function": {"name": "update", "arguments": "{}"}},
                    ],
                },
            }
        ]
        return response

    client = DeepSeekAWMOracleClient(request_fn=request)
    repeated = canonical_action(AWMAction(kind="tool", name="lookup", arguments={}))
    sample = client._sample_once(
        [{"role": "user", "content": "task"}],
        TOOLS,
        0,
        previous_canonical_action=repeated,
        no_progress_repeat_streak=2,
    )

    assert sample["action"]["name"] == "update"
    assert sample["raw_tool_call_count"] == 2
    stats = client.stats()
    assert stats["teacher_parallel_responses"] == 1
    assert stats["teacher_parallel_alternative_available"] == 1
    assert stats["teacher_parallel_fallback_applied"] == 1
    assert stats["teacher_parallel_fallback_rate"] == 1.0


def test_teacher_does_not_rescue_invalid_first_call_with_later_valid_call():
    def request(payload):
        response = _response(None)
        response["choices"] = [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "invalid-primary",
                            "function": {
                                "name": "missing_tool",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "valid-later",
                            "function": {
                                "name": "update",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
            }
        ]
        return response

    client = DeepSeekAWMOracleClient(request_fn=request)
    repeated = canonical_action(AWMAction(kind="tool", name="lookup", arguments={}))
    sample = client._sample_once(
        [{"role": "user", "content": "task"}],
        TOOLS,
        0,
        previous_canonical_action=repeated,
        no_progress_repeat_streak=2,
    )

    assert sample["action"]["name"] == "missing_tool"
    assert validate_action(AWMAction(**sample["action"]), TOOLS).kind == "invalid"
    assert client.stats()["teacher_parallel_fallback_applied"] == 0


def test_provider_fingerprint_drift_is_recorded_without_rejecting_response():
    fingerprint = "fp-a"

    def request(payload):
        response = _response("Done")
        response["system_fingerprint"] = fingerprint
        return response

    client = DeepSeekAWMOracleClient(request_fn=request)
    first = client._sample_once([{"role": "user", "content": "task"}], TOOLS, 0)
    fingerprint = "fp-b"
    second = client._sample_once([{"role": "user", "content": "task"}], TOOLS, 1)

    assert first["provider_identity"]["system_fingerprint"] == "fp-a"
    assert second["provider_identity"]["system_fingerprint"] == "fp-b"
    stats = client.stats()
    assert stats["teacher_provider_fingerprint_count"] == 2
    assert stats["teacher_provider_fingerprint_changes"] == 1


def test_provider_model_drift_still_fails_loudly():
    def request(payload):
        response = _response("Done")
        response["model"] = "unexpected-model"
        return response

    client = DeepSeekAWMOracleClient(request_fn=request)
    with pytest.raises(RuntimeError, match="returned model 'unexpected-model'"):
        client._sample_once([{"role": "user", "content": "task"}], TOOLS, 1)


def test_matcher_judges_every_candidate_teacher_pair_and_sums_booleans(tmp_path):
    def request(payload):
        body = json.loads(payload["messages"][0]["content"].split("\n", 1)[1])
        equivalent = body["candidate_message"].startswith(body["teacher_message"])
        return _response(json.dumps({"equivalent": equivalent}))

    client = DeepSeekAWMOracleClient(
        matcher_cache_path=str(tmp_path / "matcher.jsonl"),
        request_fn=request,
    )
    result = client.match_message_pairs(
        ["A", "A extra", "B"],
        ["A", "A extra detail"],
    )
    assert result["matrix"] == [[True, False, False], [True, True, False]]
    assert result["counts"] == [1, 2]
    records = [json.loads(line) for line in (tmp_path / "matcher.jsonl").read_text().splitlines()]
    assert all(record["prompt_hash"] == MATCHER_PROMPT_HASH for record in records)
    assert all(record["decoding_config"] == MATCHER_DECODING_CONFIG for record in records)


def test_matcher_reuses_one_frozen_decision_for_duplicate_pairs():
    calls = 0

    def request(payload):
        nonlocal calls
        calls += 1
        return _response('{"equivalent":true}')

    client = DeepSeekAWMOracleClient(request_fn=request)
    result = client.match_message_pairs(["same", "same", "same"], ["different"])

    assert result == {"counts": [3], "matrix": [[True, True, True]]}
    assert calls == 1
    assert client.stats()["matcher_pair_evaluations"] == 3
    assert client.stats()["matcher_unique_pairs"] == 1


def test_matcher_failure_raises_instead_of_becoming_false_label():
    client = DeepSeekAWMOracleClient(request_fn=lambda payload: _response('{"wrong":true}'))
    with pytest.raises(ValueError, match="equivalent"):
        client.match_message_pairs(["teacher"], ["candidate"])


def test_expert_sees_exact_student_visible_state_without_candidates():
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    messages = build_teacher_messages(chat)
    assert messages[0]["content"].startswith("policy")
    assert TEACHER_SINGLE_ACTION_INSTRUCTION in messages[0]["content"]
    assert messages[1] == chat[1]
    assert messages[2]["reasoning_content"] == ""
    assert "reasoning_content" not in chat[2]
    assert TEACHER_SINGLE_ACTION_INSTRUCTION not in chat[0]["content"]
    assert messages is not chat
