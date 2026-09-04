import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_system.environments.env_package.tau_bench.actions import ParsedAction
from agent_system.environments.env_package.tau_bench.oracle import (
    ORACLE_PROTOCOL_VERSION,
    TauTeacherClient,
    build_teacher_messages,
)


def _lookup_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
            },
        }
    ]


def test_oracle_uses_three_independent_seeded_requests_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3, cache_path=str(tmp_path / "cache.jsonl"))
    seeds = []
    lock = threading.Lock()

    def fake_sample_once(*, messages, tools, seed):
        with lock:
            sample_index = len(seeds)
            seeds.append(seed)
        if sample_index < 2:
            return ParsedAction(kind="tool", name="lookup", arguments={"id": 1})
        return ParsedAction(kind="message", content="I can help with that.")

    monkeypatch.setattr(client, "_sample_once", fake_sample_once)
    actions = client.sample_multiset(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "hello"}],
        tools=_lookup_tools(),
    )
    assert len(seeds) == 3
    assert len(set(seeds)) == 3
    assert len(actions) == 3
    assert actions[0] == actions[1]

    cached = client.sample_multiset(
        state_fingerprint="state-a",
        messages=[
            {
                "role": "user",
                "content": "changed but fingerprint controls cache",
            }
        ],
        tools=_lookup_tools(),
    )
    assert cached == actions
    assert len(seeds) == 3
    assert client.stats()["cache_hits"] == 1
    record = json.loads((tmp_path / "cache.jsonl").read_text().strip())
    assert record["protocol_version"] == ORACLE_PROTOCOL_VERSION == 8
    assert len(record["teacher_samples"]) == 3
    assert [sample["sample_index"] for sample in record["teacher_samples"]] == [0, 1, 2]
    assert record["valid_samples"] == 3
    assert "oracle_actions" not in record


def test_oracle_v8_ignores_v7_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    cache_path = tmp_path / "legacy.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "protocol_version": 7,
                "state_fingerprint": "legacy-state",
                "model": "qwen3-32b",
                "api_base": "http://127.0.0.1:8000/v1",
                "samples": 3,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "enable_thinking": True,
                "max_tokens": 8192,
                "teacher_samples": [{"kind": "message", "content": "legacy"}] * 3,
            }
        )
        + "\n"
    )
    client = TauTeacherClient(samples=3, cache_path=str(cache_path))
    monkeypatch.setattr(
        client,
        "_sample_once",
        lambda **_: ParsedAction(kind="message", content="fresh"),
    )

    actions = client.sample_multiset(
        state_fingerprint="legacy-state",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert [action["content"] for action in actions] == ["fresh"] * 3
    assert client.stats()["cache_records_loaded"] == 0


def test_oracle_singleflight_generates_one_set_for_concurrent_state(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3, cache_path=str(tmp_path / "cache.jsonl"))
    calls = 0
    calls_lock = threading.Lock()
    callers = threading.Barrier(2)

    def fake_sample_once(*, messages, tools, seed):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return ParsedAction(kind="tool", name="lookup", arguments={"id": seed})

    def invoke():
        callers.wait()
        return client.sample_multiset(
            state_fingerprint="shared",
            messages=[{"role": "user", "content": "hello"}],
            tools=_lookup_tools(),
        )

    monkeypatch.setattr(client, "_sample_once", fake_sample_once)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))

    assert results[0] == results[1]
    assert calls == 3
    stats = client.stats()
    assert stats["cache_lookups"] == 2
    assert stats["cache_misses"] == 1
    assert stats["cache_singleflight_waits"] == 1
    assert stats["cache_generated_sets"] == 1


def test_oracle_retries_only_an_invalid_vote(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(
        samples=3,
        cache_path=str(tmp_path / "cache.jsonl"),
        teacher_validity_max_retries=2,
    )
    bad_seed = client._seed("retry-state", 1, 0)
    seeds = []

    def fake_sample_once(*, messages, tools, seed):
        seeds.append(seed)
        if seed == bad_seed:
            return ParsedAction(kind="tool", name="lookup", arguments={"unexpected": 1})
        return ParsedAction(kind="message", content=f"vote-{seed}")

    monkeypatch.setattr(client, "_sample_once", fake_sample_once)
    actions = client.sample_multiset(
        state_fingerprint="retry-state",
        messages=[{"role": "user", "content": "hello"}],
        tools=_lookup_tools(),
    )

    assert len(actions) == 3
    assert len(seeds) == 4
    assert client.stats()["teacher_validity_retries"] == 1
    assert client.stats()["teacher_validity_retry_recovered"] == 1
    assert client.stats()["teacher_validity_retry_exhausted"] == 0


def test_oracle_persists_and_refills_partial_valid_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    cache_path = tmp_path / "cache.jsonl"
    client = TauTeacherClient(
        samples=3,
        cache_path=str(cache_path),
        teacher_validity_max_retries=2,
    )
    failing_seeds = {client._seed("partial-state", 2, retry) for retry in range(3)}
    fail_vote = True
    seeds = []

    def fake_sample_once(*, messages, tools, seed):
        seeds.append(seed)
        if fail_vote and seed in failing_seeds:
            return ParsedAction(kind="invalid", error="still malformed")
        return ParsedAction(kind="message", content=f"vote-{seed}")

    monkeypatch.setattr(client, "_sample_once", fake_sample_once)
    first = client.sample_multiset(
        state_fingerprint="partial-state",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )
    assert len(first) == 2
    assert len(seeds) == 5

    fail_vote = False
    reloaded = TauTeacherClient(
        samples=3,
        cache_path=str(cache_path),
        teacher_validity_max_retries=2,
    )
    monkeypatch.setattr(reloaded, "_sample_once", fake_sample_once)
    second = reloaded.sample_multiset(
        state_fingerprint="partial-state",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert len(second) == 3
    assert len(seeds) == 6
    records = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
    assert [record["valid_samples"] for record in records] == [2, 3]
    assert [sample["sample_index"] for sample in records[-1]["teacher_samples"]] == [0, 1, 2]
    stats = reloaded.stats()
    assert stats["cache_records_loaded"] == 1
    assert stats["cache_partial_hits"] == 1
    assert stats["cache_refill_attempts"] == 1
    assert stats["cache_refill_votes"] == 1


def test_oracle_transport_failure_drops_only_its_vote(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(
        samples=3,
        cache_path=str(tmp_path / "cache.jsonl"),
    )
    error_seed = client._seed("transport-state", 1, 0)

    def fake_sample_once(*, messages, tools, seed):
        if seed == error_seed:
            raise RuntimeError("endpoint unavailable")
        return ParsedAction(kind="message", content=f"vote-{seed}")

    monkeypatch.setattr(client, "_sample_once", fake_sample_once)
    actions = client.sample_multiset(
        state_fingerprint="transport-state",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert len(actions) == 2
    assert client.stats()["teacher_vote_request_failures"] == 1


def test_teacher_context_is_student_visible_by_default_and_privileged_on_opt_in():
    visible_chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]
    privileged_context = {
        "user_scenario": {"instructions": "change flight"},
        "reference_resolution_actions": [{"name": "lookup", "arguments": {"id": "x"}}],
    }

    default_messages = build_teacher_messages(visible_chat)
    assert default_messages == visible_chat
    assert "reference_resolution_actions" not in default_messages[0]["content"]

    privileged_messages = build_teacher_messages(
        visible_chat,
        privileged_context=privileged_context,
        use_privileged_context=True,
    )
    assert "PRIVILEGED TEACHER CONTEXT" in privileged_messages[0]["content"]
    assert "reference_resolution_actions" in privileged_messages[0]["content"]
    assert privileged_messages[-1] == {"role": "user", "content": "hello"}


def test_semantic_matcher_counts_every_duplicate_teacher_sample(monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": '{"matches":[false,true,false]}'}}]}

    monkeypatch.setattr(client, "_post", fake_post)
    matched = client.match_message_pairs(
        ["I can help.", "I can help.", "other"],
        [" i can   HELP. ", "candidate"],
    )

    assert matched == {
        "counts": [2, 2],
        "matrix": [[True, True, False], [True, True, False]],
    }
    assert len(calls) == 1
    assert calls[0]["messages"][0]["content"].count('"candidate"') >= 3
    assert client.stats()["semantic_exact_matches"] == 2
    assert client.stats()["semantic_batch_requests"] == 1
    assert client.stats()["semantic_batch_failures"] == 0


def test_semantic_matcher_cache_persists_across_clients(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    cache_path = tmp_path / "matcher.jsonl"
    first = TauTeacherClient(samples=3, matcher_cache_path=str(cache_path))
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": '{"matches":[true]}'}}]}

    monkeypatch.setattr(first, "_post", fake_post)
    assert first.match_message_pairs(["teacher"], ["candidate"]) == {
        "counts": [1],
        "matrix": [[True]],
    }
    assert len(calls) == 1

    second = TauTeacherClient(samples=3, matcher_cache_path=str(cache_path))
    monkeypatch.setattr(
        second,
        "_post",
        lambda payload: (_ for _ in ()).throw(AssertionError("persistent matcher cache should avoid an API call")),
    )
    assert second.match_message_pairs(["teacher"], ["candidate"]) == {
        "counts": [1],
        "matrix": [[True]],
    }
    stats = second.stats()
    assert stats["matcher_cache_records_loaded"] == 1
    assert stats["matcher_cache_hits"] == 1
    assert stats["matcher_cache_hit_rate"] == 1.0


def test_semantic_matcher_returns_one_empty_row_per_candidate_without_teacher_messages(
    monkeypatch,
):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)

    assert client.match_message_pairs([], ["first", "second"]) == {
        "counts": [0, 0],
        "matrix": [[], []],
    }


def test_semantic_pair_matcher_falls_back_to_unique_pairs(monkeypatch, caplog):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        prompt = payload["messages"][0]["content"]
        if '"pairs"' in prompt:
            return {"choices": [{"message": {"content": "not json"}}]}
        if '"candidate": "first"' in prompt:
            return {"choices": [{"message": {"content": '{"match":true}'}}]}
        if '"candidate": "second"' in prompt:
            return {"choices": [{"message": {"content": '{"match":false}'}}]}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(client, "_post", fake_post)
    matched = client.match_message_pairs(
        ["oracle", "oracle"],
        ["first", "first", "second"],
    )

    assert matched["counts"] == [2, 2, 0]
    assert matched["matrix"] == [
        [True, True],
        [True, True],
        [False, False],
    ]
    assert len(calls) == 3
    stats = client.stats()
    assert stats["semantic_batch_requests"] == 1
    assert stats["semantic_batch_failures"] == 1
    assert stats["semantic_retries"] == 2
    assert stats["semantic_individual_requests"] == 2
    assert stats["semantic_individual_failures"] == 0
    assert "retrying 2 unique pair" in caplog.text


def test_semantic_pair_matcher_raises_after_individual_failure(monkeypatch, caplog):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)
    calls = []

    def malformed_response(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(client, "_post", malformed_response)
    with pytest.raises(RuntimeError, match="1/1 unique pair"):
        client.match_message_pairs(
            ["oracle"],
            ["oracle", "candidate"],
        )

    assert len(calls) == 2
    stats = client.stats()
    assert stats["semantic_batch_failures"] == 1
    assert stats["semantic_individual_requests"] == 1
    assert stats["semantic_individual_failures"] == 1
    assert stats["semantic_failures"] == 1
    assert "aborting the state group" in caplog.text


def test_semantic_pair_matcher_rejects_string_boole(monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)

    def fake_post(payload):
        prompt = payload["messages"][0]["content"]
        key = "matches" if '"pairs"' in prompt else "match"
        value = '["false"]' if key == "matches" else '"false"'
        return {"choices": [{"message": {"content": f'{{"{key}":{value}}}'}}]}

    monkeypatch.setattr(client, "_post", fake_post)

    with pytest.raises(RuntimeError, match="1/1 unique pair"):
        client.match_message_pairs(["oracle"], ["candidate"])
    assert client.stats()["semantic_individual_failures"] == 1


def test_oracle_disables_parallel_tool_calls(monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ask the user"}}]}

    monkeypatch.setattr(client, "_post", fake_post)
    action = client._sample_once(messages=[], tools=[], seed=7)
    assert action.kind == "message"
    assert captured["parallel_tool_calls"] is False
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 20
    assert captured["min_p"] == 0.0
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}


def test_oracle_keeps_first_tool_call_when_provider_returns_parallel_calls(
    monkeypatch,
):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    client = TauTeacherClient(samples=3)
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "first",
                                "arguments": '{"id": 1}',
                            }
                        },
                        {
                            "function": {
                                "name": "second",
                                "arguments": '{"id": 2}',
                            }
                        },
                    ]
                }
            }
        ]
    }

    action = client._response_action(response)

    assert action.name == "first"
    assert action.arguments == {"id": 1}
    assert client.stats()["parallel_tool_calls_truncated"] == 1
