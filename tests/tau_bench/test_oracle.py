import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agent_system.environments.env_package.tau_bench.actions import ParsedAction
from agent_system.environments.env_package.tau_bench.oracle import (
    ORACLE_PROTOCOL_VERSION,
    OpenRouterOracleClient,
    build_teacher_messages,
)


def test_oracle_uses_three_independent_seeded_requests_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3, cache_path=str(tmp_path / "cache.jsonl"))
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
        tools=[],
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
        tools=[],
    )
    assert cached == actions
    assert len(seeds) == 3
    assert client.stats()["cache_hits"] == 1
    record = json.loads((tmp_path / "cache.jsonl").read_text().strip())
    assert record["protocol_version"] == ORACLE_PROTOCOL_VERSION == 6
    assert len(record["teacher_samples"]) == 3
    assert "oracle_actions" not in record


def test_oracle_v6_ignores_old_deduplicated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    cache_path = tmp_path / "legacy.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "protocol_version": 5,
                "state_fingerprint": "legacy-state",
                "model": "deepseek/deepseek-v4-flash",
                "samples": 3,
                "reasoning_effort": "xhigh",
                "max_tokens": 4096,
                "oracle_actions": [
                    {"kind": "message", "content": "deduplicated"}
                ],
            }
        )
        + "\n"
    )
    client = OpenRouterOracleClient(samples=3, cache_path=str(cache_path))
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
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3, cache_path=str(tmp_path / "cache.jsonl"))
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
            tools=[],
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


def test_teacher_context_is_student_visible_by_default_and_privileged_on_opt_in():
    visible_chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]
    privileged_context = {
        "user_scenario": {"instructions": "change flight"},
        "reference_resolution_actions": [
            {"name": "lookup", "arguments": {"id": "x"}}
        ],
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
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {
            "choices": [
                {"message": {"content": '{"matches":[false,true,false]}'}}
            ]
        }

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


def test_semantic_matcher_returns_one_empty_row_per_candidate_without_teacher_messages(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)

    assert client.match_message_pairs([], ["first", "second"]) == {
        "counts": [0, 0],
        "matrix": [[], []],
    }


def test_semantic_pair_matcher_falls_back_to_unique_pairs(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
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


def test_semantic_pair_matcher_falls_back_to_false_after_failure(
    monkeypatch, caplog
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    calls = []

    def malformed_response(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(client, "_post", malformed_response)
    matched = client.match_message_pairs(
        ["oracle"],
        ["oracle", "candidate"],
    )

    assert matched == {
        "counts": [1, 0],
        "matrix": [[True], [False]],
    }
    assert len(calls) == 2
    stats = client.stats()
    assert stats["semantic_batch_failures"] == 1
    assert stats["semantic_individual_requests"] == 1
    assert stats["semantic_individual_failures"] == 1
    assert stats["semantic_failures"] == 1
    assert "treating it as a non-match" in caplog.text


def test_semantic_pair_matcher_rejects_string_boole(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)

    def fake_post(payload):
        prompt = payload["messages"][0]["content"]
        key = "matches" if '"pairs"' in prompt else "match"
        value = '["false"]' if key == "matches" else '"false"'
        return {
            "choices": [
                {"message": {"content": f'{{"{key}":{value}}}'}}
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    assert client.match_message_pairs(["oracle"], ["candidate"])["counts"] == [0]
    assert client.stats()["semantic_individual_failures"] == 1


def test_oracle_disables_parallel_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ask the user"}}]}

    monkeypatch.setattr(client, "_post", fake_post)
    action = client._sample_once(messages=[], tools=[], seed=7)
    assert action.kind == "message"
    assert captured["parallel_tool_calls"] is False


def test_oracle_keeps_first_tool_call_when_provider_returns_parallel_calls(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
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
