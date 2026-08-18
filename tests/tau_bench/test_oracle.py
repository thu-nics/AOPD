import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agent_system.environments.env_package.tau_bench.actions import ParsedAction
from agent_system.environments.env_package.tau_bench.oracle import (
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
    actions = client.sample_oracle_set(
        state_fingerprint="state-a",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )
    assert len(seeds) == 3
    assert len(set(seeds)) == 3
    assert len(actions) == 2

    cached = client.sample_oracle_set(
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
        return client.sample_oracle_set(
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


def test_semantic_matcher_deduplicates_and_uses_valid_batch_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": '{"matches":[true]}'}}]}

    monkeypatch.setattr(client, "_post", fake_post)
    matches = client.match_messages(
        ["I can help."],
        [" i can   HELP. ", "different", "different"],
    )

    assert matches == [True, True, True]
    assert len(calls) == 1
    assert '"candidate_messages": ["different"]' in calls[0]["messages"][0]["content"]
    assert client.stats()["semantic_exact_matches"] == 1
    assert client.stats()["semantic_batch_requests"] == 1
    assert client.stats()["semantic_batch_failures"] == 0
    assert client.stats()["semantic_individual_requests"] == 0


def test_semantic_matcher_falls_back_to_individual_candidates(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        prompt = payload["messages"][0]["content"]
        if '"candidate_messages"' in prompt:
            return {"choices": [{"message": {"content": "not json"}}]}
        if '"candidate_message": "first"' in prompt:
            return {"choices": [{"message": {"content": '{"match":true}'}}]}
        if '"candidate_message": "second"' in prompt:
            return {"choices": [{"message": {"content": '{"match":false}'}}]}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(client, "_post", fake_post)
    matches = client.match_messages(
        ["oracle"],
        ["first", "first", "second"],
    )

    assert matches == [True, True, False]
    assert len(calls) == 3
    stats = client.stats()
    assert stats["semantic_batch_requests"] == 1
    assert stats["semantic_batch_failures"] == 1
    assert stats["semantic_retries"] == 2
    assert stats["semantic_individual_requests"] == 2
    assert stats["semantic_individual_failures"] == 0
    assert stats["semantic_failures"] == 0
    assert "retrying 2 unresolved candidate message" in caplog.text


def test_semantic_matcher_falls_back_to_false_after_individual_failure(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    calls = []

    def malformed_response(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(client, "_post", malformed_response)
    matches = client.match_messages(
        ["oracle"],
        ["oracle", "candidate"],
    )

    assert matches == [True, False]
    assert len(calls) == 2
    stats = client.stats()
    assert stats["semantic_batch_failures"] == 1
    assert stats["semantic_individual_requests"] == 1
    assert stats["semantic_individual_failures"] == 1
    assert stats["semantic_failures"] == 1
    assert "treating it as a non-match" in caplog.text


def test_semantic_matcher_rejects_string_boole_at_both_levels(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)

    def fake_post(payload):
        prompt = payload["messages"][0]["content"]
        key = "matches" if '"candidate_messages"' in prompt else "match"
        value = '["false"]' if key == "matches" else '"false"'
        return {"choices": [{"message": {"content": f'{{"{key}":{value}}}'}}]}

    monkeypatch.setattr(client, "_post", fake_post)

    assert client.match_messages(["oracle"], ["candidate"]) == [False]
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
