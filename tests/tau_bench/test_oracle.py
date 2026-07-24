import threading

from agent_system.environments.env_package.tau_bench.actions import ParsedAction
from agent_system.environments.env_package.tau_bench.oracle import (
    OpenRouterOracleClient,
    build_expert_messages,
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
        messages=[{"role": "user", "content": "changed but fingerprint controls cache"}],
        tools=[],
    )
    assert cached == actions
    assert len(seeds) == 3
    assert client.stats()["cache_hits"] == 1


def test_expert_context_contains_privileged_reference_but_no_student_candidates():
    messages = build_expert_messages(
        policy="policy",
        task={
            "id": "task-1",
            "user_scenario": {"instructions": "change flight"},
            "evaluation_criteria": {
                "actions": [{"name": "lookup", "arguments": {"id": "x"}}]
            },
        },
        history=[{"role": "user", "content": "hello"}],
    )
    assert "reference_resolution_actions" in messages[0]["content"]
    assert "student" not in messages[0]["content"].lower()
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_semantic_matcher_deduplicates_and_retries_invalid_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    responses = iter(
        [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": '{"matches":[true]}'}}]},
        ]
    )
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return next(responses)

    monkeypatch.setattr(client, "_post", fake_post)
    matches = client.match_messages(
        ["I can help."],
        [" i can   HELP. ", "different", "different"],
    )
    assert matches == [True, True, True]
    assert len(calls) == 2
    assert '"candidate_messages": ["different"]' in calls[-1]["messages"][0]["content"]
    assert client.stats()["semantic_exact_matches"] == 1
    assert client.stats()["semantic_retries"] == 1


def test_semantic_matcher_fails_closed_after_malformed_responses(monkeypatch):
    import pytest

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    monkeypatch.setattr(
        client,
        "_post",
        lambda payload: {"choices": [{"message": {"content": "{}"}}]},
    )
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.match_messages(["oracle"], ["candidate"])


def test_semantic_matcher_rejects_string_boole(monkeypatch):
    import pytest

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    client = OpenRouterOracleClient(samples=3)
    monkeypatch.setattr(
        client,
        "_post",
        lambda payload: {
            "choices": [{"message": {"content": '{"matches":["false"]}'}}]
        },
    )
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.match_messages(["oracle"], ["candidate"])



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
