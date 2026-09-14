"""Cross-environment pair protocol: decoding, vote restoration and single-flight."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_system.environments.semantic_matcher import match_message_matrix, matcher_boolean, matcher_decoding_config


def response(value=True, finish="stop"):
    return {"model": "deepseek-flash", "choices": [{"finish_reason": finish, "message": {"content": json.dumps({"equivalent": value})}}]}


def test_deepseek_message_and_tool_default_uses_low_thinking_without_sampling_overrides():
    config = matcher_decoding_config("deepseek")
    assert config == {"thinking": {"type": "enabled"}, "reasoning_effort": "low", "max_tokens": 32768, "stream": False, "response_format": {"type": "json_object"}}
    disabled = matcher_decoding_config("deepseek", enable_thinking=False, max_tokens=1024)
    assert disabled["thinking"] == {"type": "disabled"}
    assert (disabled["temperature"], disabled["top_p"], disabled["max_tokens"]) == (0.0, 1.0, 1024)
    assert "reasoning_effort" not in disabled


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": True}, {"enable_thinking": "false"}, {"reasoning_effort": "invalid"}, {"enable_thinking": False, "reasoning_effort": "low"}])
def test_invalid_decoding_fails_loudly(kwargs):
    with pytest.raises(ValueError):
        matcher_decoding_config("deepseek", **kwargs)


@pytest.mark.parametrize("value,finish", [("true", "stop"), (1, "stop"), (True, "length"), (False, "content_filter")])
def test_incomplete_or_non_boolean_is_not_a_verdict(value, finish):
    with pytest.raises(ValueError):
        matcher_boolean(response(value, finish))


def test_pair_parallelism_deduplicates_work_and_restores_multiset():
    barrier = threading.Barrier(2)
    calls = []

    def pair(teacher, candidate):
        calls.append((teacher, candidate))
        barrier.wait(timeout=5)
        return candidate == "A"

    result = match_message_matrix(["T"] * 3, ["A", "A", "B"], pair, max_workers=2)
    assert result == {"counts": [3, 3, 0], "matrix": [[True] * 3, [True] * 3, [False] * 3]}
    assert sorted(calls) == [("T", "A"), ("T", "B")]


@pytest.mark.parametrize("family", ["tau", "awm"])
@pytest.mark.parametrize("kind", ["message", "tool"])
def test_clients_share_decoding_and_singleflight(monkeypatch, tmp_path, family, kind):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    entered, release = threading.Event(), threading.Event()
    calls = []

    def post(payload, **kwargs):
        calls.append(payload)
        entered.set()
        assert release.wait(timeout=5)
        return response()

    path = str(tmp_path / "matcher.jsonl")
    if family == "tau":
        from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient

        client = TauTeacherClient(matcher_provider="deepseek", matcher_model="deepseek-v4-flash", matcher_api_base="https://api.deepseek.com", matcher_api_key_env="DEEPSEEK_API_KEY", matcher_cache_path=path)
        monkeypatch.setattr(client, "_post", post)
    else:
        from agent_system.environments.env_package.awm.runtime.oracle import DeepSeekAWMOracleClient

        client = DeepSeekAWMOracleClient(request_fn=post, matcher_cache_path=path)
    if kind == "message":
        operation = lambda: client.match_message_pairs(["teacher"] * 3, ["candidate", "candidate"])
        expected = {"counts": [3, 3], "matrix": [[True] * 3, [True] * 3]}
    else:
        operation = lambda: client.match_tool_argument_pairs([{"tool": "lookup"}] * 3)
        expected = [True] * 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(operation) for _ in range(4)]
        assert entered.wait(timeout=5)
        release.set()
        assert [future.result(timeout=5) for future in futures] == [expected] * 4
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_tokens"] == 32768
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert "temperature" not in calls[0] and "top_p" not in calls[0]
    assert operation() == expected  # False/true cache hits also retain all vote positions.
    assert len(calls) == 1


@pytest.mark.parametrize("family", ["tau", "awm"])
@pytest.mark.parametrize("kind", ["message", "tool"])
def test_clients_do_not_cache_truncation_and_can_retry(monkeypatch, family, kind):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    replies = iter([response(finish="length"), response(False)])
    if family == "tau":
        from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient

        client = TauTeacherClient()
        monkeypatch.setattr(client, "_post", lambda *a, **k: next(replies))
    else:
        from agent_system.environments.env_package.awm.runtime.oracle import DeepSeekAWMOracleClient

        client = DeepSeekAWMOracleClient(request_fn=lambda *a, **k: next(replies))
    operation = (lambda: client.match_message_pairs(["teacher"], ["candidate"])) if kind == "message" else (lambda: client.match_tool_argument_pairs([{"tool": "lookup"}]))
    with pytest.raises(ValueError, match="incomplete"):
        operation()
    assert not client._matcher_cache
    assert operation() == ({"counts": [0], "matrix": [[False]]} if kind == "message" else [False])
