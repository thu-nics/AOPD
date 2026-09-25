"""Standard Tau scheduling, reward safeguards and external teacher contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_protocol import _tau_scoring_worker

from agent_system.environments.env_package.tau_bench.matcher_profiles import MESSAGE_CONCISE, TOOL_CONCISE, order_evidence
from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient
from examples.tau_bench.train.prepare_data import build_train_rows


def test_three_domain_schedule_and_cycle_are_fixed():
    pools = {d: [{"id": str(i)} for i in range(n)] for d, n in zip(("airline", "retail", "telecom"), (30, 74, 74), strict=True)}
    rows = build_train_rows(pools, counts=dict.fromkeys(pools, 8), num_batches=10)
    assert len(rows) == 240
    for offset in range(0, 240, 24):
        assert [r["env_kwargs"]["domain"] for r in rows[offset : offset + 24]] == list(pools) * 8
    for domain, size in (("airline", 30), ("retail", 74), ("telecom", 74)):
        ids = [r["env_kwargs"]["task_id"] for r in rows if r["env_kwargs"]["domain"] == domain]
        assert ids == [str(i % size) for i in range(80)]


def test_three_domain_metrics_include_telecom_and_keep_separate_outcomes():
    from agent_system.environments.env_package.tau_bench.manager import TauBenchEnvironmentManager

    manager = object.__new__(TauBenchEnvironmentManager)
    manager.oracle_actor = None
    episodes = [
        [dict(tau_domain=domain, terminal_success=success, protocol_reward=float(success), action_kind="tool", is_action_valid=valid, move_optimal=hit, oracle_set_size=2)]
        for domain, success, valid, hit in [("airline", True, True, True), ("retail", False, True, False), ("telecom", False, False, False), ("telecom", True, True, True)]
    ]
    metrics = manager.success_evaluator(total_infos=episodes)
    for domain, count in (("airline", 1), ("retail", 1), ("telecom", 2)):
        assert metrics[f"env/{domain}/trajectory_count"].tolist() == [count]
        assert metrics[f"env/{domain}/trajectory_share"].tolist() == [count / 4]
    for name in ("success_rate", "valid_action_rate", "oracle_hit_rate", "protocol_reward"):
        assert metrics[f"env/telecom/{name}"].tolist() == [0, 1]
    assert metrics["env/airline/success_rate"].tolist() == [1]
    assert metrics["env/retail/success_rate"].tolist() == [0]


@pytest.mark.parametrize("mode", ["appearance", "frequency_weighted"])
def test_standard_rewards_keep_minus_one_transfer_guard(mode):
    from agent_system.environments.env_package.tau_bench.actions import TRANSFER_HANDOFF_MESSAGE

    worker = _tau_scoring_worker(mode)
    worker.transfer_reward_guard_enabled = True
    worker._transfer_succeeded = False
    rows = asyncio.run(worker.step_candidate_group([TRANSFER_HANDOFF_MESSAGE, "B", "C", ""]))[0]
    assert rows[0][1] == -1 and rows[0][3]["transfer_without_tool"]
    assert rows[0][3]["move_optimal"] is False


def test_internal_budget_is_distinct_from_agent_decision_limit():
    worker = _tau_scoring_worker("frequency_weighted")
    worker._last_step_hit_decision_limit = False
    worker._env = SimpleNamespace(_simulation_run=SimpleNamespace(termination_reason="max_steps"))
    assert worker._terminal_reason(True) == "internal_step_limit"
    assert worker._terminal_reason(False) is None
    worker._last_step_hit_decision_limit = True
    assert worker._terminal_reason(True) == "decision_limit"


def test_disabled_matcher_ignores_stale_profile(monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    client = TauTeacherClient(matcher_enabled=False, matcher_profile="qwen38_concise", matcher_enable_thinking=True, matcher_max_tokens=128)
    assert client.matcher_profile == "default"
    assert client.matcher_enabled is False


def test_qwen38_profile_payload_and_cache_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    path = str(tmp_path / "matcher.jsonl")
    captured = []
    client = TauTeacherClient(matcher_profile="qwen38_concise", matcher_cache_path=path)

    def post(payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": '{"equivalent":true}'}}]}

    monkeypatch.setattr(client, "_post_matcher", post)
    result = client.match_message_pairs(["Teacher text"], ["Candidate text"], [{"role": "user", "content": "context"}], [])
    assert result["counts"] == [1]
    payload = captured[0]
    assert payload["temperature"] == 0.7 and payload["top_p"] == 0.8
    assert payload["max_tokens"] == 32768 and payload["top_k"] == 20
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["messages"][0]["content"] == MESSAGE_CONCISE
    assert list(json.loads(payload["messages"][1]["content"]))[:2] == ["public_context", "tools"]
    assert client.tool_instruction == TOOL_CONCISE
    assert client.tool_matcher_decoding == client.matcher_decoding
    reloaded = TauTeacherClient(matcher_profile="qwen38_concise", matcher_cache_path=path)
    assert reloaded.stats()["matcher_cache_records_loaded"] == 1
    other = TauTeacherClient(matcher_cache_path=path)
    assert other.stats()["matcher_cache_records_loaded"] == 0
    assert list(order_evidence({"candidate_arguments": {}, "public_context": [], "tool": {}, "teacher_arguments": {}}, tool=True)) == ["public_context", "tool", "teacher_arguments", "candidate_arguments"]


def test_teacher_import_across_endpoint_preserves_model_and_decoding_checks(monkeypatch, tmp_path):
    from agent_system.environments.env_package.tau_bench.actions import ParsedAction

    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    source = tmp_path / "teacher.jsonl"
    original = TauTeacherClient(api_base="http://remote/v1", matcher_enabled=False, cache_path=str(source))
    monkeypatch.setattr(original, "_sample_once", lambda **kw: ParsedAction(kind="message", content="Hello"))
    kwargs = dict(state_fingerprint="s", messages=[{"role": "user", "content": "hi"}], tools=[])
    original.sample_multiset(**kwargs)
    content = source.read_bytes()
    local = TauTeacherClient(api_base="http://localhost/v1", matcher_enabled=False, cache_path=str(tmp_path / "new.jsonl"), teacher_cache_import_paths=[str(source)])
    monkeypatch.setattr(local, "_sample_once", lambda **kw: pytest.fail("must reuse"))
    assert len(local.sample_multiset(**kwargs)) == 3
    assert source.read_bytes() == content
    for change in ({"model": "different-model"}, {"temperature": 0.7}):
        other = TauTeacherClient(api_base="http://localhost/v1", matcher_enabled=False, teacher_cache_import_paths=[str(source)], **change)
        assert other._import_samples("s", kwargs["messages"], [], "student_visible") == []
