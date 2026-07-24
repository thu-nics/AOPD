import json
import random
from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.tau_bench.envs import (
    QUALIFICATION_PROTOCOL_VERSION,
    TAU2_COMMIT,
    TERMINAL_REWARD_PROTOCOL,
    TauBenchWorker,
    compatibility_patch_sha256,
    interleave_grouped_domains,
    load_qualification_manifest,
    select_uniform_argmax,
    validate_tau_runtime_protocol,
)
from agent_system.environments.env_package.tau_bench.oracle import (
    ORACLE_PROTOCOL_VERSION,
)


def manifest_payload(*, airline=20, retail=50):
    return {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "terminal_reward_protocol": TERMINAL_REWARD_PROTOCOL,
        "tau2_commit": TAU2_COMMIT,
        "tau2_compatibility_patch_sha256": compatibility_patch_sha256(),
        "expert_model": "deepseek/deepseek-v4-flash",
        "user_llm": "openrouter/qwen/qwen3.6-27b",
        "user_temperature": 0.0,
        "user_reasoning_enabled": False,
        "oracle_reasoning_effort": "xhigh",
        "oracle_max_tokens": 4096,
        "oracle_samples_per_state": 3,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "trials_per_task": 4,
        "stable_tasks": {
            "airline": [f"a{i}" for i in range(airline)],
            "retail": [f"r{i}" for i in range(retail)],
        },
        "test_tasks": {
            "airline": [f"ta{i}" for i in range(20)],
            "retail": [f"tr{i}" for i in range(40)],
        },
    }


def test_grouped_domain_schedule_keeps_outcome_replicas_contiguous():
    labels = interleave_grouped_domains({"airline": 4, "retail": 4}, group_n=4)
    assert len(labels) == 32
    assert all(len(set(labels[index : index + 4])) == 1 for index in range(0, 32, 4))
    assert labels.count("airline") == labels.count("retail") == 16


def test_uniform_argmax_never_selects_lower_reward():
    rng = random.Random(3)
    selected = {select_uniform_argmax([-1.0, 0.0, 1.0, 1.0], rng) for _ in range(100)}
    assert selected == {2, 3}


def test_all_invalid_candidates_still_use_uniform_argmax():
    rng = random.Random(11)
    selected = {select_uniform_argmax([-1.0] * 4, rng) for _ in range(100)}
    assert selected == {0, 1, 2, 3}


def test_manifest_hard_fails_below_qualification_threshold(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest_payload(airline=19)),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Airline=19"):
        load_qualification_manifest(path)


def test_manifest_accepts_shared_formal_subset(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest_payload()),
        encoding="utf-8",
    )
    manifest = load_qualification_manifest(path)
    assert len(manifest["stable_tasks"]["retail"]) == 50


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", 999, "protocol_version mismatch"),
        ("tau2_commit", "wrong", "pinned commit"),
        ("tau2_compatibility_patch_sha256", "wrong", "patch mismatch"),
        ("user_reasoning_enabled", True, "disable user-simulator reasoning"),
        ("oracle_samples_per_state", 2, "three oracle samples"),
        ("oracle_protocol_version", 2, "oracle protocol mismatch"),
        ("trials_per_task", 3, "four trials"),
    ],
)
def test_manifest_rejects_protocol_drift(tmp_path, field, value, message):
    payload = manifest_payload()
    payload[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        load_qualification_manifest(path)


def test_runtime_protocol_matches_user_and_oracle_configuration():
    payload = manifest_payload()
    config = SimpleNamespace(
        user_llm="openrouter/qwen/qwen3.6-27b",
        user_temperature=0.0,
        user_reasoning_enabled=False,
        oracle=SimpleNamespace(
            model="deepseek/deepseek-v4-flash",
            samples=3,
            reasoning_effort="xhigh",
            max_tokens=4096,
        ),
    )
    validate_tau_runtime_protocol(payload, config, require_oracle=True)
    config.oracle.samples = 2
    with pytest.raises(RuntimeError, match="oracle_samples_per_state"):
        validate_tau_runtime_protocol(payload, config, require_oracle=True)


def test_runtime_protocol_rejects_user_simulator_drift():
    payload = manifest_payload()
    config = SimpleNamespace(
        user_llm="different-user",
        user_temperature=0.0,
        user_reasoning_enabled=False,
    )
    with pytest.raises(RuntimeError, match="user_llm"):
        validate_tau_runtime_protocol(payload, config, require_oracle=False)


def test_qualification_resume_protocol_rejects_mixed_runs(tmp_path):
    from examples.tau_bench.qualify_expert import ensure_resume_protocol

    path = tmp_path / "qualification_protocol.json"
    ensure_resume_protocol(path, {"model": "a", "max_steps": 30})
    ensure_resume_protocol(path, {"model": "a", "max_steps": 30})
    with pytest.raises(RuntimeError, match="different protocol"):
        ensure_resume_protocol(path, {"model": "b", "max_steps": 30})


def test_finished_tau_worker_step_is_an_idempotent_zero_reward_noop():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=0.0,
        user_reasoning_enabled=False,
    )
    worker._done = True
    worker._last_observation = "terminal observation"
    worker._last_info = {"protocol_reward": 1.0}

    observation, reward, done, info = worker.step("ignored action")

    assert observation == "terminal observation"
    assert reward == 0.0
    assert done is True
    assert info["terminal_reason"] == "already_done"
    assert info["protocol_reward"] == 0.0



def test_terminal_reward_uses_db_and_communicate_but_not_nl(monkeypatch):
    from agent_system.environments.env_package.tau_bench.envs import (
        _db_communicate_reward,
    )
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator import evaluator

    calls = []

    class Result:
        def __init__(self, reward):
            self.reward = reward

        def model_dump(self, mode):
            assert mode == "json"
            return {"reward": self.reward}

    def fake_evaluate_simulation(*, evaluation_type, **kwargs):
        calls.append(evaluation_type.value)
        return Result({"env": 0.5, "communicate": 0.25}[evaluation_type.value])

    monkeypatch.setattr(evaluator, "evaluate_simulation", fake_evaluate_simulation)
    task = SimpleNamespace(
        evaluation_criteria=SimpleNamespace(
            reward_basis=[
                RewardType.DB,
                RewardType.COMMUNICATE,
                RewardType.NL_ASSERTION,
            ]
        )
    )
    fake_env = SimpleNamespace(
        _simulation_run=object(),
        _get_task=lambda: task,
        solo_mode=False,
        domain="airline",
    )

    reward, info = _db_communicate_reward(fake_env)

    assert reward == 0.125
    assert calls == ["env", "communicate"]
    assert json.loads(info)["protocol"] == "tau_db_x_communicate"


def test_qualification_retries_transient_trial_errors(monkeypatch):
    from examples.tau_bench import qualify_expert

    responses = iter(
        [
            {"error": "temporary"},
            {"error": "temporary"},
            {"success": True},
        ]
    )
    monkeypatch.setattr(qualify_expert, "run_trial", lambda **kwargs: next(responses))

    result = qualify_expert.run_trial_with_retries(task_id="task")

    assert result["success"] is True
    assert result["attempt"] == 3
