from pathlib import Path

import pytest

from aopd.launch import build_plan


def runtime():
    return {
        "model": "/models/student",
        "student": {"gpus": [4, 7], "tp": 1, "sp": 2},
        "services": {"remote": {"mode": "api", "provider": "openai-compatible", "model": "qwen", "base_url": "http://localhost:9000/v1"}},
        "roles": {name: {"service": "remote"} for name in ["teacher", "user", "matcher"]},
        "sources": {"tau": "/deps/tau", "awm": "/deps/awm", "awm_data": "/deps/awm-data", "envscaler": "/deps/envscaler"},
    }


def test_tau_is_three_domain_and_no_automatic_validation(tmp_path):
    plan = build_plan("tau", runtime(), tmp_path / "run")
    env = plan["env"]
    assert [env[k] for k in ["AIRLINE_TRAJ", "RETAIL_TRAJ", "TELECOM_TRAJ"]] == ["8"] * 3
    assert env["TRAIN_STEPS"] == "50"
    assert env["CUDA_VISIBLE_DEVICES"] == "4,7"
    assert env["N_GPUS"] == "2"
    assert env["VAL_BEFORE_TRAIN"] == "false"
    assert env["TEST_FREQ"] == "-1"


def test_tau_requires_api_teacher(tmp_path):
    r = runtime()
    del r["roles"]["teacher"]
    with pytest.raises(ValueError, match="missing API roles.*teacher"):
        build_plan("tau", r, tmp_path)


def test_main_roles_do_not_implicitly_follow_teacher(tmp_path):
    r = runtime()
    r["roles"].update({name: {"service": "remote"} for name in ["terminal_judge", "runtime_judge"]})
    r["data"] = {"awm_pool": "/data/awm.parquet", "awm_manifest": "/data/awm.json", "envscaler_pool": "/data/env.parquet", "envscaler_manifest": "/data/env.json"}
    plan = build_plan("main", r, tmp_path)
    assert plan["env"]["AWM_PER_STEP"] == "59"
    assert plan["env"]["ENVSCALER_PER_STEP"] == "5"
    assert plan["env"]["TRAIN_STEPS"] == "100"
    assert "scripts" not in str(Path(plan["command"][1]).name)


def test_check_does_not_create_run_or_contact_server(tmp_path):
    target = tmp_path / "absent"
    build_plan("tau", runtime(), target)
    assert not target.exists()


def test_invalid_token_budget_is_rejected_before_services(tmp_path):
    r = runtime()
    r["student"]["ppo_tokens_per_gpu"] = 1024
    with pytest.raises(ValueError, match="token budget"):
        build_plan("tau", r, tmp_path)


def test_six_gpu_tau_requires_explicit_divisible_minibatch(tmp_path):
    r = runtime()
    r["student"]["gpus"] = list(range(6))
    with pytest.raises(ValueError, match="PPO_MINI_BATCH"):
        build_plan("tau", r, tmp_path)
    r["training"] = {"PPO_MINI_BATCH": 24}
    assert build_plan("tau", r, tmp_path)["env"]["PPO_MINI_BATCH"] == "24"


def test_unconsumed_generation_parameter_rejected(tmp_path):
    r = runtime()
    r["roles"]["user"]["generation"] = {"reasoning_effort": "high"}
    with pytest.raises(ValueError, match="user.*generation"):
        build_plan("tau", r, tmp_path)


def test_recipe_method_cannot_be_overridden_by_training_mapping(tmp_path):
    r = runtime()
    r["training"] = {"METHOD": "outcome"}
    with pytest.raises(ValueError, match="reserved"):
        build_plan("tau", r, tmp_path)


@pytest.mark.parametrize("smoke", [False, True])
def test_tau_plan_preserves_budget_and_uses_external_teacher(tmp_path, smoke):
    r = runtime()
    plan = build_plan("tau", r, tmp_path, smoke=smoke)
    env = plan["env"]
    assert plan["active_roles"] == ["matcher", "teacher", "user"]
    assert (env["MAX_PROMPT"], env["MAX_RESPONSE"], env["MAX_MODEL_LEN"]) == ("24576", "4096", "32768")
    if smoke:
        assert env["SMOKE_TRAIN_STEPS"] == "1"


def test_explicit_student_optimizer_offload_is_forwarded(tmp_path):
    r = runtime()
    r["student"]["optimizer_offload"] = True
    plan = build_plan("tau", r, tmp_path)
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" in plan["command"]
    r["student"]["optimizer_offload"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        build_plan("tau", r, tmp_path)


def test_legacy_shell_settings_do_not_leak_into_public_launch(monkeypatch):
    from aopd.launch import launch_environment

    monkeypatch.setenv("TAU_UNUSED_EXPERIMENT_FLAG", "true")
    monkeypatch.setenv("SMOKE", "1")
    monkeypatch.setenv("TEST_RELEASE_KEY", "secret-not-in-manifest")
    roles = {"teacher": {"api_key_env": "TEST_RELEASE_KEY"}}
    env = launch_environment({"env": {"SMOKE": "0"}}, roles)
    assert "TAU_UNUSED_EXPERIMENT_FLAG" not in env
    assert env["SMOKE"] == "0"
    assert env["TEST_RELEASE_KEY"] == "secret-not-in-manifest"
