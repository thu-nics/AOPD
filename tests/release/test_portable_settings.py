"""Portable launch paths and explicit trainer-validation configuration."""

import copy
import sys

import pytest
from test_recipes import runtime

from aopd import runtime as runtime_module
from aopd.launch import ROOT, build_plan


def main_runtime():
    result = runtime()
    result["sources"].update({"awm": "deps/awm", "awm_data": "data/awm", "envscaler": "deps/envscaler"})
    result["roles"].update({name: {"service": "remote"} for name in ("runtime_judge", "terminal_judge")})
    result["data"] = {"awm_pool": "data/awm.parquet", "awm_manifest": "data/awm.json", "envscaler_pool": "data/env.parquet", "envscaler_manifest": "data/env.json"}
    return result


def validation_settings():
    return {"every_steps": 7, "before_train": True, "domains": ["airline", "retail"], "split": "test", "trials": 2, "batch_size": 8, "max_steps": 19}


def add_validation_user(result):
    result["services"]["validation"] = {"mode": "api", "provider": "openai-compatible", "model": "validation-model", "base_url": "https://validation.invalid/v1", "api_key_env": "VALIDATION_ONLY_KEY"}
    result["roles"]["validation_user"] = {"service": "validation", "generation": {"enable_thinking": False, "temperature": 0.3, "top_p": 0.75, "max_tokens": 1234}}


def test_main_rejects_shuffle_instead_of_silently_overriding_it(tmp_path):
    result = main_runtime()
    result["training"] = {"SHUFFLE": True}
    with pytest.raises(ValueError, match="(?i)shuffle"):
        build_plan("main", result, tmp_path)


def test_runtime_paths_normalize_once_without_mutating_role_identities(tmp_path):
    result = main_runtime()
    result.update(model="models/student", python="venv/bin/python")
    result["sources"] = {"tau": "deps/tau", "awm": "deps/awm", "awm_data": "data/awm", "envscaler": "deps/envscaler"}
    result["services"]["local"] = {"mode": "local", "model": "served-name", "model_path": "models/teacher", "python": "serve/bin/python", "gpus": [0], "port": 9001}
    before = copy.deepcopy(result)

    normalized = runtime_module.normalize_runtime_paths(result, tmp_path)

    assert result == before
    assert normalized["model"] == str(tmp_path / "models/student")
    assert normalized["python"] == str(tmp_path / "venv/bin/python")
    for section in ("sources", "data"):
        assert normalized[section] == {key: str(tmp_path / value) for key, value in before[section].items()}
    local = normalized["services"]["local"]
    assert local["model_path"] == str(tmp_path / "models/teacher")
    assert local["python"] == str(tmp_path / "serve/bin/python")
    assert local["model"] == "served-name"
    assert normalized["services"]["remote"] == before["services"]["remote"]
    assert runtime_module.normalize_runtime_paths(normalized, tmp_path / "other") == normalized


def test_interpreter_normalization_preserves_virtualenv_symlink(tmp_path):
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    result = runtime_module.normalize_runtime_paths({"model": "model", "python": "venv/bin/python", "services": {"aux": {"mode": "local", "model_path": "aux", "python": "venv/bin/python"}}}, tmp_path)
    assert result["python"] == str(executable)
    assert result["services"]["aux"]["python"] == str(executable)


def test_plan_anchors_runtime_paths_to_release_and_cli_paths_to_cwd(monkeypatch, tmp_path):
    result = main_runtime()
    result.update(model="models/student", python="venv/bin/python")
    result["sources"] = {"awm": "deps/awm", "awm_data": "data/awm", "envscaler": "deps/envscaler"}
    monkeypatch.chdir(tmp_path)

    plan = build_plan("main", result, "run", resume="checkpoint")

    assert plan["env"]["MODEL_PATH"] == str(ROOT / "models/student")
    assert plan["env"]["PYTHON"] == str(ROOT / "venv/bin/python")
    assert plan["env"]["TRAIN_DATA"] == str(ROOT / "data/awm.parquet")
    assert plan["env"]["ENVSCALER_MANIFEST"] == str(ROOT / "data/env.json")
    assert plan["env"]["AWM_SOURCE_DIR"] == str(ROOT / "deps/awm")
    assert plan["env"]["ENVSCALER_ROOT"] == str(ROOT / "deps/envscaler")
    assert plan["env"]["RUN_DIR"] == str(tmp_path / "run")
    assert plan["env"]["RESUME_FROM_PATH"] == str(tmp_path / "checkpoint")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("missing", ["model", "data", "python"])
def test_runtime_path_preflight_rejects_missing_inputs(tmp_path, missing):
    result = runtime()
    model = tmp_path / "student"
    model.mkdir()
    source = tmp_path / "tau"
    source.mkdir()
    result.update(model=str(model), python=sys.executable, sources={"tau": str(source)})
    if missing == "model":
        result["model"] = str(tmp_path / "missing-model")
    elif missing == "python":
        result["python"] = str(tmp_path / "missing-python")
    else:
        result["data"] = {"customer_briefs": str(tmp_path / "missing-briefs.json")}

    with pytest.raises((ValueError, FileNotFoundError), match="missing"):
        runtime_module.validate_runtime_paths(result)


def test_missing_launch_input_fails_before_gpu_checks_or_service_start(monkeypatch, tmp_path):
    from aopd import __main__ as launcher

    result = runtime()
    result["model"] = str(tmp_path / "missing-model")
    run_dir = tmp_path / "not-created"
    plan = build_plan("tau-full", result, run_dir)

    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("Missing launch inputs reached GPU or service operations")

    monkeypatch.setattr(launcher, "require_free_gpus", unexpected_side_effect)
    monkeypatch.setattr(launcher, "local_services", unexpected_side_effect)
    monkeypatch.setattr(launcher, "probe_roles", unexpected_side_effect)
    with pytest.raises((ValueError, FileNotFoundError), match="missing"):
        launcher.run_plan(plan, result, run_dir)
    assert not run_dir.exists()


@pytest.mark.parametrize("recipe", ["main", "tau-full"])
def test_disabled_validation_does_not_resolve_or_activate_its_role(tmp_path, recipe):
    result = main_runtime() if recipe == "main" else runtime()
    result["validation"] = {"every_steps": -1, "before_train": False}
    result["roles"]["validation_user"] = {"service": "not-configured"}

    plan = build_plan(recipe, result, tmp_path)

    assert "validation_user" not in plan["active_roles"]
    assert plan["env"]["TEST_FREQ"] == "-1"
    assert plan["env"]["VAL_BEFORE_TRAIN"] == "false"
    assert not any(key.startswith("TAU_VALIDATION_USER_") for key in plan["env"])


@pytest.mark.parametrize("missing", ["tau", "validation_user"])
def test_main_validation_requires_explicit_source_and_user(tmp_path, missing):
    result = main_runtime()
    result["validation"] = validation_settings()
    add_validation_user(result)
    if missing == "tau":
        del result["sources"]["tau"]
    else:
        del result["roles"]["validation_user"]

    with pytest.raises(ValueError, match=missing):
        build_plan("main", result, tmp_path)


def test_main_validation_uses_its_dedicated_user_and_schedule(tmp_path):
    result = main_runtime()
    result["validation"] = validation_settings()
    add_validation_user(result)

    plan = build_plan("main", result, tmp_path)

    assert "validation_user" in plan["active_roles"]
    assert plan["env"]["TEST_FREQ"] == "7"
    assert plan["env"]["VAL_BEFORE_TRAIN"] == "true"
    assert plan["env"]["VAL_BATCH"] == "8"
    assert plan["env"]["TAU_VALIDATION_USER_API_BASE"] == "https://validation.invalid/v1"
    assert plan["env"]["TAU_VALIDATION_USER_API_KEY_ENV"] == "VALIDATION_ONLY_KEY"
    assert plan["env"]["TAU_VALIDATION_USER_TEMPERATURE"] == "0.3"
    assert plan["env"]["TAU_VALIDATION_USER_MAX_TOKENS"] == "1234"
    assert "++env.tau.user_llm=openai/validation-model" in plan["command"]
    assert "++env.tau.user_reasoning_enabled=false" in plan["command"]


def test_tau_validation_reuses_training_user_and_forwards_all_controls(tmp_path):
    result = runtime()
    result["validation"] = validation_settings()
    result["roles"]["user"]["generation"] = {"enable_thinking": False, "temperature": 0.6, "top_p": 0.8, "max_tokens": 2048}

    plan = build_plan("tau-full", result, tmp_path)
    env = plan["env"]

    assert {key: env[key] for key in ("TEST_FREQ", "VAL_BEFORE_TRAIN", "VAL_BATCH", "VALIDATION_DOMAINS", "VALIDATION_SPLIT", "VALIDATION_TRIALS", "EVAL_MAX_STEPS")} == {
        "TEST_FREQ": "7",
        "VAL_BEFORE_TRAIN": "true",
        "VAL_BATCH": "8",
        "VALIDATION_DOMAINS": "airline,retail",
        "VALIDATION_SPLIT": "test",
        "VALIDATION_TRIALS": "2",
        "EVAL_MAX_STEPS": "19",
    }
    for field in ("PROVIDER", "API_BASE", "API_KEY_ENV", "TEMPERATURE", "TOP_P", "MAX_TOKENS"):
        assert env[f"TAU_VALIDATION_USER_{field}"] == env[f"TAU_USER_{field}"]
    assert "++env.tau.validation_user.user_llm=openai/qwen" in plan["command"]
    assert "++env.tau.validation_user.user_reasoning_enabled=false" in plan["command"]


def test_tau_validation_override_keeps_training_user_unchanged(tmp_path):
    result = runtime()
    result["validation"] = {"before_train": True}
    add_validation_user(result)

    plan = build_plan("tau-full", result, tmp_path)

    assert "validation_user" in plan["active_roles"]
    assert plan["env"]["TAU_USER_MODEL"] == "openai/qwen"
    assert plan["env"]["TAU_USER_API_BASE"] == "http://localhost:9000/v1"
    assert "++env.tau.validation_user.user_llm=openai/validation-model" in plan["command"]
    assert plan["env"]["TAU_VALIDATION_USER_API_KEY_ENV"] == "VALIDATION_ONLY_KEY"
    assert plan["env"]["TAU_VALIDATION_USER_TOP_P"] == "0.75"


@pytest.mark.parametrize("setting,value", [("TEST_FREQ", 5), ("VAL_BEFORE_TRAIN", True)])
def test_training_mapping_cannot_bypass_validation_configuration(tmp_path, setting, value):
    result = runtime()
    result["training"] = {setting: value}
    with pytest.raises(ValueError, match="(?i)validation|reserved"):
        build_plan("tau-full", result, tmp_path)
