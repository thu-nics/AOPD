"""The release exposes standard training, without experiment-specific modes."""

import inspect

import pytest
from test_recipes import runtime

from aopd import data
from aopd.launch import RECIPES, build_plan


def test_public_recipes_are_main_and_tau():
    assert RECIPES == ("main", "tau")


@pytest.mark.parametrize("recipe", ["tau-full", "tau-a1", "tau-a4", "tau-a5", "tau-s1", "tau-s2"])
def test_removed_recipes_fail_before_launch(recipe, tmp_path):
    with pytest.raises(ValueError, match="unknown recipe"):
        build_plan(recipe, runtime(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_standard_tau_needs_external_roles_but_no_dataset_bundle(tmp_path):
    plan = build_plan("tau", runtime(), tmp_path)
    assert plan["active_roles"] == ["matcher", "teacher", "user"]
    expected = {
        "AIRLINE_TRAJ": "8",
        "RETAIL_TRAJ": "8",
        "TELECOM_TRAJ": "8",
        "TRAIN_STEPS": "50",
        "ROLLOUT_N": "4",
        "LR": "1e-06",
        "WARMUP_STEPS": "0",
        "FREQUENCY_BONUS_SCALE": "0.5",
        "STATE_GROUP_ADVANTAGE_MODE": "mean_then_batch_whiten",
        "MAX_PROMPT": "24576",
        "MAX_RESPONSE": "4096",
        "MAX_MODEL_LEN": "32768",
    }
    assert {key: plan["env"][key] for key in expected} == expected
    assert not any("SELF" in key or "ABLATION" in key for key in plan["env"])


def test_frozen_data_contains_only_main_environment_evidence():
    assert len(data.RELEASE_FILE_SHA256) == 13
    assert all(path.startswith(("awm/", "envscaler/")) for path in data.RELEASE_FILE_SHA256)
    assert list(inspect.signature(data.bundle).parameters) == ["research_runs", "output"]
