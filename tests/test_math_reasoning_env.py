from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.math_reasoning.envs import (
    MathReasoningEnvironmentManager,
    build_math_reasoning_envs,
)


def test_math_reasoning_environment_scores_dapo_answers():
    envs = build_math_reasoning_envs(env_num=1, group_n=2)
    manager = MathReasoningEnvironmentManager(
        envs,
        SimpleNamespace(env=SimpleNamespace()),
    )
    observations, infos = manager.reset(
        kwargs=[
            {"question": "1+1?", "ground_truth": "2", "data_source": "dapo"},
            {"question": "2+2?", "ground_truth": "4", "data_source": "dapo"},
        ]
    )
    assert observations["text"] == ["1+1?", "2+2?"]
    assert [info["data_source"] for info in infos] == ["dapo", "dapo"]

    _, rewards, dones, step_infos = manager.step(
        [r"Answer: \boxed{2}", r"Answer: \boxed{5}"]
    )
    assert rewards.tolist() == pytest.approx([1.0, -1.0])
    assert dones.tolist() == [True, True]
    assert [info["won"] for info in step_infos] == [True, False]
    assert all(info["is_action_valid"] for info in step_infos)


def test_math_reasoning_environment_requires_exact_batch():
    envs = build_math_reasoning_envs(env_num=2, group_n=2)
    with pytest.raises(ValueError, match="Expected 4 math items"):
        envs.reset([{"question": "x", "ground_truth": "x"}])


def test_math_reasoning_state_group_scores_all_candidates_and_selects_best():
    envs = build_math_reasoning_envs(env_num=1, group_n=1)
    manager = MathReasoningEnvironmentManager(
        envs,
        SimpleNamespace(
            env=SimpleNamespace(
                rollout=SimpleNamespace(
                    selection_mode="best",
                    random_select_prob=0.0,
                )
            )
        ),
    )
    manager.reset(
        kwargs=[
            {
                "question": "1+1?",
                "ground_truth": "2",
                "data_source": "math_dapo",
            }
        ]
    )

    candidates, selected, _, rewards, dones, infos = manager.state_group_step(
        [["not an answer", r"Answer: \boxed{2}"]]
    )

    assert [candidate[1] for candidate in candidates[0]] == [-1.0, 1.0]
    assert candidates[0][0][3]["parse_ok"] is False
    assert candidates[0][0][3]["is_action_valid"] == 0
    assert selected.tolist() == [1]
    assert rewards.tolist() == pytest.approx([1.0])
    assert dones.tolist() == [True]
    assert infos[0]["vpr_game"] == "math"
