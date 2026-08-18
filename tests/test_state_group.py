import numpy as np
import pytest

from agent_system.environments.teacher_reward import (
    teacher_match_reward,
    validate_teacher_reward_config,
)
from verl.trainer.ppo.state_group import (
    compute_state_group_row_advantages,
)


def test_group_whiten_uses_sample_std_per_state_group():
    advantages = compute_state_group_row_advantages(
        np.asarray([0.0, 2.0, 0.0, 4.0], dtype=np.float32),
        np.asarray(["a", "a", "b", "b"], dtype=object),
        np.ones(4, dtype=bool),
        mode="group_whiten",
        eps=1e-8,
    )

    expected = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(
        advantages,
        [-expected, expected, -expected, expected],
        rtol=1e-6,
    )


def test_mean_then_batch_whiten_preserves_between_group_scale():
    advantages = compute_state_group_row_advantages(
        np.asarray([0.0, 2.0, 0.0, 4.0], dtype=np.float32),
        np.asarray(["a", "a", "b", "b"], dtype=object),
        np.ones(4, dtype=bool),
        mode="mean_then_batch_whiten",
        eps=1e-8,
    )

    assert advantages[3] == pytest.approx(2.0 * advantages[1])
    assert advantages.std(ddof=0) == pytest.approx(1.0)


def test_environment_scope_whitens_mixed_domains_independently():
    advantages = compute_state_group_row_advantages(
        np.asarray([0.0, 2.0, 0.0, 4.0], dtype=np.float32),
        np.asarray(["a", "a", "b", "b"], dtype=object),
        np.ones(4, dtype=bool),
        mode="mean_then_batch_whiten",
        eps=1e-8,
        partitions=np.asarray(["sudoku", "sudoku", "sokoban", "sokoban"]),
    )

    np.testing.assert_allclose(advantages, [-1.0, 1.0, -1.0, 1.0])


@pytest.mark.parametrize("count", [1, 2, 3])
def test_appearance_reward_ignores_duplicate_frequency(count):
    assert (
        teacher_match_reward(
            count,
            teacher_sample_count=3,
            mode="appearance",
            frequency_bonus_scale=2.0,
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0.0), (1, 1.0), (2, 1.25), (3, 1.5)],
)
def test_frequency_weighted_reward_uses_soft_multiset_bonus(count, expected):
    assert teacher_match_reward(
        count,
        teacher_sample_count=3,
        mode="frequency_weighted",
        frequency_bonus_scale=0.5,
    ) == pytest.approx(expected)


def test_teacher_reward_config_rejects_unknown_mode_and_invalid_scale():
    with pytest.raises(ValueError, match="teacher reward mode"):
        validate_teacher_reward_config("unknown", 0.5)
    with pytest.raises(ValueError, match="frequency_bonus_scale"):
        validate_teacher_reward_config("appearance", -0.1)
