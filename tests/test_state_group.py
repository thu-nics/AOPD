import random

import numpy as np
import pytest

from agent_system.environments.teacher_reward import (
    select_with_appearance_counterfactual,
    teacher_match_reward,
    teacher_selection_diagnostics,
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


def test_appearance_counterfactual_does_not_perturb_real_rng_stream():
    rng = random.Random(0)
    reference_rng = random.Random(0)

    selected, appearance_selected = select_with_appearance_counterfactual(
        [1.5, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        rng,
    )
    reference_selected = reference_rng.choice([0])

    assert selected == reference_selected == 0
    assert appearance_selected in {0, 1}
    assert rng.getstate() == reference_rng.getstate()


def test_identical_score_sets_produce_identical_real_and_counterfactual_selection():
    selected, appearance_selected = select_with_appearance_counterfactual(
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        random.Random(4),
    )
    assert selected == appearance_selected


def test_teacher_selection_diagnostics_separates_action_kinds_and_counterfactual():
    metrics = teacher_selection_diagnostics(
        [
            [
                {"action_kind": "tool", "teacher_frequency": 3},
                {"action_kind": "tool", "teacher_frequency": 0},
                {"action_kind": "message", "teacher_frequency": 1},
                {"action_kind": "invalid", "teacher_frequency": 0},
            ]
        ],
        [
            [
                {
                    "action_kind": "tool",
                    "appearance_counterfactual_action_kind": "message",
                    "frequency_changed_selection": True,
                    "frequency_changed_selection_to_tool": True,
                    "frequency_changed_selection_to_message": False,
                }
            ]
        ],
    )

    expected = {
        "tool_candidate_teacher_match_count_mean": 1.5,
        "message_candidate_teacher_match_count_mean": 1.0,
        "selected_tool_action_rate": 1.0,
        "selected_message_action_rate": 0.0,
        "appearance_counterfactual_selected_tool_rate": 0.0,
        "appearance_counterfactual_selected_message_rate": 1.0,
        "frequency_changed_selection_rate": 1.0,
        "frequency_changed_selection_to_tool_rate": 1.0,
        "frequency_changed_selection_to_message_rate": 0.0,
    }
    for name, value in expected.items():
        assert metrics[name] == value
    assert metrics["diagnostic_state_group_count"] == 0.0


def test_stopping_diagnostics_explain_missing_and_unselected_messages():
    teacher_message = [{"kind": "message", "content": "done"}]
    teacher_tool = [{"kind": "tool", "name": "lookup", "arguments": {}}]

    def row(
        group,
        kind,
        *,
        teacher,
        frequency=0,
        score=0.0,
        selected=False,
        action="",
        turn=0,
    ):
        return {
            "state_group_uid": group,
            "action_kind": kind,
            "teacher_multiset": teacher,
            "teacher_frequency": frequency,
            "selection_score": score,
            "state_group_selected": selected,
            "parsed_action": action,
            "turn_index": turn,
        }

    candidates = [
        [
            row("g1", "message", teacher=teacher_message, frequency=1, score=1, selected=True),
            row("g1", "tool", teacher=teacher_message, action="tool-a"),
            row("g1", "tool", teacher=teacher_message, action="tool-b"),
            row("g1", "invalid", teacher=teacher_message),
            row("g2", "tool", teacher=teacher_message, action="tool-a", selected=True, turn=5),
            row("g2", "tool", teacher=teacher_message, action="tool-b", turn=5),
            row("g2", "tool", teacher=teacher_message, action="tool-c", turn=5),
            row("g2", "invalid", teacher=teacher_message, turn=5),
            row("g3", "message", teacher=teacher_message, frequency=1, score=1, turn=10),
            row("g3", "tool", teacher=teacher_message, score=1.5, selected=True, action="tool-a", turn=10),
            row("g3", "tool", teacher=teacher_message, action="tool-b", turn=10),
            row("g3", "invalid", teacher=teacher_message, turn=10),
            row("g4", "message", teacher=teacher_message, frequency=1, score=1, turn=20),
            row("g4", "tool", teacher=teacher_message, score=1, selected=True, action="duplicate", turn=20),
            row("g4", "tool", teacher=teacher_message, score=1, action="duplicate", turn=20),
            row("g4", "invalid", teacher=teacher_message, score=-1, turn=20),
            row("g5", "message", teacher=teacher_tool),
            row("g5", "tool", teacher=teacher_tool, frequency=1, score=1, selected=True, action="tool-a"),
            row("g5", "tool", teacher=teacher_tool, action="tool-b"),
            row("g5", "invalid", teacher=teacher_tool),
        ]
    ]

    metrics = teacher_selection_diagnostics(candidates, [[]])

    assert metrics["diagnostic_state_group_count"] == 5.0
    assert metrics["candidate_action_count"] == 20.0
    assert metrics["candidate_message_rate"] == pytest.approx(0.2)
    assert metrics["teacher_has_message_group_rate"] == pytest.approx(0.8)
    assert metrics["teacher_has_message_student_missing_rate"] == pytest.approx(0.25)
    teacher_message_partition = (
        metrics["teacher_has_message_student_missing_rate"],
        metrics["teacher_message_unmatched_candidate_rate"],
        metrics["teacher_message_matched_selected_rate"],
        metrics["teacher_message_matched_not_selected_rate"],
    )
    assert teacher_message_partition == pytest.approx((0.25, 0.0, 0.25, 0.5))
    assert sum(teacher_message_partition) == pytest.approx(1.0)
    assert metrics["teacher_all_tool_student_message_rate"] == 1.0
    assert metrics["message_candidate_match_rate_when_teacher_message"] == 1.0
    assert metrics["matched_message_not_selected_rate"] == pytest.approx(2 / 3)
    assert metrics["matched_message_lower_score_rate"] == pytest.approx(0.5)
    assert metrics["matched_message_tied_tool_selected_rate"] == pytest.approx(0.5)
    assert metrics["matched_message_tied_duplicate_tool_bias_rate"] == 1.0
    assert metrics["unmatched_message_tool_selected_rate"] == 1.0
    assert metrics["turn_1_5_state_group_count"] == 2.0
    assert metrics["turn_1_5_selected_message_rate"] == pytest.approx(0.5)
    assert metrics["turn_6_10_teacher_message_rate"] == 1.0
    assert metrics["turn_11_20_student_message_candidate_rate"] == 1.0
    assert metrics["turn_21_plus_selected_message_rate"] == 0.0


def test_repeated_tool_diagnostics_require_identical_post_action_observation():
    selected = [
        [
            {"action_kind": "tool", "parsed_action": "lookup:{}", "observation": "same"},
            {
                "action_kind": "tool",
                "parsed_action": "lookup:{}",
                "observation": "same",
                "teacher_frequency": 1,
            },
            {
                "action_kind": "tool",
                "parsed_action": "lookup:{}",
                "observation": "same",
                "teacher_frequency": 0,
            },
        ],
        [
            {"action_kind": "tool", "parsed_action": "update:{}", "observation": "before"},
            {"action_kind": "tool", "parsed_action": "update:{}", "observation": "after"},
        ],
    ]

    metrics = teacher_selection_diagnostics([], selected)

    assert metrics["repeated_tool_transition_count"] == 3.0
    assert metrics["consecutive_same_tool_call_rate"] == 1.0
    assert metrics["consecutive_same_tool_same_observation_rate"] == pytest.approx(2 / 3)
    assert metrics["repeat_streak_ge3_trajectory_rate"] == pytest.approx(0.5)
    assert metrics["repeat_streak_max_mean"] == pytest.approx(2.0)
    assert metrics["repeated_tool_teacher_match_rate"] == pytest.approx(0.5)
