import random

import pytest

from agent_system.environments.rollout_progress import (
    NoProgressTracker,
    select_history_aware_with_appearance_counterfactual,
    validate_progress_config,
)


def test_nonrepeat_selection_prefers_only_tied_maximum_alternative():
    selection = select_history_aware_with_appearance_counterfactual(
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        ["lookup:{}", "update:{}", "finish:{}"],
        "lookup:{}",
        random.Random(4),
    )

    assert selection.selected_index == 1
    assert selection.appearance_index == 1
    assert selection.nonrepeat_alternative_available is True
    assert selection.nonrepeat_preference_applied is True
    assert selection.selection_type == "uniform_argmax_nonrepeat_preferred"


def test_nonrepeat_selection_never_drops_to_lower_reward():
    selection = select_history_aware_with_appearance_counterfactual(
        [2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        ["lookup:{}", "update:{}", "finish:{}"],
        "lookup:{}",
        random.Random(0),
    )

    assert selection.selected_index == 0
    assert selection.nonrepeat_preference_applied is False


def test_nonrepeat_selection_preserves_rng_without_applicable_preference():
    rng = random.Random(13)
    reference = random.Random(13)
    selection = select_history_aware_with_appearance_counterfactual(
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        ["a", "b", "c"],
        None,
        rng,
    )

    assert selection.selected_index == reference.choice([0, 1])
    assert selection.appearance_index == selection.selected_index
    assert rng.getstate() == reference.getstate()


def test_no_progress_tracker_marks_third_identical_candidate():
    tracker = NoProgressTracker()
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")

    assert tracker.prospective_repeat_flags(
        action_kinds=["tool", "tool", "message", "invalid"],
        canonical_actions=["lookup:{}", "update:{}", "message:done", "invalid"],
        min_streak=3,
    ) == [True, False, False, False]
    assert tracker.reached(4) is False


def test_no_progress_tracker_reaches_four_and_resets_on_progress():
    tracker = NoProgressTracker()
    for _ in range(4):
        tracker.record(
            action_kind="tool",
            canonical_action="lookup:{}",
            observation="same",
        )
    assert tracker.reached(4) is True

    tracker.record(
        action_kind="tool",
        canonical_action="lookup:{}",
        observation="changed",
    )
    assert tracker.repeat_streak == 1


def test_non_tool_action_resets_no_progress_streak():
    tracker = NoProgressTracker()
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="message", canonical_action="message:done", observation="reply")

    assert tracker.repeat_streak == 0
    assert tracker.last_canonical_action is None


def test_progress_config_validates_cap_before_termination():
    assert validate_progress_config(
        repeat_reward_cap_enabled=True,
        repeat_reward_cap_min_streak=3,
        repeat_reward_cap_value=0,
        repeat_termination_enabled=True,
        repeat_termination_max_streak=4,
    ) == (True, 3, 0.0, True, 4)
    with pytest.raises(ValueError, match="must not precede"):
        validate_progress_config(
            repeat_reward_cap_enabled=True,
            repeat_reward_cap_min_streak=4,
            repeat_reward_cap_value=0,
            repeat_termination_enabled=True,
            repeat_termination_max_streak=3,
        )
