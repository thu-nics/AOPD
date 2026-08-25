import random

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agent_system.environments.rollout_progress import (
    NoProgressTracker,
    select_history_aware_with_appearance_counterfactual,
    validate_no_progress_config,
)
from agent_system.multi_turn_rollout.rollout_loop import _replace_dataproto_rows
from verl import DataProto


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


def test_no_progress_tracker_requires_two_identical_tool_observations():
    tracker = NoProgressTracker()
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")

    inspection = tracker.inspect_candidate_actions(
        action_kinds=["tool"] * 4,
        canonical_actions=["lookup:{}"] * 4,
        enabled=True,
        min_repeat_streak=2,
    )

    assert inspection == {
        "trigger": True,
        "repeat_streak": 2,
        "candidate_unique_action_count": 1,
        "repeated_canonical_action": "lookup:{}",
    }


@pytest.mark.parametrize(
    ("kinds", "actions"),
    [
        (["tool"] * 4, ["lookup:{}", "update:{}", "lookup:{}", "lookup:{}"]),
        (["invalid"] * 4, ["lookup:{}"] * 4),
        (["tool"] * 4, ["different:{}"] * 4),
    ],
)
def test_no_progress_tracker_does_not_trigger_on_noncollapsed_group(kinds, actions):
    tracker = NoProgressTracker()
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")

    assert not tracker.inspect_candidate_actions(
        action_kinds=kinds,
        canonical_actions=actions,
        enabled=True,
        min_repeat_streak=2,
    )["trigger"]


def test_non_tool_action_resets_no_progress_streak():
    tracker = NoProgressTracker()
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="tool", canonical_action="lookup:{}", observation="same")
    tracker.record(action_kind="message", canonical_action="message:done", observation="reply")

    assert tracker.repeat_streak == 0
    assert tracker.last_canonical_action is None


def test_no_progress_config_is_exactly_zero_or_one_round():
    assert validate_no_progress_config(enabled=False, max_rounds=0, min_repeat_streak=2) == (False, 0, 2)
    assert validate_no_progress_config(enabled=True, max_rounds=1, min_repeat_streak=2) == (True, 1, 2)
    with pytest.raises(ValueError, match="enabled=true"):
        validate_no_progress_config(enabled=True, max_rounds=0, min_repeat_streak=2)
    with pytest.raises(ValueError, match="0 or 1"):
        validate_no_progress_config(enabled=True, max_rounds=2, min_repeat_streak=2)


def test_selective_regeneration_replaces_only_requested_rows():
    destination = DataProto(
        batch=TensorDict(
            {
                "responses": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]]),
                "rollout_log_probs": torch.zeros(4, 2),
            },
            batch_size=[4],
        ),
        non_tensor_batch={"raw": np.asarray(["a", "b", "c", "d"], dtype=object)},
    )
    source = DataProto(
        batch=TensorDict(
            {
                "responses": torch.tensor([[8, 8], [9, 9]]),
                "rollout_log_probs": torch.ones(2, 2),
            },
            batch_size=[2],
        ),
        non_tensor_batch={"raw": np.asarray(["x", "y"], dtype=object)},
    )

    _replace_dataproto_rows(destination, source, np.asarray([1, 3]))

    assert destination.batch["responses"].tolist() == [
        [1, 1],
        [8, 8],
        [3, 3],
        [9, 9],
    ]
    assert destination.batch["rollout_log_probs"].tolist() == [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]
    assert destination.non_tensor_batch["raw"].tolist() == ["a", "x", "c", "y"]
