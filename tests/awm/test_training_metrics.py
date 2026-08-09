import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    _compute_awm_action_diversity_metrics,
    _sampled_entropy_response_mask,
    _sampled_token_entropy_metrics,
    compute_advantage,
)


def test_sampled_entropy_excludes_padding_and_nonfinite_tokens():
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((3, 3), dtype=torch.long),
            "attention_mask": torch.ones((3, 5), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
            "loss_mask": torch.tensor(
                [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]],
                dtype=torch.long,
            ),
        },
        non_tensors={"is_padding": np.asarray([False, False, True])},
    )
    log_probs = torch.tensor([[-1.0, -3.0, -9.0], [-2.0, float("nan"), -4.0], [-8.0, -8.0, -8.0]])

    mask = _sampled_entropy_response_mask(data, response_length=3)
    metrics = _sampled_token_entropy_metrics(log_probs, mask, "rollout/sampled_token_entropy_all")

    assert mask.tolist() == [[True, True, False], [True, False, True], [False] * 3]
    assert metrics["rollout/sampled_token_entropy_all"] == pytest.approx(2.5)
    assert metrics["rollout/sampled_token_entropy_all_token_count"] == 4.0
    assert metrics["rollout/sampled_token_entropy_all_nonfinite_rate"] == 0.0


def test_sampled_entropy_reports_nonfinite_selected_log_probs():
    metrics = _sampled_token_entropy_metrics(
        torch.tensor([[-1.0, float("nan")]]),
        torch.ones((1, 2), dtype=torch.bool),
        "actor/sampled_token_entropy_train",
    )

    assert metrics["actor/sampled_token_entropy_train"] == 1.0
    assert metrics["actor/sampled_token_entropy_train_token_count"] == 1.0
    assert metrics["actor/sampled_token_entropy_train_nonfinite_rate"] == 0.5


def test_train_sampled_entropy_uses_dapo_equal_reward_mask():
    rewards = np.asarray([2.0, 1.0, 0.0, -1.0] + [0.0] * 4, dtype=np.float32)
    token_rewards = torch.zeros((8, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray([f"task-{index}" for index in range(8)], dtype=object),
            "state_group_uid": np.asarray(["mixed"] * 4 + ["equal"] * 4),
            "rewards": rewards,
            "vpr_game": np.asarray(["awm"] * 8),
        },
    )

    data = compute_advantage(data, AdvantageEstimator.DAPO)
    mask = _sampled_entropy_response_mask(data, response_length=2)
    metrics = _sampled_token_entropy_metrics(
        torch.full((8, 2), -2.0),
        mask,
        "actor/sampled_token_entropy_train",
    )

    assert mask[:4].all()
    assert not mask[4:].any()
    assert metrics["actor/sampled_token_entropy_train"] == 2.0
    assert metrics["actor/sampled_token_entropy_train_token_count"] == 8.0


def test_awm_action_diversity_uses_canonical_actions_and_ignores_padding():
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros((9, 1), dtype=torch.long)},
        non_tensors={
            "state_group_uid": np.asarray(["first"] * 4 + ["second"] * 4 + ["first"], dtype=object),
            "parsed_action": np.asarray(
                ["A", "A", "B", "B", "C", "C", "C", "C", "different"],
                dtype=object,
            ),
            "awm_scenario": np.asarray(["scenario"] * 9, dtype=object),
            "is_padding": np.asarray([False] * 8 + [True]),
        },
    )

    metrics = _compute_awm_action_diversity_metrics(data)

    assert metrics["state_group/awm/canonical_unique_action_count_mean"] == pytest.approx(1.5)
    assert metrics["state_group/awm/canonical_unique_action_rate"] == pytest.approx(0.375)
    assert metrics["state_group/awm/canonical_all_identical_rate"] == pytest.approx(0.5)
