import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    _should_skip_dapo_state_group_update,
    compute_advantage,
)


def test_semantic_mask_rows_do_not_affect_group_statistics_or_gradients():
    raw_rewards = np.asarray([2.0, 0.0, 999.0, -1.0], dtype=np.float32)
    token_rewards = torch.zeros((4, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
            "state_group_uid": np.asarray(["state"] * 4, dtype=object),
            "rewards": raw_rewards,
            "semantic_train_mask": np.asarray([True, True, False, True]),
            "is_padding": np.asarray([False, False, False, True]),
            "vpr_game": np.asarray(["awm"] * 4, dtype=object),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    advantages = result.batch["advantages"][:, 0]
    assert advantages[0] > 0 > advantages[1]
    torch.testing.assert_close(advantages[2:], torch.zeros(2))
    assert result.non_tensor_batch["dapo_skip_loss"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert result.batch["response_mask"][2:].sum().item() == 0
    assert result.meta_info["dapo/semantic_supervision_sample_rate"] == 2 / 3


def test_group_with_fewer_than_two_supervised_candidates_is_fully_masked():
    raw_rewards = np.asarray([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    token_rewards = torch.zeros((4, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
            "state_group_uid": np.asarray(["state"] * 4, dtype=object),
            "rewards": raw_rewards,
            "semantic_train_mask": np.asarray([True, False, False, False]),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    assert result.non_tensor_batch["dapo_skip_loss"].all()
    assert result.batch["response_mask"].sum().item() == 0
    assert result.meta_info["dapo/missing_supervision_group_rate"] == 1.0
    assert _should_skip_dapo_state_group_update(result.meta_info) is True


def test_non_state_group_dapo_does_not_trigger_group_skip():
    assert _should_skip_dapo_state_group_update({}) is False
    assert _should_skip_dapo_state_group_update({"dapo/effective_state_groups": 1.0}) is False


def test_skipped_oracle_metrics_classify_equal_reward_awm_groups():
    raw_rewards = np.asarray(
        [2.0] * 4 + [0.0] * 4 + [2.0, 0.0, -1.0, 1.0],
        dtype=np.float32,
    )
    token_rewards = torch.zeros((12, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray([f"t{index}" for index in range(12)], dtype=object),
            "state_group_uid": np.asarray(
                ["oracle"] * 4 + ["unmatched"] * 4 + ["mixed"] * 4,
                dtype=object,
            ),
            "rewards": raw_rewards,
            "move_optimal": np.asarray(
                [True] * 4 + [False] * 4 + [True, False, False, True],
                dtype=bool,
            ),
            "vpr_game": np.asarray(["awm"] * 12, dtype=object),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    assert result.non_tensor_batch["dapo_skip_loss"].tolist() == ([True] * 8 + [False] * 4)
    assert result.meta_info["dapo/skipped_oracle_rate"] == 0.5
    assert result.meta_info["dapo/skipped_all_oracle_group_rate"] == 0.5
    assert result.meta_info["dapo/awm/skipped_oracle_rate"] == 0.5
    assert result.meta_info["dapo/awm/skipped_all_oracle_group_rate"] == 0.5
