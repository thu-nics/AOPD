import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    _compute_dapo_effective_row_mask,
    _pad_compacted_policy_batch,
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
            "move_optimal": np.asarray([True, False, True, True]),
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
    assert result.meta_info["dapo/oracle_hit_rate"] == 0.5
    assert result.meta_info["dapo/awm/oracle_hit_rate"] == 0.5


def test_runtime_mask_is_an_independent_dapo_gradient_gate():
    raw_rewards = np.asarray([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    token_rewards = torch.zeros((4, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray([f"t{index}" for index in range(4)], dtype=object),
            "state_group_uid": np.asarray(["state"] * 4, dtype=object),
            "rewards": raw_rewards,
            "semantic_train_mask": np.ones(4, dtype=bool),
            "runtime_train_mask": np.asarray([False, False, False, False]),
            "move_optimal": np.ones(4, dtype=bool),
            "vpr_game": np.asarray(["awm"] * 4, dtype=object),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    assert result.non_tensor_batch["dapo_skip_loss"].all()
    assert result.batch["response_mask"].sum().item() == 0
    assert result.meta_info["dapo/missing_supervision_group_rate"] == 1.0
    assert result.meta_info["dapo/oracle_hit_rate"] == 0.0
    assert result.meta_info["dapo/awm/oracle_hit_rate"] == 0.0


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
    assert result.meta_info["dapo/oracle_hit_rate"] == 0.5
    assert result.meta_info["dapo/awm/skipped_oracle_rate"] == 0.5
    assert result.meta_info["dapo/awm/skipped_all_oracle_group_rate"] == 0.5
    assert result.meta_info["dapo/awm/oracle_hit_rate"] == 0.5


def test_effective_row_compaction_matches_dapo_skip_semantics():
    raw_rewards = np.asarray(
        [1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 2.0],
        dtype=np.float32,
    )
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones((8, 2), dtype=torch.long),
        },
        non_tensors={
            "uid": np.asarray([f"t{index}" for index in range(8)], dtype=object),
            "state_group_uid": np.asarray(
                ["equal"] * 4 + ["effective"] * 4,
                dtype=object,
            ),
            "rewards": raw_rewards,
            "semantic_train_mask": np.asarray(
                [True] * 7 + [False],
                dtype=bool,
            ),
            "runtime_train_mask": np.ones(8, dtype=bool),
            "is_padding": np.zeros(8, dtype=bool),
        },
    )

    effective = _compute_dapo_effective_row_mask(data)

    assert effective.tolist() == [False] * 4 + [True, True, True, False]


def test_compacted_policy_padding_is_explicitly_gradient_free():
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones((3, 2), dtype=torch.long),
            "responses": torch.ones((3, 2), dtype=torch.long),
        },
        non_tensors={
            "is_padding": np.zeros(3, dtype=bool),
            "dapo_skip_loss": np.zeros(3, dtype=bool),
            "uid": np.asarray(["a", "b", "c"], dtype=object),
        },
    )

    compact, pad_size = _pad_compacted_policy_batch(
        data,
        np.asarray([0, 2], dtype=np.int64),
        divisor=4,
    )

    assert pad_size == 2
    assert len(compact) == 4
    assert compact.non_tensor_batch["is_padding"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert compact.non_tensor_batch["dapo_skip_loss"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert compact.batch["response_mask"][-2:].sum().item() == 0


def test_precomputed_policy_rows_must_match_advantage_masks():
    raw_rewards = np.asarray([1.0, 0.0, -1.0, 2.0], dtype=np.float32)
    token_rewards = torch.zeros((4, 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray([f"t{index}" for index in range(4)], dtype=object),
            "state_group_uid": np.asarray(["state"] * 4, dtype=object),
            "rewards": raw_rewards,
            "dapo_skip_loss": np.ones(4, dtype=bool),
        },
    )

    with pytest.raises(RuntimeError, match="disagree with advantage masks"):
        compute_advantage(data, AdvantageEstimator.DAPO)
