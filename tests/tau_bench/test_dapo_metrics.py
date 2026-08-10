import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage


def test_skipped_oracle_metrics_are_reported_per_tau_domain():
    raw_rewards = np.asarray(
        [1.0] * 4 + [0.0] * 4 + [1.0, 0.0, -1.0, 1.0],
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
                ["airline-oracle"] * 4 + ["retail-unmatched"] * 4 + ["retail-mixed"] * 4,
                dtype=object,
            ),
            "rewards": raw_rewards,
            "move_optimal": np.asarray(
                [True] * 4 + [False] * 4 + [True, False, False, True],
                dtype=bool,
            ),
            "vpr_game": np.asarray(
                ["tau_airline"] * 4 + ["tau_retail"] * 8,
                dtype=object,
            ),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    assert result.meta_info["dapo/oracle_hit_rate"] == 0.5
    assert result.meta_info["dapo/skipped_oracle_rate"] == 0.5
    assert result.meta_info["dapo/skipped_all_oracle_group_rate"] == 0.5
    assert result.meta_info["dapo/tau_airline/oracle_hit_rate"] == 1.0
    assert result.meta_info["dapo/tau_airline/skipped_oracle_rate"] == 1.0
    assert result.meta_info["dapo/tau_airline/skipped_all_oracle_group_rate"] == 1.0
    assert result.meta_info["dapo/tau_retail/oracle_hit_rate"] == 0.25
    assert result.meta_info["dapo/tau_retail/skipped_oracle_rate"] == 0.0
    assert result.meta_info["dapo/tau_retail/skipped_all_oracle_group_rate"] == 0.0
