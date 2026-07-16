from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_system.reward_manager.dapo_turn import DAPOTurnRewardManager
from verl import DataProto


def test_dapo_turn_reward_applies_linear_overlong_penalty():
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((2, 2), dtype=torch.long),
            "responses": torch.ones((2, 4), dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]],
                dtype=torch.long,
            ),
        },
        non_tensors={"rewards": np.asarray([1.0, 0.0], dtype=np.float32)},
    )
    manager = DAPOTurnRewardManager(
        tokenizer=None,
        num_examine=0,
        max_resp_len=4,
        overlong_buffer_cfg=SimpleNamespace(
            enable=True,
            len=2,
            penalty_factor=1.0,
            log=True,
        ),
    )

    result = manager(data, return_dict=True)

    reward_tensor = result["reward_tensor"]
    assert reward_tensor[0, 2].item() == pytest.approx(0.5)
    assert reward_tensor[1, 3].item() == pytest.approx(-1.0)
    assert result["reward_extra_info"]["overlong_reward"] == pytest.approx([-0.5, -1.0])
