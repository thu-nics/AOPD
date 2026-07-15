import numpy as np
import torch

import pytest

from gigpo.core_gigpo import compute_vineppo_advantage


class _FakeData:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {}

    def __len__(self):
        return self.batch["response_mask"].shape[0]


def _row_values(t):
    return t[:, 0].detach().cpu().numpy()


def test_vineppo_advantage_formula_and_terminal_next_zero():
    data = _FakeData(
        batch={"response_mask": torch.ones(3, 2)},
        non_tensor_batch={
            "rewards": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "vine_v_curr": np.asarray([0.5, 1.0, 4.0], dtype=np.float32),
            "vine_v_next": np.asarray([2.0, 100.0, 0.0], dtype=np.float32),
            "is_terminal": np.asarray([False, True, False], dtype=bool),
            "is_padding": np.asarray([False, False, True], dtype=bool),
        },
    )

    adv, ret = compute_vineppo_advantage(data, gamma=0.5, normalize_adv=False)

    np.testing.assert_allclose(_row_values(adv), [1.5, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_row_values(ret), [1.5, 1.0, 0.0], atol=1e-6)
    assert data.meta_info["vineppo/num_rows"] == 2.0


def test_vineppo_normalization_excludes_padding_and_uses_response_mask():
    data = _FakeData(
        batch={"response_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])},
        non_tensor_batch={
            "rewards": np.asarray([0.0, 2.0, 100.0], dtype=np.float32),
            "vine_v_curr": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "vine_v_next": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "is_terminal": np.asarray([False, False, False], dtype=bool),
            "is_padding": np.asarray([False, False, True], dtype=bool),
        },
    )

    adv, _ = compute_vineppo_advantage(data, normalize_adv=True)

    # The valid token advantages are [-1/sqrt(3), -1/sqrt(3), 2/sqrt(3)].
    # Whitening is token-weighted, matching the official VinePPO PPO path.
    np.testing.assert_allclose(
        _row_values(adv)[:2],
        [-1.0 / np.sqrt(3.0), 2.0 / np.sqrt(3.0)],
        atol=1e-6,
    )
    valid_adv = adv[data.batch["response_mask"].bool()]
    assert valid_adv.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert valid_adv.std(unbiased=True).item() == pytest.approx(1.0, abs=1e-6)
    assert adv[1, 1].item() == 0.0
    assert adv[2].sum().item() == 0.0
    assert data.meta_info["vineppo/token_raw_adv_mean"] == pytest.approx(2.0 / 3.0)
    assert data.meta_info["vineppo/all_zero_advantage"] == 0.0


def test_vineppo_all_zero_advantage_is_reported_for_update_skip():
    data = _FakeData(
        batch={"response_mask": torch.ones(2, 2)},
        non_tensor_batch={
            "rewards": np.asarray([-1.0, 0.0], dtype=np.float32),
            "vine_v_curr": np.asarray([-1.0, -1.0], dtype=np.float32),
            "vine_v_next": np.asarray([0.0, -1.0], dtype=np.float32),
            "is_terminal": np.asarray([True, False], dtype=bool),
        },
    )

    adv, _ = compute_vineppo_advantage(data, normalize_adv=True)

    assert adv.abs().sum().item() == 0.0
    assert data.meta_info["vineppo/raw_adv_abs_max"] == 0.0
    assert data.meta_info["vineppo/zero_adv_row_rate"] == 1.0
    assert data.meta_info["vineppo/all_zero_advantage"] == 1.0
