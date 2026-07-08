import numpy as np
import torch

from gigpo.core_gigpo import compute_turn_level_ppo_advantage


class _FakeData:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {}

    def __len__(self):
        return len(self.non_tensor_batch["rewards"])


def _make_data(rewards, values, traj_uids=None, turns=None, terminals=None, padding=None, response_len=3):
    n = len(rewards)
    response_mask = torch.ones(n, response_len, dtype=torch.float32)
    value_tensor = torch.zeros(n, response_len, dtype=torch.float32)
    for i, value in enumerate(values):
        value_tensor[i, :response_len] = float(value)
    if traj_uids is None:
        traj_uids = ["traj"] * n
    if turns is None:
        turns = list(range(n))
    if terminals is None:
        terminals = [False] * n
        terminals[-1] = True
    if padding is None:
        padding = [False] * n
    return _FakeData(
        batch={"response_mask": response_mask, "values": value_tensor},
        non_tensor_batch={
            "rewards": np.asarray(rewards, dtype=np.float32),
            "traj_uid": np.asarray(traj_uids, dtype=object),
            "turn_index": np.asarray(turns, dtype=np.int32),
            "is_terminal": np.asarray(terminals, dtype=bool),
            "is_padding": np.asarray(padding, dtype=bool),
        },
    )


def _row_scalars(token_tensor):
    return token_tensor[:, 0].detach().cpu().numpy()


def test_single_trajectory_raw_tdgae_gamma1_lambda1():
    data = _make_data(rewards=[0, 0, 1], values=[0.2, 0.3, 0.4])
    advantages, returns = compute_turn_level_ppo_advantage(
        data, gamma=1.0, lam=1.0, normalize_adv=False
    )

    np.testing.assert_allclose(_row_scalars(returns), [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(_row_scalars(advantages), [0.8, 0.7, 0.6], atol=1e-6)


def test_shuffled_rows_are_grouped_by_traj_and_turn_index():
    data = _make_data(
        rewards=[1, 0, -1, 0],
        values=[0.5, 0.2, -0.1, 0.3],
        traj_uids=["b", "a", "a", "b"],
        turns=[1, 0, 1, 0],
        terminals=[True, False, True, False],
    )
    advantages, returns = compute_turn_level_ppo_advantage(
        data, gamma=1.0, lam=1.0, normalize_adv=False
    )

    # traj a: row 1 -> row 2 has returns [-1, -1]
    # traj b: row 3 -> row 0 has returns [1, 1]
    np.testing.assert_allclose(_row_scalars(returns), [1.0, -1.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(_row_scalars(advantages), [0.5, -1.2, -0.9, 0.7], atol=1e-6)


def test_padding_rows_are_zeroed_and_excluded_from_normalization():
    data = _make_data(
        rewards=[0, 1, 100],
        values=[0.0, 0.0, 100.0],
        traj_uids=["a", "a", "pad"],
        turns=[0, 1, 0],
        terminals=[False, True, True],
        padding=[False, False, True],
    )
    advantages, returns = compute_turn_level_ppo_advantage(data, normalize_adv=True)

    assert data.batch["response_mask"][2].sum().item() == 0
    assert advantages[2].sum().item() == 0
    assert returns[2].sum().item() == 0
    assert data.meta_info["turn_level_ppo/padding_rate"] == 1 / 3


def test_terminal_failure_negative_returns():
    data = _make_data(rewards=[0, -1], values=[0.0, 0.0], terminals=[False, True])
    advantages, returns = compute_turn_level_ppo_advantage(
        data, gamma=1.0, lam=1.0, normalize_adv=False
    )

    np.testing.assert_allclose(_row_scalars(returns), [-1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(_row_scalars(advantages), [-1.0, -1.0], atol=1e-6)


def test_default_value_token_uses_pre_action_state_value():
    data = _make_data(rewards=[1], values=[0.0], terminals=[True], response_len=3)
    data.batch["values"][0] = torch.tensor([0.2, 0.5, 0.9])

    advantages, returns = compute_turn_level_ppo_advantage(data, normalize_adv=False)

    np.testing.assert_allclose(_row_scalars(returns), [1.0], atol=1e-6)
    # Default first-token value is 0.2, so raw advantage is 1.0 - 0.2.
    np.testing.assert_allclose(_row_scalars(advantages), [0.8], atol=1e-6)
