"""Unit tests for the VPR turn-level advantage estimator."""

import numpy as np
import pytest


def _compute_vpr_per_turn_advantages(rewards, turn_indices, min_group_size=4, eps=1e-8):
    """Pure-numpy per-turn normalization — extracted for isolated testing."""
    per_step_rewards = np.array(rewards, dtype=np.float32)
    turn_indices = np.array(turn_indices, dtype=np.int32)
    n = len(per_step_rewards)
    row_advantages = np.zeros(n, dtype=np.float32)
    global_mean = per_step_rewards.mean()
    global_std = per_step_rewards.std() + eps
    for t in np.unique(turn_indices):
        mask = turn_indices == t
        group = per_step_rewards[mask]
        if len(group) >= min_group_size:
            mean_t = group.mean()
            std_t = group.std() + eps
        else:
            mean_t = global_mean
            std_t = global_std
        row_advantages[mask] = (group - mean_t) / std_t
    return row_advantages


def make_mock_data(rewards, turn_indices, response_len=4):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    batch_size = len(rewards)
    response_mask = torch.zeros(batch_size, response_len)
    for i in range(batch_size):
        response_mask[i, :response_len] = 1.0
    batch = {"response_mask": response_mask}
    non_tensor_batch = {
        "rewards": np.array(rewards, dtype=np.float32),
        "turn_index": np.array(turn_indices, dtype=np.int32),
    }
    return SimpleNamespace(batch=batch, non_tensor_batch=non_tensor_batch)


def compute_advantage_fn(data, min_group_size=4, eps=1e-8, outcome_reward_scale=0.0):
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location(
        "core_gigpo_test",
        "gigpo/core_gigpo.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["core_gigpo_test"] = mod
    spec.loader.exec_module(mod)
    return mod.compute_vpr_turn_level_advantage(data, min_group_size, eps, outcome_reward_scale)


# ── Pure numpy tests (no torch required) ────────────────────────────────────

class TestPerTurnNormalizationNumpy:
    def test_per_turn_groups_normalized_independently(self):
        rewards = [1.0, 0.0, 0.5, -0.5]
        turns = [0, 0, 2, 2]
        adv = _compute_vpr_per_turn_advantages(rewards, turns, min_group_size=2)
        # Turn-0: mean=0.5, std=0.5 → adv[0]≈+1.0, adv[1]≈-1.0
        # Turn-2: mean=0.0, std=0.5 → adv[2]≈+1.0, adv[3]≈-1.0
        assert adv[0] > 0 and adv[1] < 0
        assert adv[2] > 0 and adv[3] < 0
        assert abs(adv[0] - adv[2]) < 1e-5, "Same normalized value across turns"

    def test_fallback_when_below_min_group_size(self):
        # 2 episodes at turn 5 (< min_group_size=4)
        rewards = [1.0, -1.0]
        turns = [4, 4]
        adv = _compute_vpr_per_turn_advantages(rewards, turns, min_group_size=4)
        # Fallback to global: mean=0.0, std=1.0 → adv≈[+1.0, -1.0]
        assert adv[0] > 0 and adv[1] < 0
        assert abs(adv[0] + adv[1]) < 1e-5, "Symmetric around 0"

    def test_regression_guard_per_turn_vs_global(self):
        rewards = [10.0, 0.0, 1.0, 0.0]
        turns = [0, 0, 1, 1]
        adv_per_turn = _compute_vpr_per_turn_advantages(rewards, turns, min_group_size=2)
        r = np.array(rewards, dtype=np.float32)
        global_adv = (r - r.mean()) / (r.std() + 1e-8)
        assert not np.allclose(adv_per_turn, global_adv, atol=0.01), \
            "Per-turn should differ from global normalization"

    def test_outcome_reward_on_terminal(self):
        rewards = [0.5, -0.5]
        turns = [0, 0]
        # Simulate outcome reward: terminal_success=[True, False], scale=1.0
        is_terminal = np.array([True, True], dtype=bool)
        terminal_success = np.array([True, False], dtype=bool)
        outcome_scale = 1.0
        modified = np.array(rewards, dtype=np.float32)
        modified += is_terminal.astype(np.float32) * (outcome_scale * terminal_success.astype(np.float32))
        # modified = [0.5+1.0, -0.5+0.0] = [1.5, -0.5]
        adv = _compute_vpr_per_turn_advantages(modified, turns, min_group_size=2)
        assert adv[0] > adv[1], "Success episode should have higher advantage"


# ── Torch-dependent tests ────────────────────────────────────────────────────

def test_per_turn_normalization_distinct():
    rewards = [1.0, 0.0, 0.5, -0.5]
    turn_indices = [0, 0, 2, 2]
    data = make_mock_data(rewards, turn_indices)
    advantages, returns = compute_advantage_fn(data, min_group_size=2)
    last_token = advantages.numpy()[:, -1]
    assert last_token[0] > 0, "Turn-0 optimal should have positive advantage"
    assert last_token[1] < 0, "Turn-0 suboptimal should have negative advantage"
    assert last_token[2] > 0, "Turn-2 optimal should have positive advantage"
    assert last_token[3] < 0, "Turn-2 suboptimal should have negative advantage"


def test_fallback_when_few_same_turn():
    rewards = [1.0, -1.0]
    turn_indices = [4, 4]
    data = make_mock_data(rewards, turn_indices)
    advantages, _ = compute_advantage_fn(data, min_group_size=4)
    last_token = advantages.numpy()[:, -1]
    assert last_token[0] > 0
    assert last_token[1] < 0


def test_advantage_regression_guard():
    rewards = [10.0, 0.0, 1.0, 0.0]
    turn_indices = [0, 0, 1, 1]
    data = make_mock_data(rewards, turn_indices)
    adv_per_turn, _ = compute_advantage_fn(data, min_group_size=2)
    r = np.array(rewards, dtype=np.float32)
    global_adv = (r - r.mean()) / (r.std() + 1e-8)
    per_turn_vals = adv_per_turn.numpy()[:, -1]
    assert not np.allclose(per_turn_vals, global_adv, atol=0.01), \
        "Per-turn and global normalization should produce different results"
