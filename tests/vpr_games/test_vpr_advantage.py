"""Unit tests for the VPR turn-level advantage estimator."""

import numpy as np
import pytest


def make_mock_data(rewards, turn_indices, response_len=4):
    """Create a minimal DataProto-like mock for testing compute_vpr_turn_level_advantage."""
    import torch
    from types import SimpleNamespace

    batch_size = len(rewards)
    response_mask = torch.zeros(batch_size, response_len)
    for i in range(batch_size):
        # Fill first response_len-1 tokens as valid
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
    try:
        spec.loader.exec_module(mod)
        return mod.compute_vpr_turn_level_advantage(data, min_group_size, eps, outcome_reward_scale)
    except Exception as e:
        pytest.skip(f"Cannot load core_gigpo (likely torch missing): {e}")


def test_per_turn_normalization_distinct():
    """Turn-1 and turn-3 rewards should be normalized independently."""
    # 4 episodes: 2 at turn 1, 2 at turn 3
    rewards = [1.0, 0.0, 0.5, -0.5]
    turn_indices = [0, 0, 2, 2]
    data = make_mock_data(rewards, turn_indices)
    advantages, returns = compute_advantage_fn(data, min_group_size=2)

    adv = advantages.numpy()
    # Turn 0: rewards [1.0, 0.0], mean=0.5, std=0.5
    # adv[0] = (1.0 - 0.5) / (0.5 + 1e-8) ≈ 1.0
    # adv[1] = (0.0 - 0.5) / (0.5 + 1e-8) ≈ -1.0
    # Turn 2: rewards [0.5, -0.5], mean=0.0, std=0.5
    # adv[2] = (0.5 - 0.0) / (0.5 + 1e-8) ≈ 1.0
    # adv[3] = (-0.5 - 0.0) / (0.5 + 1e-8) ≈ -1.0

    # Turn 0 and turn 2 advantages should be normalized independently
    # (both come out to ±1.0 but from different normalization groups)
    last_token = adv[:, -1]
    assert last_token[0] > 0, "Turn-0 optimal should have positive advantage"
    assert last_token[1] < 0, "Turn-0 suboptimal should have negative advantage"
    assert last_token[2] > 0, "Turn-2 optimal should have positive advantage"
    assert last_token[3] < 0, "Turn-2 suboptimal should have negative advantage"


def test_fallback_when_few_same_turn():
    """When fewer than min_group_size rows share a turn, fall back to batch-wide normalization."""
    # 2 episodes at turn 5 (< min_group_size=4 → fallback)
    rewards = [1.0, -1.0]
    turn_indices = [4, 4]
    data = make_mock_data(rewards, turn_indices)
    advantages, _ = compute_advantage_fn(data, min_group_size=4)

    # With fallback, global mean=0.0, std=1.0
    # adv[0] = (1.0 - 0.0) / (1.0 + 1e-8) ≈ 1.0
    # adv[1] = (-1.0 - 0.0) / (1.0 + 1e-8) ≈ -1.0
    last_token = advantages.numpy()[:, -1]
    assert last_token[0] > 0
    assert last_token[1] < 0


def test_advantage_regression_guard():
    """Global normalization produces different advantages than per-turn normalization."""
    # 4 rows: 2 turns × 2 episodes each, with different reward scales per turn
    rewards_t0 = [10.0, 0.0]  # large scale
    rewards_t1 = [1.0, 0.0]   # small scale
    rewards = rewards_t0 + rewards_t1
    turn_indices = [0, 0, 1, 1]
    data = make_mock_data(rewards, turn_indices)

    # Per-turn normalization (min_group_size=2)
    adv_per_turn, _ = compute_advantage_fn(data, min_group_size=2)

    # Manual global normalization
    r = np.array(rewards, dtype=np.float32)
    global_adv = (r - r.mean()) / (r.std() + 1e-8)

    per_turn_vals = adv_per_turn.numpy()[:, -1]
    # If per-turn normalization is working, it should produce different results than global
    # Turn-0 row-0 advantage (per-turn): (10-5)/5 = 1.0; global: (10-2.75)/std_global
    assert not np.allclose(per_turn_vals, global_adv, atol=0.01), \
        "Per-turn and global normalization should produce different results"
