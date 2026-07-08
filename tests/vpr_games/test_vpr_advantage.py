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
    return SimpleNamespace(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


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


# ── Divisibility-padding exclusion ──────────────────────────────────────────

def _make_data_with_padding(rewards, turn_indices, is_padding, response_len=4):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    bs = len(rewards)
    response_mask = torch.ones(bs, response_len)
    batch = {"response_mask": response_mask}
    non_tensor_batch = {
        "rewards": np.array(rewards, dtype=np.float32),
        "turn_index": np.array(turn_indices, dtype=np.int32),
        "is_padding": np.array(is_padding, dtype=bool),
    }
    return SimpleNamespace(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


class TestPaddingExclusion:
    def test_padding_does_not_change_real_advantages_or_contribute_loss(self):
        """Divisibility padding (duplicate rows with extreme rewards) must not change any
        real row's advantage, and padded rows must carry zero advantage and zeroed
        response mask (so they contribute no loss)."""
        torch = pytest.importorskip("torch")
        real_rewards = [1.0, -1.0, 0.0, 1.0, -1.0]   # odd real-row count
        real_turns = [0, 0, 1, 1, 0]
        expected = _compute_vpr_per_turn_advantages(real_rewards, real_turns)

        # Two padding rows duplicate real positions but with extreme rewards that would
        # badly skew global/per-turn statistics if (incorrectly) counted.
        rewards = real_rewards + [999.0, -999.0]
        turns = real_turns + [0, 1]
        is_padding = [False] * 5 + [True] * 2
        data = _make_data_with_padding(rewards, turns, is_padding)

        token_adv, returns = compute_advantage_fn(data, min_group_size=4, outcome_reward_scale=0.0)
        row_adv = token_adv[:, 0].cpu().numpy()

        # Real rows match the padding-free reference exactly.
        np.testing.assert_allclose(row_adv[:5], expected, atol=1e-5)
        # Padded rows carry zero advantage and zeroed response mask → no loss/gradient.
        assert np.allclose(row_adv[5:], 0.0)
        rm = data.batch["response_mask"]
        assert rm[5:].sum().item() == 0, "Padded rows must have a zeroed response mask"
        assert rm[:5].sum().item() > 0, "Real rows must retain their response mask"
        assert torch.allclose(returns, token_adv)

    def test_no_padding_field_behaves_as_all_real(self):
        """Absent is_padding (e.g. a no-padding batch) must behave as all-real."""
        rewards = [1.0, -1.0, 0.0, 1.0]
        turns = [0, 0, 1, 1]
        data = make_mock_data(rewards, turns)  # no is_padding key
        token_adv, _ = compute_advantage_fn(data, min_group_size=4, outcome_reward_scale=0.0)
        expected = _compute_vpr_per_turn_advantages(rewards, turns)
        np.testing.assert_allclose(token_adv[:, 0].cpu().numpy(), expected, atol=1e-5)


def test_state_group_advantage_normalizes_within_state_group():
    data = make_mock_data([2.0, -1.0, 0.0, 0.0], [0, 0, 0, 0])
    data.non_tensor_batch["state_group_uid"] = np.array(["a", "a", "b", "b"], dtype=object)
    data.non_tensor_batch["state_group_selected"] = np.array([True, False, True, False])
    data.non_tensor_batch["state_group_unique_action_rate"] = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    data.non_tensor_batch["move_optimal"] = np.array([True, False, False, False])
    data.non_tensor_batch["legal_non_oracle"] = np.array([False, True, True, True])
    data.non_tensor_batch["is_action_valid"] = np.array([True, True, True, False])
    data.non_tensor_batch["parsed_action"] = np.array(["reveal 1 1", "flag 1 2", "reveal 1 3", "flag 1 4"], dtype=object)
    data.non_tensor_batch["oracle_tier"] = np.array(["safe_reveal", "", "", ""], dtype=object)
    data.non_tensor_batch["state_group_selection_type"] = np.array(["random", "random", "best", "best"], dtype=object)

    advantages, _ = compute_advantage_fn(data, min_group_size=4)
    last_token = advantages.numpy()[:, -1]
    assert last_token[0] > 0
    assert last_token[1] < 0
    assert last_token[2] == 0
    assert last_token[3] == 0
    assert data.meta_info["state_group_best_reward_mean"] == 1.0
    assert data.meta_info["state_group_selected_oracle_rate"] == 0.5
    assert data.meta_info["state_group_selected_safe_reveal_rate"] == 0.5
    assert data.meta_info["state_group_selected_certain_flag_rate"] == 0.0
    assert data.meta_info["state_group_selected_guess_rate"] == 0.0
    assert data.meta_info["state_group_selected_non_oracle_reveal_rate"] == 0.5
    assert data.meta_info["state_group_selected_non_oracle_flag_rate"] == 0.0
    assert data.meta_info["state_group_random_selected_rate"] == 0.5
    assert data.meta_info["state_group_best_selected_rate"] == 0.5
    assert data.meta_info["state_group_random_selected_oracle_rate"] == 1.0
    assert data.meta_info["state_group_best_selected_oracle_rate"] == 0.0
    assert data.meta_info["state_group_random_selected_valid_action_rate"] == 1.0
    assert data.meta_info["state_group_best_selected_valid_action_rate"] == 1.0
    assert data.meta_info["state_group_candidate_valid_action_rate"] == 0.75
    assert data.meta_info["state_group_candidate_invalid_action_rate"] == 0.25
    assert data.meta_info["state_group_candidate_oracle_rate"] == 0.25
    assert data.meta_info["state_group_candidate_oracle_reveal_rate"] == 0.25
    assert data.meta_info["state_group_candidate_oracle_flag_rate"] == 0.0
    assert data.meta_info["state_group_candidate_safe_reveal_rate"] == 0.25
    assert data.meta_info["state_group_candidate_certain_flag_rate"] == 0.0
    assert data.meta_info["state_group_candidate_guess_rate"] == 0.0
    assert data.meta_info["state_group_candidate_non_oracle_reveal_rate"] == 0.25
    assert data.meta_info["state_group_candidate_non_oracle_flag_rate"] == 0.5
    assert data.meta_info["state_group_skipped_equal_reward_rate"] == 0.5
    assert data.meta_info["state_group_train_sample_rate"] == 0.5
    assert data.batch["response_mask"][:2].sum().item() > 0
    assert data.batch["response_mask"][2:].sum().item() == 0
    np.testing.assert_array_equal(
        data.non_tensor_batch["vpr_skip_loss"],
        np.array([False, False, True, True]),
    )


def test_state_group_zero_std_group_gets_zero_advantage_and_zero_loss_mask():
    data = make_mock_data([0.3, 0.3], [0, 0])
    data.non_tensor_batch["state_group_uid"] = np.array(["same", "same"], dtype=object)
    advantages, _ = compute_advantage_fn(data, min_group_size=4)
    assert np.allclose(advantages.numpy(), 0.0)
    assert data.batch["response_mask"].sum().item() == 0
    np.testing.assert_array_equal(data.non_tensor_batch["vpr_skip_loss"], np.array([True, True]))
    assert data.meta_info["state_group_zero_std_rate"] == 1.0
    assert data.meta_info["state_group_skipped_equal_reward_rate"] == 1.0
    assert data.meta_info["state_group_train_sample_rate"] == 0.0


def test_vpr_state_group_skip_update_threshold_uses_equal_reward_rate():
    from verl.trainer.ppo.ray_trainer import _should_skip_vpr_state_group_update

    should_skip, equal_rate, product = _should_skip_vpr_state_group_update(
        {
            "state_group_skipped_equal_reward_rate": 0.95,
            "state_group_skipped_oracle_rate": 0.99,
        },
        {"skip_update_equal_reward_threshold": 0.9},
    )
    assert should_skip is True
    assert equal_rate == pytest.approx(0.95)
    assert product == pytest.approx(0.9405)

    should_skip, equal_rate, product = _should_skip_vpr_state_group_update(
        {
            "state_group_skipped_equal_reward_rate": 0.89,
            "state_group_skipped_oracle_rate": 1.0,
        },
        {"skip_update_equal_reward_threshold": 0.9},
    )
    assert should_skip is False
    assert equal_rate == pytest.approx(0.89)
    assert product == pytest.approx(0.89)


def test_vpr_state_group_skip_update_threshold_null_disables_gate():
    from verl.trainer.ppo.ray_trainer import _should_skip_vpr_state_group_update

    should_skip, equal_rate, product = _should_skip_vpr_state_group_update(
        {
            "state_group_skipped_equal_reward_rate": 1.0,
            "state_group_skipped_oracle_rate": 1.0,
        },
        {"skip_update_equal_reward_threshold": None},
    )
    assert should_skip is False
    assert equal_rate == 0.0
    assert product == 0.0
