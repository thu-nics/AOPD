import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, compute_advantage


def test_single_turn_grpo_does_not_require_agent_trajectory_ids():
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.asarray(["a", "a", "b", "b"], dtype=object)},
    )

    result = compute_advantage(data, AdvantageEstimator.GRPO)

    row_advantages = result.batch["advantages"][:, 0]
    assert row_advantages[0] < 0 < row_advantages[1]
    assert row_advantages[2] < 0 < row_advantages[3]
    assert torch.isfinite(result.batch["advantages"]).all()


def test_dapo_uses_state_group_uid_for_agent_rows():
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
            "state_group_uid": np.asarray(["s0", "s0", "s1", "s1"], dtype=object),
            "rewards": np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    row_advantages = result.batch["advantages"][:, 0]
    assert row_advantages[0] < 0 < row_advantages[1]
    torch.testing.assert_close(row_advantages[2:], torch.zeros(2))


def test_dapo_trajectory_level_advantage_propagates_terminal_outcome_to_all_turns():
    rewards = torch.tensor(
        [
            [0.0, 0.0],  # successful trajectory, turn 0
            [0.0, 0.0],  # failed trajectory, turn 0
            [0.0, 1.0],  # successful trajectory, terminal turn
            [0.0, 0.0],  # failed trajectory, terminal turn
        ],
        dtype=torch.float32,
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["group"] * 4, dtype=object),
            "traj_uid": np.asarray(["success", "failure", "success", "failure"], dtype=object),
        },
    )

    result = compute_advantage(
        data,
        AdvantageEstimator.DAPO,
        dapo_trajectory_level_advantage=True,
    )

    advantages = result.batch["advantages"][:, 0]
    assert advantages[0] > 0
    assert advantages[2] > 0
    assert advantages[1] < 0
    assert advantages[3] < 0
    torch.testing.assert_close(advantages[0], advantages[2])
    torch.testing.assert_close(advantages[1], advantages[3])


def test_dapo_accepts_variable_state_group_sizes():
    raw_rewards = np.asarray(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    token_rewards = torch.zeros((len(raw_rewards), 2), dtype=torch.float32)
    token_rewards[:, -1] = torch.from_numpy(raw_rewards)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_rewards,
            "response_mask": torch.ones_like(token_rewards),
        },
        non_tensors={
            "uid": np.asarray([f"t{index}" for index in range(12)], dtype=object),
            "state_group_uid": np.asarray(
                ["math-state"] * 8 + ["game-state"] * 4, dtype=object
            ),
            "rewards": raw_rewards,
            "vpr_game": np.asarray(["math"] * 8 + ["sudoku"] * 4, dtype=object),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    advantages = result.batch["advantages"][:, 0]
    assert torch.isfinite(advantages).all()
    assert torch.any(advantages[:8] < 0) and torch.any(advantages[:8] > 0)
    assert torch.any(advantages[8:] < 0) and torch.any(advantages[8:] > 0)
    assert result.meta_info["dapo/raw_state_groups"] == 2.0
    assert result.meta_info["dapo/effective_state_groups"] == 2.0
    assert result.meta_info["dapo/math/raw_state_groups"] == 1.0
    assert result.meta_info["dapo/sudoku/raw_state_groups"] == 1.0



def test_dapo_ignores_divisibility_padding_rows():
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["t0", "t1", "t1"], dtype=object),
            "state_group_uid": np.asarray(["s0", "s0", "s0"], dtype=object),
            "rewards": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
            "is_padding": np.asarray([False, False, True]),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    row_advantages = result.batch["advantages"][:, 0]
    assert row_advantages[0] < 0 < row_advantages[1]
    torch.testing.assert_close(row_advantages[2], torch.tensor(0.0))
    torch.testing.assert_close(result.batch["response_mask"][2], torch.zeros(2))


def test_state_group_dapo_filters_raw_equal_reward_groups_before_length_shaping():
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, -0.5], [0.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["t0", "t0", "t1", "t1"], dtype=object),
            "traj_uid": np.asarray(["t0", "t0", "t1", "t1"], dtype=object),
            "state_group_uid": np.asarray(["s0", "s0", "s1", "s1"], dtype=object),
            "rewards": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "vpr_game": np.asarray(["sokoban", "sokoban", "sudoku", "sudoku"], dtype=object),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.DAPO)

    torch.testing.assert_close(result.batch["advantages"][:2], torch.zeros(2, 2))
    torch.testing.assert_close(result.batch["response_mask"][:2], torch.zeros(2, 2))
    assert result.non_tensor_batch["dapo_skip_loss"].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert result.meta_info["dapo/skipped_equal_reward_rate"] == 0.5
    assert result.meta_info["dapo/sokoban/skipped_equal_reward_rate"] == 1.0
    assert result.meta_info["dapo/sokoban/train_sample_rate"] == 0.0
    assert result.meta_info["dapo/sudoku/effective_state_groups"] == 1.0
    assert result.meta_info["dapo/sudoku/train_sample_rate"] == 1.0
    assert torch.isfinite(result.batch["advantages"]).all()


def test_data_metrics_accept_standard_single_turn_dapo_batch():
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    zeros = torch.zeros((2, 2), dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "token_level_scores": zeros,
            "token_level_rewards": zeros,
            "advantages": zeros,
            "returns": zeros,
            "responses": torch.ones((2, 2), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
        },
        non_tensors={"uid": np.asarray(["a", "b"], dtype=object)},
    )

    metrics = compute_data_metrics(data, use_critic=False)

    assert metrics["critic/score/mean"] == 0.0
    assert "episode/reward/mean" not in metrics


def test_validation_supports_standard_single_turn_rollout():
    raw_prompt_ids = np.empty(2, dtype=object)
    raw_prompt_ids[0] = [10, 11]
    raw_prompt_ids[1] = [20, 21]
    test_data = {
        "input_ids": torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "position_ids": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        "raw_prompt_ids": raw_prompt_ids,
        "data_source": np.asarray(["math_dapo", "math_dapo"], dtype=object),
    }

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            return " ".join(str(int(token)) for token in token_ids)

    class ActorRollout:
        world_size = 8

        @staticmethod
        def generate_sequences(gen_batch):
            batch_size = len(gen_batch)
            attention_mask = torch.ones((batch_size, 4), dtype=torch.long)
            attention_mask[1, -1] = 0
            return DataProto.from_dict(
                tensors={
                    "prompts": gen_batch.batch["input_ids"],
                    "responses": torch.tensor(
                        [[12, 13]] * batch_size, dtype=torch.long
                    ),
                    "attention_mask": attention_mask,
                }
            )

    class RewardFn:
        @staticmethod
        def __call__(batch, return_dict=False):
            assert batch.non_tensor_batch["data_source"].tolist() == [
                "math_dapo",
                "math_dapo",
            ]
            reward = torch.zeros_like(
                batch.batch["responses"], dtype=torch.float32
            )
            reward[0, -1] = 0.75
            reward[1, 0] = -1.0
            result = {
                "reward_tensor": reward,
                "reward_extra_info": {
                    "score": [1.0, -1.0],
                    "acc": [True, False],
                    "pred": ["42", "[INVALID]"],
                    "overlong_reward": [-0.25, 0.0],
                    "overlong": [True, False],
                },
            }
            return result if return_dict else reward

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "val_kwargs": {"n": 1, "do_sample": True}
                }
            },
            "reward_model": {
                "enable": False,
                "reward_manager": "dapo",
            },
            "trainer": {
                "log_val_generations": 0,
                "validation_data_dir": None,
            },
        }
    )
    trainer.tokenizer = Tokenizer()
    trainer.val_dataloader = [test_data]
    trainer.actor_rollout_wg = ActorRollout()
    trainer.val_reward_fn = RewardFn()
    trainer.traj_collector = None
    trainer.val_envs = None
    trainer.global_steps = 0

    metrics = trainer._validate()

    assert metrics == {
        "val/math_dapo/test_score": -0.125,
        "val/math_dapo/num_samples": 2,
        "val/math_dapo/accuracy": 0.5,
        "val/math_dapo/correct_count": 1,
        "val/math_dapo/raw_score/mean": 0.0,
        "val/math_dapo/valid_answer_rate": 0.5,
        "val/math_dapo/invalid_answer_rate": 0.5,
        "val/math_dapo/overlong_rate": 0.5,
        "val/math_dapo/overlong_penalty/mean": -0.125,
        "val/math_dapo/response_length/mean": 1.5,
        "val/math_dapo/response_length/p50": 1.5,
        "val/math_dapo/response_length/p95": 1.95,
        "val/math_dapo/response_length/max": 2.0,
        "val/math_dapo/response_length/clip_ratio": 0.5,
    }

def test_dynamic_dapo_last_refill_is_capped_to_complete_target_groups():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "env": {"rollout": {"n": 4}},
            "algorithm": {"filter_groups": {"max_num_gen_batches": 2}},
        }
    )
    calls = 0

    def fake_rollout(**kwargs):
        nonlocal calls
        calls += 1
        batch_list = [
            [{"uid": f"group-{index // 4}"}] for index in range(32)
        ]
        rewards = np.zeros(32, dtype=np.float32)
        if calls == 1:
            rewards[:4] = np.asarray([0.0, 1.0, 0.0, 1.0])
        return (
            batch_list,
            rewards,
            np.ones(32, dtype=np.float32),
            {"env/success_rate": rewards.copy()},
            np.asarray([f"traj-{index}" for index in range(32)], dtype=object),
            np.zeros(32, dtype=np.float32),
        )

    collector.vanilla_multi_turn_loop = fake_rollout
    result = collector.dynamic_multi_turn_loop(None, None, None)

    assert calls == 2
    assert len(result[0]) == 32
    assert all(len(values) == 32 for values in result[1:3])
    assert len(result[3]["env/success_rate"]) == 32
    assert all(len(values) == 32 for values in result[4:])
