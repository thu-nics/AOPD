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
    raw_prompt_ids = np.empty(1, dtype=object)
    raw_prompt_ids[0] = [10, 11]
    test_data = {
        "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "raw_prompt_ids": raw_prompt_ids,
        "data_source": np.asarray(["math_dapo"], dtype=object),
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
            return DataProto.from_dict(
                tensors={
                    "prompts": gen_batch.batch["input_ids"],
                    "responses": torch.tensor(
                        [[12, 13]] * batch_size, dtype=torch.long
                    ),
                    "attention_mask": torch.ones(
                        (batch_size, 4), dtype=torch.long
                    ),
                }
            )

    class RewardFn:
        @staticmethod
        def __call__(batch, return_dict=False):
            assert batch.non_tensor_batch["data_source"].tolist() == [
                "math_dapo"
            ]
            reward = torch.zeros_like(
                batch.batch["responses"], dtype=torch.float32
            )
            reward[:, -1] = 1.0
            result = {"reward_tensor": reward, "reward_extra_info": {}}
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

    assert metrics == {"val/math_dapo/test_score": 1.0}
