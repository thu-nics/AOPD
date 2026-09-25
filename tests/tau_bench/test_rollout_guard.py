"""Exercise the real rollout metadata bridge and the actual DAPO estimator."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.environments.env_package.tau_bench.manager import TauBenchEnvironmentManager
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, _pad_compacted_policy_batch, compute_advantage


@pytest.mark.parametrize("all_negative", [False, True])
@pytest.mark.parametrize("mode", ["mean_then_batch_whiten", "group_whiten"])
def test_rollout_guard_survives_masking_compaction_and_padding(all_negative, mode):
    manager = TauBenchEnvironmentManager(None, None, None)
    rewards = [-1.0] * 4 if all_negative else [-1.0, 0.0, 0.0, 0.0]

    class Environment:
        def reset(self, **kwargs):
            return {"text": ["state", "masked state"]}, [{}, {}]

        def state_group_step(self, groups, **kwargs):
            assert len(groups) == 2 and all(len(group) == 4 for group in groups)
            candidates = [
                [("next", reward, True, {"tau_domain": "airline", "vpr_game": "tau_airline", "transfer_without_tool": rank == 0, "semantic_train_mask": not masked, "runtime_train_mask": True, "action_kind": "message", "state_group_advanced": rank == 1}) for rank, reward in enumerate(rewards)]
                for masked in (False, True)
            ]
            return candidates, [1, 1], {"text": ["done", "done"]}, np.asarray([rewards[1]] * 2), np.ones(2, dtype=bool), [group[1][3] for group in candidates]

        success_evaluator = manager.success_evaluator

    collector = object.__new__(TrajectoryCollector)
    collector.config = OmegaConf.create({"env": {"env_name": "metadata_fixture", "max_steps": 1, "rollout": {"n": 4}}})
    collector.tokenizer = SimpleNamespace(batch_decode=lambda tokens, **kwargs: ["message"] * len(tokens))

    def preprocess(gen_batch, obs):
        n = len(gen_batch)
        return DataProto.from_dict(tensors={name: torch.ones((n, 2), dtype=torch.long) for name in ("input_ids", "attention_mask", "position_ids")}, non_tensors={"raw_prompt_ids": np.asarray([[1, 1]] * n, dtype=object)})

    def generate(batch):
        return DataProto.from_dict(tensors={name: torch.ones((len(batch), 2), dtype=torch.long) for name in ("input_ids", "responses")})

    collector.preprocess_batch = preprocess
    actor = SimpleNamespace(world_size=3, generate_sequences=generate)  # 8 -> 9 -> 8 inference padding.
    episodes, _, _, metrics, _, _ = collector._state_group_multi_turn_loop_once(DataProto.from_dict(tensors={"input_ids": torch.ones((2, 1))}), actor, Environment())
    assert metrics["env/transfer_without_tool_candidate_rate"].item() == 0.25
    rows = [row for episode in episodes for row in episode]
    assert [row["semantic_train_mask"] for row in rows] == [True] * 4 + [False] * 4
    assert [row["transfer_without_tool"] for row in rows] == [True, False, False, False] * 2
    non_tensors = {name: np.asarray([row[name] for row in rows]) for name in ("uid", "state_group_uid", "vpr_game", "rewards", "semantic_train_mask", "runtime_train_mask", "transfer_without_tool")}
    non_tensors["is_padding"] = np.zeros(8, dtype=bool)
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones((8, 2)), "token_level_rewards": torch.tensor([[0.0, r] for r in rewards * 2])}, non_tensors=non_tensors)
    # Exercise actual policy-row selection/padding, not just copying dictionaries.
    batch, padding = _pad_compacted_policy_batch(batch, np.arange(8), divisor=9)
    assert padding == 1 and batch.non_tensor_batch["is_padding"][-1]
    batch = compute_advantage(batch, AdvantageEstimator.DAPO, state_group={"advantage_mode": mode})
    metric = "dapo/transfer_without_tool_positive_advantage_rate"
    if all_negative:
        assert batch.batch["advantages"].abs().sum() == 0
        assert metric not in batch.meta_info
    else:
        assert batch.batch["advantages"][0].max() < 0
        assert batch.meta_info[metric] == 0.0
        assert batch.batch["advantages"][4:].abs().sum() == 0
        compacted, _ = _pad_compacted_policy_batch(batch, np.arange(4), divisor=3)
        assert compacted.non_tensor_batch["transfer_without_tool"].tolist() == [True, False, False, False, True, False]
        assert compacted.batch["response_mask"][4:].sum() == 0
