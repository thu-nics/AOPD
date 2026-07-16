from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_package.vpr_games.mixed.envs import MixedVPRMultiProcessEnv, interleave_counts
from agent_system.environments.env_package.vpr_games.mixed.manager import MixedVPRManager
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def test_interleave_counts_preserves_dapo_mix_and_spreads_sudoku():
    labels = interleave_counts({"math": 64, "sokoban": 9, "sudoku": 3, "minesweeper": 20})
    assert Counter(labels) == Counter(math=64, sokoban=9, sudoku=3, minesweeper=20)
    sudoku_positions = [index for index, label in enumerate(labels) if label == "sudoku"]
    assert max(b - a for a, b in zip(sudoku_positions, sudoku_positions[1:])) <= 33


def test_mixed_success_metrics_are_not_zero_diluted():
    manager = MixedVPRManager.__new__(MixedVPRManager)
    manager.config = OmegaConf.create(
        {"env": {"history_length": 0, "sokoban": {"num_boxes": 2}, "minesweeper": {"rows": 5, "cols": 5, "mines": 2}}}
    )
    total_infos = [
        [{"vpr_game": "sokoban", "terminal_success": True, "completion_rate": 0.5}],
        [{"vpr_game": "sudoku", "terminal_success": False, "completion_rate": 0.25}],
        [{"vpr_game": "minesweeper", "terminal_success": True, "completion_rate": 0.75}],
    ]
    metrics = manager.success_evaluator(total_infos=total_infos)
    np.testing.assert_allclose(metrics["env/sokoban/success_rate"], [1.0])
    np.testing.assert_allclose(metrics["env/sudoku/success_rate"], [0.0])
    np.testing.assert_allclose(metrics["env/minesweeper/success_rate"], [1.0])
    np.testing.assert_allclose(metrics["env/sudoku/completion_rate"], [0.25])
    np.testing.assert_allclose(metrics["env/sokoban/trajectory_count"], [1.0])
    assert "env/sokoban/mine_hit_rate" not in metrics
    assert "env/minesweeper/boxes_on_target" not in metrics
    assert "env/mine_hit_rate" not in metrics
    assert "env/boxes_on_target" not in metrics
    np.testing.assert_allclose(metrics["env/completion_rate"], [0.5, 0.25, 0.75])


def test_mixed_snapshot_restore_fails_before_cross_game_state_corruption():
    envs = MixedVPRMultiProcessEnv([], [], [])
    with pytest.raises(NotImplementedError, match="snapshot-based VinePPO"):
        envs.snapshot_states()
    with pytest.raises(NotImplementedError, match="snapshot-based VinePPO"):
        envs.restore_states([])


def test_mixed_math_prompt_pool_rotates_between_dynamic_attempts():
    envs = MixedVPRMultiProcessEnv([None], [0], ["math"])
    kwargs = [
        {
            "task": "math",
            "question": "q0",
            "ground_truth": "a0",
            "data_source": "dapo",
            "question_pool": ["q0", "q1"],
            "ground_truth_pool": ["a0", "a1"],
            "data_source_pool": ["dapo", "dapo"],
        }
    ]

    first_observations, _ = envs.reset(kwargs)
    second_kwargs = [{**kwargs[0], "dynamic_attempt": 1}]
    second_observations, _ = envs.reset(second_kwargs)

    assert first_observations == ["q0"]
    assert second_observations == ["q1"]
    envs.close()


def test_mixed_dapo_dynamic_sampling_keeps_games_once_and_refills_math():
    game = [[{"vpr_game": "sokoban", "rewards": 2.0}]]
    equal_math = [
        {"vpr_game": "math", "rewards": -1.0},
        {"vpr_game": "math", "rewards": -1.0},
    ]
    varied_math = [
        {"vpr_game": "math", "rewards": -1.0},
        {"vpr_game": "math", "rewards": 1.0},
    ]

    class FakeCollector:
        config = SimpleNamespace(
            env=SimpleNamespace(
                env_name="dapo_vpr_mixed",
                mixed=SimpleNamespace(
                    trajectory_counts=SimpleNamespace(math=1)
                ),
            ),
            algorithm=SimpleNamespace(
                filter_groups=SimpleNamespace(
                    enable=True, max_num_gen_batches=2
                )
            ),
        )

        def __init__(self):
            self.results = iter(
                [
                    (
                        [game[0], equal_math],
                        np.asarray([10.0, 11.0]),
                        np.asarray([1.0, 1.0]),
                        {"env/success_rate": np.asarray([1.0, 0.0])},
                        np.asarray(["game-0", "math-0"], dtype=object),
                        np.asarray([0.0, 0.0]),
                    ),
                    (
                        [varied_math],
                        np.asarray([21.0]),
                        np.asarray([1.0]),
                        {"env/success_rate": np.asarray([1.0])},
                        np.asarray(["math-1"], dtype=object),
                        np.asarray([0.0]),
                    ),
                ]
            )

        def _state_group_multi_turn_loop_once(self, gen_batch, *args, **kwargs):
            self.call_batches = getattr(self, "call_batches", [])
            self.call_batches.append(gen_batch)
            return next(self.results)

    class FakeGenBatch:
        def __init__(self, env_kwargs):
            self.non_tensor_batch = {
                "env_kwargs": np.asarray(env_kwargs, dtype=object)
            }

        def select_idxs(self, indices):
            return FakeGenBatch(self.non_tensor_batch["env_kwargs"][indices])

    gen_batch = FakeGenBatch([{"task": "sokoban"}, {"task": "math"}])
    collector = FakeCollector()
    trajectories, rewards, _, success, traj_uids, _ = (
        TrajectoryCollector.state_group_multi_turn_loop(
            collector, gen_batch, actor_rollout_wg=None, envs=None
        )
    )

    assert len(collector.call_batches[0].non_tensor_batch["env_kwargs"]) == 2
    retry_kwargs = collector.call_batches[1].non_tensor_batch["env_kwargs"]
    assert len(retry_kwargs) == 1
    assert retry_kwargs[0]["task"] == "math"
    assert retry_kwargs[0]["dynamic_attempt"] == 1
    assert trajectories == [game[0], varied_math]
    np.testing.assert_allclose(rewards, [10.0, 21.0])
    assert traj_uids.tolist() == ["game-0", "math-1"]
    np.testing.assert_allclose(success["env/success_rate"], [1.0, 1.0])
    np.testing.assert_allclose(success["env/sokoban/trajectory_count"], [1.0])
    np.testing.assert_allclose(success["env/math/trajectory_count"], [1.0])
