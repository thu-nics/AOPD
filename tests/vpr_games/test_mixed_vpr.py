from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_package.vpr_games.mixed.envs import (
    MixedVPRMultiProcessEnv,
    interleave_counts,
    interleave_grouped_counts,
)
from agent_system.environments.env_package.vpr_games.mixed.manager import MixedVPRManager
from agent_system.multi_turn_rollout.rollout_loop import (
    TrajectoryCollector,
    _build_state_group_layout,
    _resolve_state_group_sizes,
    _resolve_train_rollout_limits,
)


def test_interleave_counts_preserves_dapo_mix_and_spreads_sudoku():
    labels = interleave_counts({"math": 64, "sokoban": 9, "sudoku": 3, "minesweeper": 20})
    assert Counter(labels) == Counter(math=64, sokoban=9, sudoku=3, minesweeper=20)
    sudoku_positions = [index for index, label in enumerate(labels) if label == "sudoku"]
    assert max(b - a for a, b in zip(sudoku_positions, sudoku_positions[1:])) <= 33


def test_interleave_grouped_counts_expands_each_base_group_contiguously():
    base = interleave_counts(
        {"math": 2, "sokoban": 1, "sudoku": 1, "minesweeper": 0}
    )
    grouped = interleave_grouped_counts(
        {"math": 2, "sokoban": 1, "sudoku": 1, "minesweeper": 0}, 3
    )

    assert len(grouped) == len(base) * 3
    assert [grouped[index] for index in range(0, len(grouped), 3)] == base
    assert all(len(set(grouped[index:index + 3])) == 1 for index in range(0, len(grouped), 3))


def test_mixed_state_group_sizes_use_math_8_and_games_4():
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "dapo_vpr_mixed",
                "rollout": {"n": 8, "math_n": 8, "game_n": 4},
            }
        }
    )
    infos = [
        {"vpr_game": "math"},
        {"vpr_game": "sokoban"},
        {"vpr_game": "sudoku"},
        {"vpr_game": "minesweeper"},
    ]

    sizes = _resolve_state_group_sizes(config, infos)

    np.testing.assert_array_equal(sizes, [8, 4, 4, 4])


def test_state_group_layout_preserves_variable_candidate_blocks():
    sizes = np.asarray([8, 4, 4, 4], dtype=np.int32)
    active = np.asarray([0, 2, 3], dtype=np.int64)

    active_sizes, repeated, offsets, uids, ranks = _build_state_group_layout(
        active, sizes
    )

    np.testing.assert_array_equal(active_sizes, [8, 4, 4])
    np.testing.assert_array_equal(offsets, [0, 8, 12, 16])
    np.testing.assert_array_equal(repeated, [0] * 8 + [2] * 4 + [3] * 4)
    np.testing.assert_array_equal(ranks, list(range(8)) + list(range(4)) * 2)
    assert len(set(uids[:8])) == 1
    assert len(set(uids[8:12])) == 1
    assert len(set(uids[12:])) == 1
    assert len(set(uids)) == 3


def test_math_only_refill_keeps_math_group_size():
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "dapo_vpr_mixed",
                "rollout": {"n": 8, "math_n": 8, "game_n": 4},
            }
        }
    )

    sizes = _resolve_state_group_sizes(config, [{"vpr_game": "math"}] * 3)

    np.testing.assert_array_equal(sizes, [8, 8, 8])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_mixed_state_group_sizes_reject_invalid_values(value):
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "dapo_vpr_mixed",
                "rollout": {"n": 8, "math_n": 8, "game_n": value},
            }
        }
    )

    with pytest.raises(ValueError, match="env.rollout.game_n"):
        _resolve_state_group_sizes(config, [{"vpr_game": "sudoku"}])


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


def test_mixed_outcome_dynamic_sampling_expands_games_and_refills_math_groups():
    equal_math = [[{"rewards": 0.0}], [{"rewards": 0.0}]]
    game_group = [[{"rewards": 1.0}], [{"rewards": 0.0}]]
    varied_math = [[{"rewards": 0.0}], [{"rewards": 1.0}]]

    class FakeCollector:
        config = SimpleNamespace(
            env=SimpleNamespace(
                rollout=SimpleNamespace(n=2),
                mixed=SimpleNamespace(
                    trajectory_counts=SimpleNamespace(
                        math=1, sokoban=1, sudoku=0, minesweeper=0
                    )
                ),
            ),
            algorithm=SimpleNamespace(
                filter_groups=SimpleNamespace(max_num_gen_batches=2)
            ),
        )

        def __init__(self):
            self.results = iter(
                [
                    (
                        equal_math + game_group,
                        np.asarray([0.0, 0.0, 1.0, 0.0]),
                        np.ones(4),
                        {"env/success_rate": np.asarray([0.0, 0.0, 1.0, 0.0])},
                        np.asarray(["math-0", "math-1", "game-0", "game-1"], dtype=object),
                        np.zeros(4),
                    ),
                    (
                        varied_math,
                        np.asarray([0.0, 1.0]),
                        np.ones(2),
                        {"env/success_rate": np.asarray([0.0, 1.0])},
                        np.asarray(["math-2", "math-3"], dtype=object),
                        np.zeros(2),
                    ),
                ]
            )
            self.call_batches = []

        def vanilla_multi_turn_loop(self, gen_batch, *args, **kwargs):
            self.call_batches.append(gen_batch)
            return next(self.results)

    class FakeGenBatch:
        def __init__(self, env_kwargs):
            self.non_tensor_batch = {
                "env_kwargs": np.asarray(env_kwargs, dtype=object)
            }

        def select_idxs(self, indices):
            return FakeGenBatch(self.non_tensor_batch["env_kwargs"][indices])

        def repeat(self, repeat_times, interleave):
            assert interleave
            return FakeGenBatch(
                [
                    dict(item)
                    for item in self.non_tensor_batch["env_kwargs"]
                    for _ in range(repeat_times)
                ]
            )

    collector = FakeCollector()
    gen_batch = FakeGenBatch([{"task": "math"}, {"task": "sokoban"}])
    trajectories, rewards, _, success, traj_uids, _ = (
        TrajectoryCollector.mixed_outcome_multi_turn_loop(
            collector, gen_batch, actor_rollout_wg=None, envs=None
        )
    )

    assert trajectories == game_group + varied_math
    np.testing.assert_allclose(rewards, [1.0, 0.0, 0.0, 1.0])
    assert traj_uids.tolist() == ["game-0", "game-1", "math-2", "math-3"]
    assert len(collector.call_batches[0].non_tensor_batch["env_kwargs"]) == 4
    retry_kwargs = collector.call_batches[1].non_tensor_batch["env_kwargs"]
    assert len(retry_kwargs) == 2
    assert all(item["task"] == "math" for item in retry_kwargs)
    assert all(item["dynamic_attempt"] == 1 for item in retry_kwargs)
    np.testing.assert_allclose(success["env/math/trajectory_count"], [2.0])
    np.testing.assert_allclose(success["env/sokoban/trajectory_count"], [2.0])


def test_mixed_training_rollout_limit_only_caps_configured_task():
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "dapo_vpr_mixed",
                "max_steps": 40,
                "sokoban": {
                    "max_steps": 24,
                    "train_rollout_max_steps": 10,
                },
                "sudoku": {
                    "max_steps": 40,
                    "train_rollout_max_steps": 10,
                },
                "minesweeper": {"max_steps": 15},
            }
        }
    )
    infos = [
        {"vpr_game": "math"},
        {"vpr_game": "sokoban"},
        {"vpr_game": "sudoku"},
        {"vpr_game": "minesweeper"},
    ]

    limits = _resolve_train_rollout_limits(config, infos)

    np.testing.assert_array_equal(limits, [40, 10, 10, 40])
    collected_turns = [
        sum(step < limit for step in range(config.env.max_steps))
        for limit in limits
    ]
    assert collected_turns == [40, 10, 10, 40]
    assert config.env.sokoban.max_steps == 24
    assert config.env.sudoku.max_steps == 40


@pytest.mark.parametrize("train_limit", [0, 41, 1.5, True])
def test_mixed_training_rollout_limit_rejects_invalid_values(train_limit):
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "dapo_vpr_mixed",
                "max_steps": 40,
                "sudoku": {
                    "max_steps": 40,
                    "train_rollout_max_steps": train_limit,
                },
            }
        }
    )

    with pytest.raises(ValueError, match="train_rollout_max_steps"):
        _resolve_train_rollout_limits(config, [{"vpr_game": "sudoku"}])
