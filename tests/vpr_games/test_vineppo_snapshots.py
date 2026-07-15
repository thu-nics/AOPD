import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import pytest


def _load_with_ray_identity(name: str, path: str):
    try:
        import ray as real_ray
        orig_remote = real_ray.remote
        real_ray.remote = lambda cls: cls
        restore = lambda: setattr(real_ray, "remote", orig_remote)
    except ModuleNotFoundError:
        real_ray = MagicMock()
        real_ray.remote = lambda cls: cls
        sys.modules["ray"] = real_ray
        restore = lambda: None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        restore()


def test_minesweeper_snapshot_restore_roundtrip():
    pytest.importorskip("gem")
    mod = _load_with_ray_identity(
        "_vine_ms_envs",
        "agent_system/environments/env_package/vpr_games/minesweeper/envs.py",
    )
    worker = mod.MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=10)
    obs0, _ = worker.reset(seed=0)
    snap = worker.snapshot_state()
    worker.step("<action>reveal 1 1</action>")
    obs_restored, info = worker.restore_state(snap)

    assert obs_restored == obs0
    assert info["observation"] == obs0


def test_sudoku_snapshot_restore_roundtrip():
    mod = _load_with_ray_identity(
        "_vine_sudoku_envs",
        "agent_system/environments/env_package/vpr_games/sudoku/envs.py",
    )
    worker = mod.SudokuWorker(seed=0, max_turns=10, clues=10)
    obs0, info0 = worker.reset(seed=0)
    snap = worker.snapshot_state()
    r_s, c_s = info0["available_actions"][0].split()
    r, c = int(r_s) - 1, int(c_s) - 1
    digit = worker._env.full_grid[r][c]
    worker.step(f"<action>row {r + 1} col {c + 1} digit {digit}</action>")
    obs_restored, info = worker.restore_state(snap)

    assert obs_restored == obs0
    assert info["observation"] == obs0


def test_sokoban_snapshot_restore_roundtrip_if_available():
    pytest.importorskip("gym_sokoban")
    mod = _load_with_ray_identity(
        "_vine_sokoban_envs",
        "agent_system/environments/env_package/vpr_games/sokoban/envs.py",
    )
    worker = mod.SokobanWorker(seed=0, dim_room=(6, 6), num_boxes=1, max_steps=10, search_depth=20)
    obs0, _ = worker.reset(seed=0)
    snap = worker.snapshot_state()
    worker.step("<action>up</action>")
    obs_restored, info = worker.restore_state(snap)

    assert obs_restored == obs0
    assert info["observation"] == obs0


class _FakeBatch:
    def __init__(self):
        self.non_tensor_batch = {
            "vine_pre_snapshot": np.asarray(["s0"], dtype=object),
            "vine_post_snapshot": np.asarray([None], dtype=object),
            "vine_has_post_snapshot": np.asarray([False], dtype=bool),
            "vine_state_uid": np.asarray(["traj:s:0"], dtype=object),
            "vine_next_state_uid": np.asarray([""], dtype=object),
            "data_source": np.asarray(["fake"], dtype=object),
            "is_padding": np.asarray([False], dtype=bool),
        }
        self.meta_info = {}

    def __len__(self):
        return 1


class _FakeVineEnv:
    def __init__(self):
        self.step_counts = [0]
        self.restored_main = False

    def snapshot_states(self, active_indices=None):
        if active_indices is None:
            return ["main"]
        return [f"s{self.step_counts[int(i)]}" for i in active_indices]

    def restore_states(self, snapshots):
        if snapshots == ["main"]:
            self.restored_main = True
            self.step_counts = [0]
            return {"text": ["main"], "image": None, "anchor": None}, [{}]
        self.step_counts = [int(str(snapshot)[1:]) for snapshot in snapshots]
        return {"text": [f"state-{count}" for count in self.step_counts], "image": None, "anchor": None}, [{} for _ in snapshots]

    def step(self, actions):
        assert actions == ["go"] * len(self.step_counts)
        self.step_counts = [count + 1 for count in self.step_counts]
        dones = np.asarray([count >= 2 for count in self.step_counts], dtype=bool)
        return (
            {"text": [f"state-{count}" for count in self.step_counts], "image": None, "anchor": None},
            np.ones(len(self.step_counts), dtype=np.float32),
            dones,
            [{} for _ in self.step_counts],
        )


class _FakeUnevenDoneVineEnv:
    def __init__(self):
        self.step_counts = []

    def snapshot_states(self, active_indices=None):
        if active_indices is None:
            return ["main0", "main1"]
        return [f"s{self.step_counts[int(i)]}" for i in active_indices]

    def restore_states(self, snapshots):
        if snapshots == ["main0", "main1"]:
            self.step_counts = [0, 0]
            return {"text": ["main0", "main1"], "image": None, "anchor": None}, [{}, {}]
        self.step_counts = [int(str(snapshot)[1:]) for snapshot in snapshots]
        return {"text": [f"state-{i}-{count}" for i, count in enumerate(self.step_counts)], "image": None, "anchor": None}, [{} for _ in snapshots]

    def step(self, actions):
        assert actions == ["go"] * len(self.step_counts)
        rewards = np.ones(len(self.step_counts), dtype=np.float32)
        dones = []
        for worker_idx, count in enumerate(self.step_counts):
            next_count = count + 1
            self.step_counts[worker_idx] = next_count
            # Worker 0 finishes after one step; worker 1 finishes after two.
            dones.append(next_count >= worker_idx + 1)
        return (
            {"text": [f"state-{count}" for count in self.step_counts], "image": None, "anchor": None},
            rewards,
            np.asarray(dones, dtype=bool),
            [{} for _ in self.step_counts],
        )



def test_vineppo_mc_estimation_passes_generation_meta_and_discounts_rewards():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    collector = TrajectoryCollector(
        config=SimpleNamespace(env=SimpleNamespace(max_steps=2)),
        tokenizer=None,
    )
    seen_meta = []

    seen_chat_kwargs = []

    def fake_generate(gen_batch, obs, actor_rollout_wg, apply_chat_template_kwargs=None):
        seen_meta.append(dict(gen_batch.meta_info))
        seen_chat_kwargs.append(dict(apply_chat_template_kwargs or {}))
        assert gen_batch.meta_info["eos_token_id"] == 151645
        assert gen_batch.meta_info["pad_token_id"] == 151643
        assert gen_batch.meta_info["do_sample"] is True
        return ["go"]

    collector._generate_one_step_actions = fake_generate
    batch = _FakeBatch()

    collector.estimate_vine_values_for_batch(
        batch=batch,
        actor_rollout_wg=object(),
        envs=_FakeVineEnv(),
        vine_cfg={"num_rollouts_per_state": 1, "gamma": 0.5, "mc_enable_thinking": "False"},
        generation_meta_info={"eos_token_id": 151645, "pad_token_id": 151643, "do_sample": True},
    )

    np.testing.assert_allclose(batch.non_tensor_batch["vine_v_curr"], [1.5], atol=1e-6)
    np.testing.assert_allclose(batch.non_tensor_batch["vine_v_next"], [0.0], atol=1e-6)
    assert batch.meta_info["vine_num_states"] == 1.0
    assert batch.meta_info["vine_num_mc_rollouts"] == 1.0
    assert batch.meta_info["vineppo/mc_return_mean"] == pytest.approx(1.5)
    assert batch.meta_info["vineppo/mc_positive_return_rate"] == 1.0
    assert batch.meta_info["vineppo/mc_constant_return_state_rate"] == 1.0
    assert batch.meta_info["vineppo/mc_first_action_unique_rate"] == 1.0
    assert len(seen_meta) == 2
    assert seen_chat_kwargs == [{"enable_thinking": False}, {"enable_thinking": False}]



def test_vineppo_mc_estimation_does_not_double_count_terminal_returns():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    collector = TrajectoryCollector(
        config=SimpleNamespace(env=SimpleNamespace(max_steps=2)),
        tokenizer=None,
    )

    def fake_generate(gen_batch, obs, actor_rollout_wg, apply_chat_template_kwargs=None):
        return ["go"] * len(obs["text"])

    collector._generate_one_step_actions = fake_generate
    batch = _FakeBatch()

    collector.estimate_vine_values_for_batch(
        batch=batch,
        actor_rollout_wg=object(),
        envs=_FakeUnevenDoneVineEnv(),
        vine_cfg={"num_rollouts_per_state": 2, "gamma": 1.0},
        generation_meta_info={},
    )

    # The two MC continuations return 1 and 2. The terminal second rollout
    # must not be appended once when done and once again as a truncated live row.
    np.testing.assert_allclose(batch.non_tensor_batch["vine_v_curr"], [1.5], atol=1e-6)
    assert batch.meta_info["vineppo/mc_return_std"] == pytest.approx(0.5)
    assert batch.meta_info["vineppo/mc_constant_return_state_rate"] == 0.0
    assert batch.meta_info["vineppo/mc_first_action_unique_rate"] == 0.5



def test_vineppo_mc_estimation_rejects_unestimated_state_sampling_options():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    collector = TrajectoryCollector(
        config=SimpleNamespace(env=SimpleNamespace(max_steps=1)),
        tokenizer=None,
    )
    batch = _FakeBatch()

    with pytest.raises(ValueError, match="max_states_per_batch"):
        collector.estimate_vine_values_for_batch(
            batch=batch,
            actor_rollout_wg=object(),
            envs=_FakeVineEnv(),
            vine_cfg={"max_states_per_batch": 1},
            generation_meta_info={},
        )

    with pytest.raises(ValueError, match="state_stride"):
        collector.estimate_vine_values_for_batch(
            batch=batch,
            actor_rollout_wg=object(),
            envs=_FakeVineEnv(),
            vine_cfg={"state_stride": 2},
            generation_meta_info={},
        )
