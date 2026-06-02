"""make_envs() factory regression tests.

These tests import the real make_envs() function and verify:
- All three VPR environment names are registered
- Actor counts (train: env_num × group_n, val: env_num × 1)
- Val pool seeded at seed + 1000
- Unsupported environment name calls exit(1)
- Minesweeper grouped reset identity

Requires: Ray, Torch, GEM (skip without them).
Run under /opt/venv/verl-agent/bin/python for full coverage.

IMPORTANT: This file loads the real env_manager under a private alias so it is not
affected by MagicMock stubs that test_envs.py injects into sys.modules. When pytest
collects all test files, module-level stubs from other files run first. Loading under
a private key ("_real_env_manager") isolates this file from those stubs.
"""

import sys
import importlib.util
import pytest

_ray_available = False
try:
    import ray  # noqa: F401
    _ray_available = True
except ModuleNotFoundError:
    pass

_torch_available = False
try:
    import torch  # noqa: F401
    _torch_available = True
except ModuleNotFoundError:
    pass

_gem_available = False
try:
    import gem  # noqa: F401
    _gem_available = True
except ModuleNotFoundError:
    pass

_can_run = _ray_available and _torch_available and _gem_available

pytestmark = pytest.mark.skipif(
    not _can_run,
    reason="Requires Ray, Torch, and GEM (run under /opt/venv/verl-agent/bin/python)",
)

# Load the real env_manager under a private alias so test_envs.py's MagicMock stub
# (at sys.modules["agent_system.environments.env_manager"]) does not interfere.
_make_envs_fn = None
if _can_run:
    _em_spec = importlib.util.spec_from_file_location(
        "_real_env_manager",
        "agent_system/environments/env_manager.py",
    )
    _real_em = importlib.util.module_from_spec(_em_spec)
    sys.modules["_real_env_manager"] = _real_em
    _em_spec.loader.exec_module(_real_em)
    _make_envs_fn = _real_em.make_envs


def _init_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=8, ignore_reinit_error=True)


def _make_config(env_name, train_batch_size=2, rollout_n=2, val_batch_size=1, seed=0):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    return SimpleNamespace(
        env=SimpleNamespace(
            env_name=env_name,
            seed=seed,
            max_steps=9,
            history_length=0,
            invalid_penalty=-1.0,
            rollout=SimpleNamespace(n=rollout_n),
            resources_per_worker=OmegaConf.create({"num_cpus": 0.1, "num_gpus": 0}),
            tictactoe=SimpleNamespace(opponent="random"),
            sudoku=SimpleNamespace(n=3, clues=40, terminate_on_wrong_digit=True),
            minesweeper=SimpleNamespace(rows=5, cols=5, mines=5),
        ),
        data=SimpleNamespace(train_batch_size=train_batch_size, val_batch_size=val_batch_size),
    )


class TestMakeEnvsFactory:
    """Regression tests for make_envs() factory registration and wiring."""

    def setup_method(self):
        _init_ray()

    def teardown_method(self):
        pass

    def _make_envs(self, env_name, **kw):
        # Use the private-alias real make_envs (not the sys.modules stub from test_envs.py)
        config = _make_config(env_name, **kw)
        return _make_envs_fn(config)

    # ----- TicTacToe -----

    def test_tictactoe_registered(self):
        """vpr_tictactoe is recognized and returns (envs, val_envs)."""
        envs, val_envs = self._make_envs("vpr_tictactoe", train_batch_size=2, rollout_n=2)
        assert envs is not None
        assert val_envs is not None
        envs.close()
        val_envs.close()

    def test_tictactoe_train_actor_count(self):
        """train actors = train_batch_size × rollout.n."""
        envs, val_envs = self._make_envs("vpr_tictactoe", train_batch_size=2, rollout_n=2)
        assert len(envs.envs.workers) == 4, f"Expected 4 train actors, got {len(envs.envs.workers)}"
        envs.close()
        val_envs.close()

    def test_tictactoe_val_actor_count(self):
        """val actors = val_batch_size × 1."""
        envs, val_envs = self._make_envs("vpr_tictactoe", val_batch_size=1, rollout_n=2)
        assert len(val_envs.envs.workers) == 1
        envs.close()
        val_envs.close()

    def test_tictactoe_val_seed_is_seed_plus_1000(self):
        """Val pool seeded at seed+1000 (different initial state from train seed)."""
        envs, val_envs = self._make_envs("vpr_tictactoe", seed=0)
        # TicTacToe initial board is always empty — verify seeds differ from train
        # by checking the worker seeds
        train_seed = envs.envs.seeds[0]
        val_seed = val_envs.envs.seeds[0]
        assert val_seed == train_seed + 1000, \
            f"Val seed {val_seed} should be train_seed {train_seed} + 1000"
        envs.close()
        val_envs.close()

    # ----- Sudoku -----

    def test_sudoku_registered(self):
        envs, val_envs = self._make_envs("vpr_sudoku", train_batch_size=2, rollout_n=2)
        assert envs is not None
        assert val_envs is not None
        envs.close()
        val_envs.close()

    def test_sudoku_train_actor_count(self):
        envs, val_envs = self._make_envs("vpr_sudoku", train_batch_size=2, rollout_n=2)
        assert len(envs.envs.workers) == 4
        envs.close()
        val_envs.close()

    def test_sudoku_val_seed_offset(self):
        envs, val_envs = self._make_envs("vpr_sudoku", seed=0)
        assert val_envs.envs.seeds[0] == 1000
        envs.close()
        val_envs.close()

    # ----- Minesweeper -----

    def test_minesweeper_registered(self):
        envs, val_envs = self._make_envs("vpr_minesweeper", train_batch_size=2, rollout_n=2)
        assert envs is not None
        assert val_envs is not None
        envs.close()
        val_envs.close()

    def test_minesweeper_train_actor_count(self):
        envs, val_envs = self._make_envs("vpr_minesweeper", train_batch_size=2, rollout_n=2)
        assert len(envs.envs.workers) == 4
        envs.close()
        val_envs.close()

    def test_minesweeper_val_actor_count(self):
        """val actors = val_batch_size × 1."""
        envs, val_envs = self._make_envs("vpr_minesweeper", val_batch_size=1, rollout_n=2)
        assert len(val_envs.envs.workers) == 1
        envs.close()
        val_envs.close()

    def test_minesweeper_val_seed_is_seed_plus_1000(self):
        """Minesweeper val pool seeded at seed+1000."""
        envs, val_envs = self._make_envs("vpr_minesweeper", seed=0)
        assert val_envs.envs.seeds[0] == 1000, \
            f"Val seed {val_envs.envs.seeds[0]} should be 1000"
        envs.close()
        val_envs.close()

    def test_minesweeper_grouped_reset_identity(self):
        """group_n=2 Minesweeper: both replicas share same initial state."""
        from agent_system.environments.env_package.vpr_games.minesweeper.envs import (
            build_minesweeper_envs
        )
        envs = build_minesweeper_envs(seed=0, env_num=1, group_n=2)
        obs_list, _ = envs.reset()
        assert obs_list[0] == obs_list[1], "Group replicas must start identically"
        envs.close()

    # ----- Unsupported name -----

    def test_unsupported_env_name_exits(self):
        """make_envs() with unknown env_name calls sys.exit(1)."""
        config = _make_config("vpr_unknown")
        with pytest.raises(SystemExit) as exc_info:
            _make_envs_fn(config)
        assert exc_info.value.code == 1
