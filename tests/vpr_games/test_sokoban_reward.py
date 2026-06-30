"""Tests for the VPR Sokoban oracle and reward adapter."""

import importlib.util
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _load_direct(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import ray as _real_ray  # noqa: F401
    _ray_available = True
except ModuleNotFoundError:
    _ray_available = False
    _ray_stub = MagicMock()
    _ray_stub.remote = lambda cls: cls
    sys.modules["ray"] = _ray_stub

if _ray_available:
    import ray as _ray_real
    _orig_ray_remote = _ray_real.remote
    _ray_real.remote = lambda cls: cls
else:
    _orig_ray_remote = None

_sok_mod = _load_direct(
    "_testenvs_sokoban_envs",
    "agent_system/environments/env_package/vpr_games/sokoban/envs.py",
)

if _ray_available and _orig_ray_remote is not None:
    _ray_real.remote = _orig_ray_remote


def test_shortest_first_actions_for_one_push_map():
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = room_fixed.copy()
    room_state[1, 1] = 5
    room_state[1, 2] = 4

    actions, distance = _sok_mod._shortest_first_actions(room_fixed, room_state, 10)

    assert actions == [4]
    assert distance == 1


def test_ineffective_action_has_no_transition():
    room_fixed = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    room_state = room_fixed.copy()
    room_state[1, 1] = 5

    assert _sok_mod._apply_action(room_fixed, room_state, 1) is None


def test_worker_invalid_parse_penalty_if_gym_sokoban_available():
    pytest.importorskip("gym_sokoban")
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
        invalid_penalty=-2.0,
    )
    worker.reset(seed=0)
    _, reward, done, info = worker.step("missing action tag")

    assert reward == -2.0
    assert done
    assert info["terminal_reason"] == "invalid_action"
    assert info["illegal_action"]
    assert info["move_optimal"] is None


def test_unsolvable_corner_has_no_oracle_path():
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = room_fixed.copy()
    room_state[1, 3] = 4
    room_state[1, 2] = 5

    actions, distance = _sok_mod._shortest_first_actions(room_fixed, room_state, 10)

    assert actions == []
    assert distance is None


def test_worker_unsolvable_post_action_penalty_if_gym_sokoban_available():
    pytest.importorskip("gym_sokoban")
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
        invalid_penalty=-2.0,
    )
    worker.reset(seed=0)
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = room_fixed.copy()
    room_state[1, 1] = 5
    room_state[1, 2] = 4
    worker._env.room_fixed = room_fixed.copy()
    worker._env.room_state = room_state.copy()
    worker._env.player_position = np.array([1, 1])
    worker._env.boxes_on_target = 0
    worker._env.num_env_steps = 0
    worker._step_count = 0
    worker._done = False

    _, reward, done, info = worker.step("<action>right</action>")

    assert reward == -2.0
    assert done
    assert info["terminal_success"] is False
    assert info["terminal_reason"] == "deadlock"
    assert not info["illegal_action"]
    assert info["action_effective"] is True
