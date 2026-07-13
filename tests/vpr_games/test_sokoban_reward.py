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


@pytest.mark.parametrize(
    ("initial_len", "remaining_len", "solved", "expected"),
    [
        (12, 12, False, 0.0),
        (12, 3, False, 0.75),
        (12, 15, False, 0.0),
        (12, None, False, 0.0),
        (None, None, True, 1.0),
    ],
)
def test_path_progress_completion(initial_len, remaining_len, solved, expected):
    assert _sok_mod._path_progress_completion(initial_len, remaining_len, solved) == expected


def test_worker_step_completion_uses_post_action_oracle_distance(monkeypatch):
    pytest.importorskip("gym_sokoban")
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
    )
    worker.reset(seed=0)
    worker._initial_shortest_path_len = 10
    worker._step_count = 0
    worker._done = False

    oracle_results = iter([([4], 10), ([4], 6)])
    monkeypatch.setattr(worker, "_oracle_for_current_state", lambda _: next(oracle_results))
    monkeypatch.setattr(
        _sok_mod,
        "_apply_action",
        lambda room_fixed, room_state, action_id: room_state.copy(),
    )
    monkeypatch.setattr(
        worker._env,
        "step",
        lambda action_id: (worker._env.render(worker._mode), 0.0, False, {"won": False}),
    )
    monkeypatch.setattr(worker._env, "success", lambda: False)

    _, _, done, info = worker.step("<action>right</action>")

    assert done is False
    assert info["sokoban_shortest_path_len"] == 10
    assert info["sokoban_remaining_shortest_path_len"] == 6
    assert info["completion_rate"] == pytest.approx(0.4)


def test_snapshot_restore_preserves_initial_oracle_distance():
    pytest.importorskip("gym_sokoban")
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
    )
    worker.reset(seed=0)
    worker._initial_shortest_path_len = 17
    snapshot = worker._snapshot_state()

    worker._initial_shortest_path_len = 3
    worker._restore_state(snapshot)

    assert worker._initial_shortest_path_len == 17


def test_ineffective_action_has_no_transition():
    room_fixed = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    room_state = room_fixed.copy()
    room_state[1, 1] = 5

    assert _sok_mod._apply_action(room_fixed, room_state, 1) is None


def test_process_reward_noise_flips_oracle_and_non_oracle(monkeypatch):
    monkeypatch.setattr(_sok_mod.random, "random", lambda: 0.0)

    reward, applied = _sok_mod._maybe_flip_process_reward(True, 2.0, 0.0, 1.0)
    assert reward == 0.0
    assert applied is True

    reward, applied = _sok_mod._maybe_flip_process_reward(False, 2.0, 0.0, 1.0)
    assert reward == 2.0
    assert applied is True

    reward, applied = _sok_mod._maybe_flip_process_reward(True, 2.0, 0.0, 0.0)
    assert reward == 2.0
    assert applied is False


def test_worker_invalid_parse_penalty_if_gym_sokoban_available():
    pytest.importorskip("gym_sokoban")
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=20,
        invalid_penalty=-2.0,
        reward_noise_prob=1.0,
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
        reward_noise_prob=1.0,
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
    assert info["reward_noise_applied"] is False


def test_worker_reset_reseeds_when_initial_oracle_missing(monkeypatch):
    pytest.importorskip("gym_sokoban")
    calls = []

    def fake_shortest(room_fixed, room_state, max_depth):
        calls.append(max_depth)
        if len(calls) < 3:
            return [], None
        return [1], 2

    monkeypatch.setattr(_sok_mod, "_shortest_first_actions", fake_shortest)
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=5,
        invalid_penalty=-2.0,
    )

    _, info = worker.reset(seed=100)

    assert len(calls) == 3
    assert info["reset_seed"] == 102
    assert info["reset_retry_count"] == 2
    assert info["initial_oracle_found"] is True
    assert info["oracle_valid_actions"] == ["up"]
    assert info["sokoban_shortest_path_len"] == 2
    assert info["sokoban_initial_shortest_path_len"] == 2
    assert info["sokoban_remaining_shortest_path_len"] == 2
    assert info["completion_rate"] == 0.0


def test_worker_reset_stops_after_three_retries_if_still_missing(monkeypatch):
    pytest.importorskip("gym_sokoban")
    calls = []

    def fake_shortest(room_fixed, room_state, max_depth):
        calls.append(max_depth)
        return [], None

    monkeypatch.setattr(_sok_mod, "_shortest_first_actions", fake_shortest)
    worker = _sok_mod.SokobanWorker(
        seed=0,
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=10,
        search_depth=5,
        invalid_penalty=-2.0,
    )

    _, info = worker.reset(seed=100)

    assert len(calls) == 4
    assert info["reset_seed"] == 103
    assert info["reset_retry_count"] == 3
    assert info["initial_oracle_found"] is False
    assert info["oracle_valid_actions"] == []
    assert info["sokoban_shortest_path_len"] is None
    assert info["sokoban_initial_shortest_path_len"] is None
    assert info["sokoban_remaining_shortest_path_len"] is None
    assert info["completion_rate"] == 0.0
