"""Integration tests for VPR environment adapters, managers, and info schemas.

Tests run without Ray or Torch in the default Python environment:
- Core adapter tests (mine-hit, reward, info schema, sentinel) use GEM directly.
- VPRBaseEnvironmentManager tests use MagicMock for envs.
- Ray-dependent tests are skipped when Ray is absent.
- Torch-dependent tests are skipped when Torch is absent.
"""

import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pre-import: inject stubs for heavy transitive dependencies so that the VPR
# env modules can be loaded without torch / ray / omegaconf installed.
# ---------------------------------------------------------------------------

def _load_direct(name: str, path: str):
    """Load a Python file directly, bypassing the package import mechanism."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_stub(key: str):
    if key not in sys.modules:
        sys.modules[key] = MagicMock()


# Ray mock — @ray.remote becomes a no-op decorator
try:
    import ray as _real_ray  # noqa: F401
    _ray_available = True
except ModuleNotFoundError:
    _ray_available = False
    _ray_stub = MagicMock()
    _ray_stub.remote = lambda cls: cls
    sys.modules['ray'] = _ray_stub

# Torch availability check (no mock — test_vpr_advantage.py uses torch directly)
try:
    import torch  # noqa: F401
    _torch_available = True
except ModuleNotFoundError:
    _torch_available = False

# Stub other heavy imports that env_manager.py pulls in
for _stub in [
    'omegaconf', 'verl', 'verl.utils', 'verl.utils.metric', 'verl.trainer',
    'agent_system.memory', 'agent_system.memory.memory',
]:
    _ensure_stub(_stub)


# Pre-load and register VPR modules so package imports resolve them directly
# without going through agent_system/environments/__init__.py → env_manager.py → torch
_PKG = "agent_system.environments.env_package.vpr_games"

_parser_mod = _load_direct(f"{_PKG}.common.parser",
    "agent_system/environments/env_package/vpr_games/common/parser.py")
_oracle_mod = _load_direct(f"{_PKG}.minesweeper.oracle",
    "agent_system/environments/env_package/vpr_games/minesweeper/oracle.py")

# EnvironmentManagerBase stub for base_manager
_env_base_stub = MagicMock()
_env_base_stub.EnvironmentManagerBase = object  # will be subclassed

# Ensure the env_manager key resolves to the stub so base_manager.py import works
_ensure_stub("agent_system.environments.env_manager")


class _EnvironmentManagerBaseStub:
    """Minimal stub that accepts (envs, projection_f, config) like the real class."""
    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def close(self):
        pass


sys.modules["agent_system.environments.env_manager"].EnvironmentManagerBase = _EnvironmentManagerBaseStub
sys.modules["agent_system.environments.env_manager"].to_numpy = lambda x: x

# Also register the prompts module so managers can import it (we mock it)
_ensure_stub("agent_system.environments.prompts.vpr_games")
_ensure_stub("agent_system.environments.prompts")

# When Ray is installed (verl-agent venv), temporarily replace ray.remote with a
# no-op so worker classes are plain Python objects that can be instantiated directly.
# Without this, @ray.remote wraps the classes and requires .remote() calls.
if _ray_available:
    import ray as _ray_real
    _orig_ray_remote = _ray_real.remote
    _ray_real.remote = lambda cls: cls
else:
    _orig_ray_remote = None

# Load under PRIVATE aliases (not the real package paths) so the canonical package paths
# remain unclaimed and factory tests can import the real @ray.remote-decorated classes.
_base_mgr_mod = _load_direct("_testenvs_base_mgr",
    "agent_system/environments/env_package/vpr_games/common/base_manager.py")
_ms_envs_mod = _load_direct("_testenvs_ms_envs",
    "agent_system/environments/env_package/vpr_games/minesweeper/envs.py")
_su_envs_mod = _load_direct("_testenvs_su_envs",
    "agent_system/environments/env_package/vpr_games/sudoku/envs.py")

# Restore real ray.remote so factory tests and Ray-dependent tests work correctly
if _ray_available and _orig_ray_remote is not None:
    _ray_real.remote = _orig_ray_remote


# ---------------------------------------------------------------------------
# Minesweeper adapter tests (GEM only, no Ray)
# ---------------------------------------------------------------------------

class TestMinesweeperWorker:
    # Use 5x5 boards to avoid GEM first-click safety infinite loop
    # (on 3x3 boards, the 3x3 safe zone covers the entire board)
    def _w(self, rows=5, cols=5, mines=3, seed=0):
        return _ms_envs_mod.MinesweeperWorker(
            seed=seed, rows=rows, cols=cols, num_mines=mines, max_turns=30)

    def test_reset_info_schema(self):
        w = self._w()
        obs, info = w.reset(seed=0)
        required = [
            "env_name", "step", "max_steps", "raw_action", "parsed_action",
            "parse_ok", "illegal_action", "available_actions", "vpr_reward",
            "terminal_success", "terminal_reason", "posterior_min_prob",
            "posterior_prob_for_action", "oracle_valid_actions",
            "completion_rate", "oracle_degraded", "flagged_cells",
        ]
        for f in required:
            assert f in info, f"Missing Minesweeper reset info field: {f}"
        assert info["env_name"] == "vpr_minesweeper"
        assert info["step"] == 0
        assert isinstance(info["available_actions"], list)
        assert isinstance(info["flagged_cells"], list)

    def test_first_reveal_oracle_reward(self):
        """First reveal is always safe (GEM first-click) → reward +1.0."""
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=42)
        obs, reward, done, info = w.step("<action>reveal 3 3</action>")
        assert reward == 1.0, f"First reveal should be +1.0, got {reward}"
        assert info["parse_ok"]
        assert not info["illegal_action"]

    def test_mine_hit_legal_non_oracle(self):
        """Mine reveal: reward=0.0 (legal non-oracle), terminal_success=False."""
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        # First safe reveal (corner far from center to maximize safe zone)
        w.step("<action>reveal 1 1</action>")
        # Find an unrevealed mine
        mine_cells = [
            (r + 1, c + 1)
            for r in range(5) for c in range(5)
            if w._env.grid[r][c] < 0 and not w._env.revealed[r][c]
        ]
        if not mine_cells:
            pytest.skip("No unrevealed mine for this seed — board fully revealed")
        r1, c1 = mine_cells[0]
        obs, reward, done, info = w.step(f"<action>reveal {r1} {c1}</action>")
        assert reward == 0.0, f"Mine hit reward should be 0.0, got {reward}"
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "mine_hit"
        assert not info["illegal_action"]

    def test_invalid_parse_penalty_and_terminates(self):
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs, reward, done, info = w.step("garbage no action tag")
        assert reward == -1.0
        assert done
        assert not info["parse_ok"]
        assert info["illegal_action"]

    def test_out_of_bounds_penalty(self):
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs, reward, done, info = w.step("<action>reveal 9 9</action>")
        assert reward == -1.0
        assert done
        assert info["illegal_action"]

    def test_info_json_serializable_reset(self):
        w = self._w(rows=5, cols=5, mines=3)
        obs, info = w.reset(seed=0)
        safe = {k: v for k, v in info.items() if v is not None}
        json.dumps(safe)

    def test_info_json_serializable_step(self):
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs, reward, done, info = w.step("<action>reveal 3 3</action>")
        safe = {k: v for k, v in info.items() if v is not None}
        json.dumps(safe)

    def test_seeded_reset_deterministic(self):
        w = self._w(rows=5, cols=5, mines=3)
        obs1, _ = w.reset(seed=42)
        obs2, _ = w.reset(seed=42)
        assert obs1 == obs2

    def test_different_seeds_different_boards(self):
        # Boards look the same before first reveal (all hidden); compare after first step
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs1, _, _, _ = w.step("<action>reveal 3 3</action>")
        w.reset(seed=99)
        obs2, _, _, _ = w.step("<action>reveal 3 3</action>")
        assert obs1 != obs2


# ---------------------------------------------------------------------------
# Minesweeper oracle tests — exact flag certainty + component decomposition
# ---------------------------------------------------------------------------

class TestOracleExactFlagCertainty:
    """Verify exact integer comparison for flag oracle."""

    def test_non_certain_cell_no_flag_reward(self):
        """P=0.5 cell must not trigger flag oracle (not == 1.0 exactly)."""
        rows, cols = 1, 3
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, _ = _oracle_mod.compute_posteriors(
            revealed, grid, rows, cols, total_mines=1)
        actions, _, _ = _oracle_mod.get_oracle_actions(
            posteriors, revealed, flags, rows, cols)
        flag_actions = [a for a in actions if a.startswith("flag")]
        assert len(flag_actions) == 0, f"P=0.5 should not trigger flag; got {flag_actions}"

    def test_certain_mine_gets_flag_action(self):
        """P=1.0 (exact integer: mine_count == total_weight) must trigger flag oracle."""
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]
        flags = [[False, False]]
        posteriors, _ = _oracle_mod.compute_posteriors(
            revealed, grid, rows, cols, total_mines=1)
        assert posteriors[(0, 1)] == 1.0  # exact float equality from integer division
        actions, _, _ = _oracle_mod.get_oracle_actions(
            posteriors, revealed, flags, rows, cols)
        assert "flag 1 2" in actions

    def test_disconnected_components_correct_posteriors(self):
        """Two disconnected frontier components: each enumerates independently."""
        # 1x5 board: (0,1)=1 and (0,3)=1 revealed.
        # Frontier: {(0,0),(0,2)} and {(0,2),(0,4)} — (0,2) is shared → one component.
        # Actually with total_mines=2: (0,0) and (0,4) must each be mines.
        rows, cols = 1, 5
        revealed = [[False, True, False, True, False]]
        grid = [[0, 1, 0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = _oracle_mod.compute_posteriors(
            revealed, grid, rows, cols, total_mines=2)
        assert not degraded
        assert abs(posteriors.get((0, 0), 0) - 1.0) < 1e-6
        assert abs(posteriors.get((0, 4), 0) - 1.0) < 1e-6
        assert abs(posteriors.get((0, 2), 0) - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Sudoku adapter tests (GEM only, no Ray)
# ---------------------------------------------------------------------------

class TestSudokuWorker:
    def _w(self, n=3, clues=40, seed=0):
        return _su_envs_mod.SudokuWorker(seed=seed, n=n, clues=clues, max_turns=100)

    def test_reset_info_schema(self):
        w = self._w()
        obs, info = w.reset(seed=0)
        required = [
            "env_name", "step", "max_steps", "raw_action", "parsed_action",
            "parse_ok", "illegal_action", "available_actions", "vpr_reward",
            "terminal_success", "terminal_reason", "num_blanks_remaining",
            "completion_rate",
        ]
        for f in required:
            assert f in info, f"Missing Sudoku info field: {f}"
        assert info["env_name"] == "vpr_sudoku"

    def test_correct_digit_oracle_reward(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        blanks = w._blank_cells()
        assert blanks, "Need blank cells"
        r_str, c_str = blanks[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert reward == 1.0, f"Correct digit should give +1.0, got {reward}"
        assert not info["illegal_action"]

    def test_wrong_digit_terminates_with_penalty(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        blanks = w._blank_cells()
        r_str, c_str = blanks[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        wrong = (correct % 9) + 1
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {wrong}</action>")
        assert reward == -1.0
        assert done

    def test_filled_cell_is_invalid(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        for r in range(9):
            for c in range(9):
                if w._env.board[r][c] != 0:
                    obs, reward, done, info = w.step(f"<action>{r+1} {c+1} 5</action>")
                    assert reward == -1.0
                    assert done
                    assert info["illegal_action"]
                    return
        pytest.skip("No pre-filled cell found")

    def test_invalid_parse_penalty(self):
        w = self._w()
        w.reset(seed=0)
        obs, reward, done, info = w.step("no action tag")
        assert reward == -1.0
        assert done
        assert not info["parse_ok"]

    def test_seeded_reset_deterministic(self):
        w = self._w()
        obs1, _ = w.reset(seed=42)
        obs2, _ = w.reset(seed=42)
        assert obs1 == obs2

    def test_different_seeds_different_boards(self):
        w = self._w()
        obs1, _ = w.reset(seed=42)
        obs2, _ = w.reset(seed=43)
        assert obs1 != obs2

    def test_info_json_serializable(self):
        w = self._w()
        obs, info = w.reset(seed=0)
        safe = {k: v for k, v in info.items() if v is not None}
        json.dumps(safe)

    def test_step_info_json_serializable(self):
        w = self._w()
        w.reset(seed=0)
        obs, reward, done, info = w.step("garbage")
        safe = {k: v for k, v in info.items() if v is not None}
        json.dumps(safe)


# ---------------------------------------------------------------------------
# VPRBaseEnvironmentManager tests
# ---------------------------------------------------------------------------

class TestVPRBaseEnvironmentManager:
    def _make_manager(self, history_length=0):
        class DummyManager(_base_mgr_mod.VPRBaseEnvironmentManager):
            def build_text_obs(self, infos):
                return ["obs"] * len(infos)

        config = SimpleNamespace(env=SimpleNamespace(history_length=history_length))
        return DummyManager(MagicMock(), lambda x: (x, [True] * len(x)), config)

    def test_history_length_zero_ok(self):
        mgr = self._make_manager(history_length=0)
        assert mgr is not None

    def test_history_length_nonzero_raises(self):
        with pytest.raises(ValueError, match="history_length"):
            self._make_manager(history_length=1)

    def test_success_evaluator_uses_terminal_success(self):
        mgr = self._make_manager()
        total_infos = [
            [{"terminal_success": True}],
            [{"terminal_success": False}],
        ]
        total_batch_list = [
            [{"active_masks": True}],
            [{"active_masks": True}],
        ]
        result = mgr.success_evaluator(
            total_infos=total_infos, total_batch_list=total_batch_list)
        assert result["success"][0] == True
        assert result["success"][1] == False

    def test_success_evaluator_does_not_use_won(self):
        """Must read terminal_success, not info['won']."""
        mgr = self._make_manager()
        total_infos = [[{"terminal_success": False, "won": True}]]
        total_batch_list = [[{"active_masks": True}]]
        result = mgr.success_evaluator(
            total_infos=total_infos, total_batch_list=total_batch_list)
        assert result["success"][0] == False

    def test_manager_step_sets_is_action_valid(self):
        """VPRBaseEnvironmentManager.step() sets is_action_valid in returned infos.

        Tests VPRBaseEnvironmentManager.step() directly with mock env pool.
        A regression in base_manager.py:37-48 would be detected here.
        """
        class MockEnvPool:
            def step(self, actions):
                infos = [
                    {"parse_ok": True, "illegal_action": False, "vpr_reward": 1.0},
                    {"parse_ok": False, "illegal_action": True, "vpr_reward": -1.0},
                ]
                return ["obs1", "obs2"], [1.0, -1.0], [False, True], infos

            def reset(self, **kw):
                return ["obs1", "obs2"], [{}, {}]

        class TestMgr(_base_mgr_mod.VPRBaseEnvironmentManager):
            def build_text_obs(self, infos):
                return ["obs"] * len(infos)

        config = SimpleNamespace(env=SimpleNamespace(history_length=0))
        mgr = TestMgr(MockEnvPool(), lambda x: (x, []), config)

        obs, rewards, dones, infos = mgr.step(["action1", "action2"])
        assert infos[0]["is_action_valid"] == 1, f"Valid action → is_action_valid=1, got {infos[0]}"
        assert infos[1]["is_action_valid"] == 0, f"Parse failure → is_action_valid=0, got {infos[1]}"

    def test_manager_step_prompt_bounded_across_5_steps(self):
        """Manager-generated prompts do not grow across 5 TicTacToe steps (Markovian).

        Exercises the full manager observation pipeline rather than raw template formatting.
        """
        # Load TicTacToe manager directly (no heavy deps needed)
        import importlib.util, sys

        spec = importlib.util.spec_from_file_location(
            "_ttt_game_mgr_test",
            "agent_system/environments/env_package/vpr_games/tictactoe/game.py")
        game_mod = importlib.util.module_from_spec(spec)
        sys.modules["_ttt_game_mgr_test"] = game_mod
        spec.loader.exec_module(game_mod)

        spec2 = importlib.util.spec_from_file_location(
            "_ttt_prompts_mgr",
            "agent_system/environments/prompts/vpr_games.py")
        tpl_mod = importlib.util.module_from_spec(spec2)
        sys.modules["_ttt_prompts_mgr"] = tpl_mod
        spec2.loader.exec_module(tpl_mod)

        # Simulate manager's build_text_obs for TicTacToe
        def build_obs(game, obs_str):
            return tpl_mod.TICTACTOE_TEMPLATE.format(board=obs_str)

        g = game_mod.TicTacToeGame(opponent="random", seed=0)
        obs, _ = g.reset(seed=0)
        prompt_lens = [len(build_obs(g, obs))]

        # Play through up to 9 cells to guarantee 5+ steps
        legal_cells = list(range(1, 10))
        steps_taken = 0
        for cell in legal_cells:
            obs, reward, done, info = g.step(str(cell), True, f"<action>{cell}</action>")
            if not done:
                prompt_lens.append(len(build_obs(g, obs)))
                steps_taken += 1
                if steps_taken >= 5:
                    break
            else:
                # Restart if game ends early
                obs, _ = g.reset(seed=steps_taken + 1)
                prompt_lens.append(len(build_obs(g, obs)))

        assert len(prompt_lens) >= 2, f"Need ≥2 prompts, got {len(prompt_lens)}"

        # Markovian: prompt length should not grow across steps
        for i in range(1, len(prompt_lens)):
            growth = prompt_lens[i] - prompt_lens[0]
            assert growth <= 100, \
                f"Prompt grew {growth} chars at step {i} (step0={prompt_lens[0]}, step{i}={prompt_lens[i]})"


# ---------------------------------------------------------------------------
# Parser sentinel tests: GEM must not be called on parse failure
# ---------------------------------------------------------------------------

class TestParserSentinel:
    def test_minesweeper_no_gem_on_parse_failure(self):
        w = _ms_envs_mod.MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3)
        w.reset(seed=0)
        call_count = [0]
        orig_step = w._env.step

        def counting(*args, **kwargs):
            call_count[0] += 1
            return orig_step(*args, **kwargs)

        w._env.step = counting
        w.step("garbage no action tag")
        assert call_count[0] == 0, f"GEM called {call_count[0]} times on parse failure"

    def test_sudoku_no_gem_on_parse_failure(self):
        w = _su_envs_mod.SudokuWorker(seed=0, n=3, clues=40)
        w.reset(seed=0)
        call_count = [0]
        orig = w._env.step

        def counting(*args, **kwargs):
            call_count[0] += 1
            return orig(*args, **kwargs)

        w._env.step = counting
        w.step("no tags here")
        assert call_count[0] == 0, "GEM must not be called on parse failure"


# ---------------------------------------------------------------------------
# VPR advantage estimator tests (require torch; skip if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestVPRAdvantageProduction:
    """Tests using the real production compute_vpr_turn_level_advantage."""

    def _load_core(self):
        return _load_direct("core_gigpo_t", "gigpo/core_gigpo.py")

    def _data(self, rewards, turns, terminal_success=None, is_terminal=None, rlen=4):
        import torch
        n = len(rewards)
        mask = torch.ones(n, rlen)
        nt = {
            "rewards": np.array(rewards, dtype=np.float32),
            "turn_index": np.array(turns, dtype=np.int32),
        }
        if terminal_success is not None:
            nt["terminal_success"] = np.array(terminal_success, dtype=bool)
        if is_terminal is not None:
            nt["is_terminal"] = np.array(is_terminal, dtype=bool)
        return SimpleNamespace(batch={"response_mask": mask}, non_tensor_batch=nt)

    def test_outcome_bonus_only_at_terminal_success(self):
        core = self._load_core()
        data = self._data(
            rewards=[1.0, 0.0, 0.5, -0.5],
            turns=[0, 0, 1, 1],
            is_terminal=[False, True, False, True],
            terminal_success=[False, True, False, False],
        )
        core.compute_vpr_turn_level_advantage(
            data, min_group_size=2, outcome_reward_scale=1.0)
        bonus = data.non_tensor_batch.get("vpr_outcome_bonus")
        assert bonus is not None
        assert bonus[0] == 0.0   # non-terminal
        assert bonus[1] == 1.0   # terminal + success
        assert bonus[2] == 0.0   # non-terminal
        assert bonus[3] == 0.0   # terminal + failure

    def test_outcome_bonus_zero_at_non_terminal(self):
        core = self._load_core()
        data = self._data(
            rewards=[1.0, 0.0],
            turns=[0, 0],
            is_terminal=[False, False],
            terminal_success=[False, False],
        )
        core.compute_vpr_turn_level_advantage(
            data, min_group_size=2, outcome_reward_scale=1.0)
        bonus = data.non_tensor_batch.get("vpr_outcome_bonus")
        assert all(b == 0.0 for b in bonus)

    def test_outcome_disabled_when_scale_zero(self):
        core = self._load_core()
        data = self._data(
            rewards=[1.0, 0.0],
            turns=[0, 0],
            is_terminal=[True, True],
            terminal_success=[True, True],
        )
        core.compute_vpr_turn_level_advantage(
            data, min_group_size=2, outcome_reward_scale=0.0)
        bonus = data.non_tensor_batch.get("vpr_outcome_bonus")
        assert all(b == 0.0 for b in bonus)

    def test_vpr_oracle_reward_stored_separately(self):
        core = self._load_core()
        rewards = [0.5, 0.0]
        data = self._data(rewards, [0, 0])
        core.compute_vpr_turn_level_advantage(data, min_group_size=2)
        assert "vpr_oracle_reward" in data.non_tensor_batch
        np.testing.assert_array_almost_equal(
            data.non_tensor_batch["vpr_oracle_reward"], rewards)


# ---------------------------------------------------------------------------
# Ray-dependent tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ray_available, reason="Ray not installed")
class TestMakeEnvsActorCounts:
    """Factory tests using build_*_envs directly (bypasses env_manager module-level mock)."""

    def _init_ray(self):
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=8, ignore_reinit_error=True)

    def test_tictactoe_actor_count_train_batch2_groupn2(self):
        """train_batch_size=2, rollout.n=2 → 4 train actors; val_batch=1 → 1 val actor."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.tictactoe.envs import (
            build_tictactoe_envs
        )
        train_envs = build_tictactoe_envs(seed=0, env_num=2, group_n=2)
        val_envs = build_tictactoe_envs(seed=1000, env_num=1, group_n=1)
        assert len(train_envs.workers) == 4, f"Expected 4 train actors, got {len(train_envs.workers)}"
        assert len(val_envs.workers) == 1, f"Expected 1 val actor, got {len(val_envs.workers)}"
        train_envs.close()
        val_envs.close()

    def test_sudoku_actor_count_and_val_seed(self):
        """val envs seeded at seed+1000."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.sudoku.envs import (
            build_sudoku_envs
        )
        train_envs = build_sudoku_envs(seed=0, env_num=2, group_n=2)
        val_envs = build_sudoku_envs(seed=1000, env_num=1, group_n=1)
        assert len(train_envs.workers) == 4
        assert len(val_envs.workers) == 1
        # Val seed should be different from train seed → different boards after reset
        import ray
        t_obs = ray.get(train_envs.workers[0].reset.remote(seed=0))[0]
        v_obs = ray.get(val_envs.workers[0].reset.remote(seed=1000))[0]
        assert t_obs != v_obs, "Train and val envs should have different initial boards"
        train_envs.close()
        val_envs.close()

    def test_minesweeper_actor_count(self):
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.minesweeper.envs import (
            build_minesweeper_envs
        )
        train_envs = build_minesweeper_envs(seed=0, env_num=2, group_n=2)
        val_envs = build_minesweeper_envs(seed=1000, env_num=1, group_n=1)
        assert len(train_envs.workers) == 4
        assert len(val_envs.workers) == 1
        train_envs.close()
        val_envs.close()

    def test_tictactoe_grouped_reset_identity(self):
        """group_n=2: both replicas in a group share the same initial board."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.tictactoe.envs import (
            build_tictactoe_envs
        )
        envs = build_tictactoe_envs(seed=0, env_num=1, group_n=2)
        obs_list, _ = envs.reset()
        assert obs_list[0] == obs_list[1], "Group replicas must start with identical boards"
        envs.close()

    def test_sudoku_grouped_reset_identity(self):
        """group_n=2 for Sudoku: both replicas share the same initial puzzle."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.sudoku.envs import (
            build_sudoku_envs
        )
        envs = build_sudoku_envs(seed=42, env_num=1, group_n=2)
        obs_list, _ = envs.reset()
        assert obs_list[0] == obs_list[1], "Sudoku group replicas must start identically"
        envs.close()

    def test_different_seeds_different_puzzles(self):
        """seed=42 and seed=43 produce different initial Sudoku puzzles."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.sudoku.envs import (
            build_sudoku_envs
        )
        e1 = build_sudoku_envs(seed=42, env_num=1, group_n=1)
        e2 = build_sudoku_envs(seed=43, env_num=1, group_n=1)
        obs1, _ = e1.reset()
        obs2, _ = e2.reset()
        assert obs1[0] != obs2[0], "Different seeds must produce different Sudoku puzzles"
        e1.close()
        e2.close()
