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

# Stub optional imports only when real modules are unavailable. Avoid
# polluting sys.modules for other tests in the production training venv.
try:
    importlib.import_module('omegaconf')
except ImportError:
    _ensure_stub('omegaconf')
for _stub in ['agent_system.memory', 'agent_system.memory.memory']:
    _ensure_stub(_stub)
try:
    importlib.import_module('verl.utils.metric')
    importlib.import_module('verl.trainer')
except ImportError:
    for _stub in ['verl', 'verl.utils', 'verl.utils.metric', 'verl.trainer']:
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
_prompt_stub = sys.modules["agent_system.environments.prompts.vpr_games"]
_prompt_stub.get_vpr_game_template = lambda game, action_format="action_tag": getattr(
    _prompt_stub,
    f"{game.upper()}_TEMPLATE",
)

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
sys.modules[f"{_PKG}.common.base_manager"] = _base_mgr_mod
sys.modules["agent_system.environments.prompts.vpr_games"].MINESWEEPER_TEMPLATE = "{board}\n{unrevealed_cells}\n{flagged_cells}"
_ms_mgr_mod = _load_direct("_testenvs_ms_manager",
    "agent_system/environments/env_package/vpr_games/minesweeper/manager.py")
sys.modules["agent_system.environments.prompts.vpr_games"].SUDOKU_TEMPLATE = "{grid}\n{blank_cells}"
_su_mgr_mod = _load_direct("_testenvs_su_manager",
    "agent_system/environments/env_package/vpr_games/sudoku/manager.py")
sys.modules.pop(f"{_PKG}.common.base_manager", None)
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
            "move_optimal", "pre_exec_oracle_match", "legal_non_oracle",
            "oracle_action_set_size", "oracle_guess", "oracle_policy",
            "oracle_policy_tier", "oracle_tier", "safe_reveal_available",
            "certain_flag_available", "guess_required", "reveal_posterior_margin",
        ]
        for f in required:
            assert f in info, f"Missing Minesweeper reset info field: {f}"
        assert info["env_name"] == "vpr_minesweeper"
        assert info["step"] == 0
        assert isinstance(info["available_actions"], list)
        assert isinstance(info["flagged_cells"], list)

    def test_first_reveal_oracle_reward(self):
        """First reveal is always safe (GEM first-click) → safe reveal reward +2.0."""
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=42)
        obs, reward, done, info = w.step("<action>reveal 3 3</action>")
        assert reward == 2.0, f"First reveal should be +2.0, got {reward}"
        assert info["parse_ok"]
        assert not info["illegal_action"]

    def test_boxed_action_protocol_executes_action(self):
        w = _ms_envs_mod.MinesweeperWorker(
            seed=0,
            rows=5,
            cols=5,
            num_mines=3,
            max_turns=30,
            action_format="boxed",
        )
        w.reset(seed=42)
        _, _, _, info = w.step(r"\boxed{reveal 3 3}")
        assert info["parse_ok"] and info["parsed_action"] == "reveal 3 3"

    def test_mine_hit_uses_pre_exec_oracle_label(self):
        """Mine reveal uses the pre-execution oracle label, terminal_success=False."""
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
        if info["pre_exec_oracle_match"]:
            expected = {"safe_reveal": 2.0, "certain_flag": 2.0, "guess": 1.0}[info["oracle_tier"]]
        else:
            expected = 0.0
        assert reward == expected, f"Mine hit reward should use pre-exec label, got {reward}"
        assert info["move_optimal"] is info["pre_exec_oracle_match"]
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "mine_hit"
        assert not info["illegal_action"]

    def test_invalid_parse_penalty_and_terminates(self):
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs, reward, done, info = w.step("garbage no action tag")
        assert reward == -2.0
        assert done
        assert not info["parse_ok"]
        assert info["illegal_action"]

    def test_out_of_bounds_penalty(self):
        w = self._w(rows=5, cols=5, mines=3)
        w.reset(seed=0)
        obs, reward, done, info = w.step("<action>reveal 9 9</action>")
        assert reward == -2.0
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


class TestMinesweeperEnvironmentManager:
    def test_trajectory_metrics_reports_policy_diagnostics(self):
        mgr = _ms_mgr_mod.MinesweeperEnvironmentManager.__new__(_ms_mgr_mod.MinesweeperEnvironmentManager)
        metrics = mgr._trajectory_metrics([
            {
                "completion_rate": 0.5,
                "terminal_reason": None,
                "move_optimal": True,
                "pre_exec_oracle_match": True,
                "legal_non_oracle": False,
                "parsed_action": "reveal 1 2",
                "oracle_action_set_size": 1,
                "oracle_policy_tier": "all_oracle_actions",
                "oracle_tier": "safe_reveal",
                "safe_reveal_available": True,
                "certain_flag_available": True,
                "guess_required": False,
                "oracle_guess": False,
                "posterior_prob_for_action": 0.0,
                "posterior_min_prob": 0.0,
                "reveal_posterior_margin": 0.0,
            },
            {
                "completion_rate": 0.5,
                "terminal_reason": None,
                "move_optimal": True,
                "pre_exec_oracle_match": True,
                "legal_non_oracle": False,
                "parsed_action": "flag 1 3",
                "oracle_action_set_size": 3,
                "oracle_policy_tier": "all_oracle_actions",
                "oracle_tier": "certain_flag",
                "safe_reveal_available": False,
                "certain_flag_available": True,
                "guess_required": False,
                "oracle_guess": False,
                "posterior_prob_for_action": 1.0,
                "posterior_min_prob": 1.0,
            },
            {
                "completion_rate": 0.5,
                "terminal_reason": "mine_hit",
                "move_optimal": False,
                "pre_exec_oracle_match": False,
                "legal_non_oracle": True,
                "parsed_action": "reveal 1 4",
                "oracle_action_set_size": 5,
                "oracle_policy_tier": "all_oracle_actions",
                "oracle_tier": None,
                "safe_reveal_available": False,
                "certain_flag_available": False,
                "guess_required": True,
                "oracle_guess": False,
                "posterior_prob_for_action": 0.6,
                "posterior_min_prob": 0.2,
                "reveal_posterior_margin": 0.4,
            },
        ])

        assert metrics["env/completion_rate"] == 0.5
        assert metrics["env/mine_hit_rate"] == 1.0
        assert metrics["env/pre_exec_oracle_match_rate"] == 2 / 3
        assert metrics["env/safe_reveal_available_rate"] == 1 / 3
        assert metrics["env/certain_flag_available_rate"] == 2 / 3
        assert metrics["env/guess_required_rate"] == 1 / 3
        assert metrics["env/safe_reveal_hit_rate"] == 1.0
        assert metrics["env/certain_flag_hit_rate"] == 0.5
        assert metrics["env/guess_hit_rate"] == 0.0
        assert metrics["env/non_oracle_reveal_rate"] == 1 / 3
        assert metrics["env/non_oracle_flag_rate"] == 0.0
        assert abs(metrics["env/reveal_posterior_margin_mean"] - 0.2) < 1e-9
        assert abs(metrics["env/action_posterior_mean"] - (1.6 / 3)) < 1e-9
        assert abs(metrics["env/min_posterior_mean"] - (1.2 / 3)) < 1e-9
        assert metrics["env/oracle_hit_rate_action_set_size_1"] == 1.0
        assert metrics["env/oracle_hit_rate_action_set_size_2_4"] == 1.0
        assert metrics["env/oracle_hit_rate_action_set_size_gt4"] == 0.0
        assert metrics["env/non_oracle_mine_hit_rate"] == 1.0


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
    def _w(self, n=3, clues=40, seed=0, max_turns=100,
           terminate_on_wrong_digit=True):
        return _su_envs_mod.SudokuWorker(
            seed=seed, n=n, clues=clues, max_turns=max_turns,
            terminate_on_wrong_digit=terminate_on_wrong_digit)

    def test_reset_info_schema(self):
        w = self._w()
        obs, info = w.reset(seed=0)
        required = [
            "env_name", "step", "max_steps", "raw_action", "parsed_action",
            "parse_ok", "illegal_action", "available_actions", "vpr_reward",
            "terminal_success", "terminal_reason", "initial_blank_count",
            "num_blanks_remaining", "completion_rate", "move_optimal",
            "correct_fills", "outcome_success_correct_fills",
            "outcome_target_completion_rate",
            "pre_exec_oracle_match", "legal_non_oracle",
            "sudoku_mrv_min_candidates", "sudoku_candidate_count_for_action",
            "sudoku_forced_cell_available", "sudoku_action_is_mrv_cell",
            "sudoku_oracle_tier", "oracle_action_set_size",
        ]
        for f in required:
            assert f in info, f"Missing Sudoku info field: {f}"
        assert info["env_name"] == "vpr_sudoku"

    def _set_ambiguous_mrv_board(self, w):
        w.reset(seed=0)
        board = [row[:] for row in w._env.full_grid]
        blanks = [(0, 0), (0, 1), (3, 0), (3, 1)]
        for r, c in blanks:
            board[r][c] = 0
        w._env.board = board
        w._env.init_num_empty = len(blanks)
        w._step_count = 0
        w._done = False
        return blanks

    def test_forced_cell_oracle_reward_is_two(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        state = w._mrv_oracle_state()
        assert state["forced_available"], "Need a forced cell in the generated puzzle"
        r, c = sorted(state["forced_cells"])[0]
        correct = w._env.full_grid[r][c]
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert reward == 2.0, f"Forced-cell oracle should give +2.0, got {reward}"
        assert info["move_optimal"] is True
        assert info["pre_exec_oracle_match"] is True
        assert info["sudoku_oracle_tier"] == "forced"
        assert info["sudoku_candidate_count_for_action"] == 1
        assert info["sudoku_mrv_min_candidates"] == 1
        assert info["sudoku_forced_cell_available"] is True
        assert info["sudoku_action_is_mrv_cell"] is True
        assert info["oracle_action_set_size"] == len(state["forced_cells"])
        assert not info["illegal_action"]

    def test_boxed_action_protocol_executes_action(self):
        w = _su_envs_mod.SudokuWorker(
            seed=0,
            n=3,
            clues=40,
            max_turns=100,
            action_format="boxed",
        )
        w.reset(seed=42)
        row, col = next(
            (row, col)
            for row in range(9)
            for col in range(9)
            if w._env.board[row][col] == 0
        )
        digit = w._env.full_grid[row][col]
        _, _, _, info = w.step(fr"\boxed{{{row + 1} {col + 1} {digit}}}")
        assert info["parse_ok"]
        assert info["parsed_action"] == f"{row + 1} {col + 1} {digit}"

    def test_mrv_gt_one_correct_digit_reward_is_two(self):
        w = self._w(n=3, clues=40)
        self._set_ambiguous_mrv_board(w)
        state = w._mrv_oracle_state()
        assert not state["forced_available"]
        assert state["min_candidates"] == 2
        r, c = sorted(state["mrv_cells"])[0]
        correct = w._env.full_grid[r][c]
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert reward == 1.0, f"MRV>1 oracle should give +1.0, got {reward}"
        assert info["move_optimal"] is True
        assert info["sudoku_oracle_tier"] == "mrv"
        assert info["sudoku_mrv_min_candidates"] == 2
        assert info["sudoku_candidate_count_for_action"] == 2
        assert info["sudoku_forced_cell_available"] is False
        assert info["sudoku_action_is_mrv_cell"] is True
        assert info["oracle_action_set_size"] == len(state["mrv_cells"])

    def test_correct_digit_non_oracle_gets_partial_reward_without_wrong_digit_terminal(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        state = w._mrv_oracle_state()
        non_oracle_cells = sorted(set(state["candidate_counts"]) - set(state["oracle_cells"]))
        assert non_oracle_cells, "Need a non-oracle blank cell in the generated puzzle"
        r, c = non_oracle_cells[0]
        correct = w._env.full_grid[r][c]
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert reward == 0.5
        assert not done
        assert info["move_optimal"] is False
        assert info["pre_exec_oracle_match"] is False
        assert info["legal_non_oracle"] is True
        assert info["sudoku_oracle_tier"] is None

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

    def test_wrong_digit_can_continue_when_configured(self):
        w = self._w(n=3, clues=40, terminate_on_wrong_digit=False)
        w.reset(seed=42)
        blanks = w._blank_cells()
        r_str, c_str = blanks[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        wrong = (correct % 9) + 1
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {wrong}</action>")
        assert reward == -1.0
        assert not done
        assert info["terminal_reason"] is None
        assert info["move_optimal"] is False
        assert info["pre_exec_oracle_match"] is False
        assert info["legal_non_oracle"] is True
        assert info["num_blanks_remaining"] == info["initial_blank_count"]

    def test_candidate_group_commits_best_reward_without_candidate_mutation_leakage(self):
        w = self._w(n=3, clues=40, terminate_on_wrong_digit=False)
        w.reset(seed=42)
        state = w._mrv_oracle_state()
        mrv_cell = sorted(state["mrv_cells"])[0]
        mrv_r, mrv_c = mrv_cell
        mrv_correct = w._env.full_grid[mrv_r][mrv_c]
        wrong = (mrv_correct % 9) + 1

        non_mrv_cells = sorted(set(state["candidate_counts"]) - set(state["mrv_cells"]))
        if not non_mrv_cells:
            pytest.skip("Need a non-MRV blank cell")
        non_r, non_c = non_mrv_cells[0]
        non_correct = w._env.full_grid[non_r][non_c]

        candidates, best_idx, obs, reward, done, info = w.step_candidate_group([
            f"<action>{mrv_r+1} {mrv_c+1} {wrong}</action>",
            f"<action>{non_r+1} {non_c+1} {non_correct}</action>",
            f"<action>{mrv_r+1} {mrv_c+1} {mrv_correct}</action>",
        ])

        assert best_idx == 2
        expected_reward = 2.0 if info["sudoku_oracle_tier"] == "forced" else 1.0
        assert reward == expected_reward
        assert info["move_optimal"] is True
        assert w._env.board[mrv_r][mrv_c] == mrv_correct
        assert w._env.board[non_r][non_c] == 0, "Rejected non-MRV candidate must not mutate board"
        assert w._step_count == 1
        assert candidates[0][1] == -1.0
        assert candidates[1][1] == 0.5
        assert candidates[2][1] == 2.0

    def test_non_oracle_timeout_reason_when_wrong_digit_does_not_terminate(self):
        w = self._w(n=3, clues=40, max_turns=1, terminate_on_wrong_digit=False)
        w.reset(seed=42)
        blanks = w._blank_cells()
        r_str, c_str = blanks[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        wrong = (correct % 9) + 1
        obs, reward, done, info = w.step(f"<action>{r+1} {c+1} {wrong}</action>")
        assert reward == -2.0
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "timeout"
        assert info["legal_non_oracle"] is True

    def test_filled_cell_is_invalid(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        for r in range(9):
            for c in range(9):
                if w._env.board[r][c] != 0:
                    obs, reward, done, info = w.step(f"<action>{r+1} {c+1} 5</action>")
                    assert reward == -2.0
                    assert done
                    assert info["illegal_action"]
                    return
        pytest.skip("No pre-filled cell found")

    def test_invalid_parse_penalty(self):
        w = self._w()
        w.reset(seed=0)
        obs, reward, done, info = w.step("no action tag")
        assert reward == -2.0
        assert done
        assert not info["parse_ok"]

    def test_out_of_range_penalty(self):
        w = self._w(n=3, clues=40)
        w.reset(seed=42)
        obs, reward, done, info = w.step("<action>10 1 1</action>")
        assert reward == -2.0
        assert done
        assert info["illegal_action"]
        assert info["terminal_reason"] == "out_of_range"

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

    def test_reset_exact_blank_count_across_seeds(self):
        """clues=40 must yield exactly 40 blanks for every seed. Seeds 4 and 8 are
        regression anchors: raw GEM generation abandons uniqueness-breaking removals
        and returns 39 / 38 blanks, so the adapter must retry to hit the fixed default."""
        w = self._w(n=3, clues=40)
        for seed in [0, 4, 8, 42, 1004, 1008]:
            obs, info = w.reset(seed=seed)
            assert info["num_blanks_remaining"] == 40, (
                f"seed={seed} produced {info['num_blanks_remaining']} blanks, expected 40")

    def test_reset_same_seed_identical_board_hard_seed(self):
        """Same seed → identical board even when retries were required (seed 4)."""
        w = self._w(n=3, clues=40)
        o1, i1 = w.reset(seed=4)
        o2, i2 = w.reset(seed=4)
        assert o1 == o2
        assert i1["num_blanks_remaining"] == 40 and i2["num_blanks_remaining"] == 40

    def test_reset_unreachable_blank_count_raises(self):
        """An impossible target blank count exhausts the budget and raises ValueError
        rather than silently returning a wrong-sized board."""
        w = self._w(n=3, clues=40)
        w._target_blanks = 999  # impossible on a 9x9 board (max 81)
        w._max_generation_attempts = 5
        with pytest.raises(ValueError):
            w.reset(seed=0)


class TestSudokuEnvironmentManager:
    def test_trajectory_metrics_reports_diagnostics(self):
        mgr = _su_mgr_mod.SudokuEnvironmentManager.__new__(_su_mgr_mod.SudokuEnvironmentManager)
        metrics = mgr._trajectory_metrics([
            {
                "completion_rate": 0.25,
                "num_blanks_remaining": 30,
                "initial_blank_count": 40,
                "move_optimal": True,
                "pre_exec_oracle_match": True,
                "legal_non_oracle": False,
                "sudoku_forced_cell_available": True,
                "sudoku_action_is_mrv_cell": True,
                "sudoku_oracle_tier": "forced",
                "sudoku_mrv_min_candidates": 1,
                "oracle_action_set_size": 3,
                "parse_ok": True,
                "illegal_action": False,
                "terminal_reason": None,
            },
            {
                "completion_rate": 0.25,
                "num_blanks_remaining": 30,
                "initial_blank_count": 40,
                "move_optimal": False,
                "pre_exec_oracle_match": False,
                "legal_non_oracle": True,
                "sudoku_forced_cell_available": False,
                "sudoku_action_is_mrv_cell": True,
                "sudoku_oracle_tier": None,
                "sudoku_mrv_min_candidates": 2,
                "oracle_action_set_size": 4,
                "parse_ok": True,
                "illegal_action": False,
                "terminal_reason": "timeout",
            },
            {
                "completion_rate": 0.25,
                "num_blanks_remaining": 30,
                "initial_blank_count": 40,
                "move_optimal": None,
                "parse_ok": True,
                "illegal_action": False,
                "terminal_reason": "already_done",
            },
        ])

        assert metrics["env/completion_rate"] == 0.25
        assert metrics["env/num_blanks_remaining"] == 30.0
        assert metrics["env/blanks_remaining_rate"] == 0.75
        assert metrics["env/pre_exec_oracle_match_rate"] == 0.5
        assert metrics["env/legal_non_oracle_rate"] == 0.5
        assert metrics["env/sudoku_forced_cell_available_rate"] == 0.5
        assert metrics["env/sudoku_mrv_action_rate"] == 1.0
        assert metrics["env/sudoku_forced_oracle_rate"] == 0.5
        assert metrics["env/sudoku_mrv_oracle_rate"] == 0.0
        assert metrics["env/sudoku_mrv_min_candidates_mean"] == 1.5
        assert metrics["env/sudoku_oracle_action_set_size_mean"] == 3.5
        assert metrics["env/parse_error_rate"] == 0.0
        assert metrics["env/illegal_action_rate"] == 0.0
        assert metrics["env/terminal_timeout_rate"] == 1.0
        assert metrics["env/terminal_complete_rate"] == 0.0

    def test_trajectory_metrics_reports_partial_outcome_target(self):
        mgr = _su_mgr_mod.SudokuEnvironmentManager.__new__(
            _su_mgr_mod.SudokuEnvironmentManager
        )
        metrics = mgr._trajectory_metrics([
            {
                "completion_rate": 0.3,
                "num_blanks_remaining": 28,
                "initial_blank_count": 40,
                "correct_fills": 12,
                "outcome_success_correct_fills": 12,
                "outcome_target_completion_rate": 1.0,
                "move_optimal": True,
                "parse_ok": True,
                "illegal_action": False,
                "terminal_reason": "outcome_target",
            }
        ])

        assert metrics["env/completion_rate"] == 0.3
        assert metrics["env/sudoku_correct_fills"] == 12.0
        assert metrics["env/sudoku_outcome_target"] == 12.0
        assert metrics["env/sudoku_outcome_target_completion_rate"] == 1.0
        assert metrics["env/terminal_outcome_target_rate"] == 1.0
        assert metrics["env/terminal_complete_rate"] == 0.0
        assert metrics["env/terminal_timeout_rate"] == 0.0


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
        assert result["env/success_rate"][0] == 1.0
        assert result["env/success_rate"][1] == 0.0

    def test_success_evaluator_does_not_use_won(self):
        """Must read terminal_success, not info['won']."""
        mgr = self._make_manager()
        total_infos = [[{"terminal_success": False, "won": True}]]
        total_batch_list = [[{"active_masks": True}]]
        result = mgr.success_evaluator(
            total_infos=total_infos, total_batch_list=total_batch_list)
        assert result["env/success_rate"][0] == 0.0

    def test_success_evaluator_reports_valid_and_oracle_rates(self):
        """Common env metrics: valid-action rate and oracle-hit rate (over measurable moves)."""
        mgr = self._make_manager()
        total_infos = [
            # 2 steps: one valid+optimal move, one valid+suboptimal move
            [
                {"terminal_success": None, "is_action_valid": 1, "move_optimal": True},
                {"terminal_success": True, "is_action_valid": 1, "move_optimal": False},
            ],
            # 1 invalid step with no measurable oracle move
            [{"terminal_success": False, "is_action_valid": 0, "move_optimal": None}],
        ]
        total_batch_list = [[{"active_masks": True}], [{"active_masks": True}]]
        result = mgr.success_evaluator(
            total_infos=total_infos, total_batch_list=total_batch_list)
        for key in ("env/success_rate", "env/valid_action_rate", "env/oracle_hit_rate"):
            assert key in result and len(result[key]) == 2
        assert result["env/valid_action_rate"][0] == 1.0
        assert result["env/valid_action_rate"][1] == 0.0
        assert result["env/oracle_hit_rate"][0] == 0.5   # 1 of 2 measurable moves optimal
        assert result["env/oracle_hit_rate"][1] == 0.0    # no measurable moves → 0.0

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

    def test_manager_step_prompt_bounded_and_markovian(self):
        """TicTacToeEnvironmentManager.step() prompts are Markovian: bounded and history-free.

        Constructs real TicTacToeEnvironmentManager with a mock env pool that simulates
        a non-terminating 9-step TicTacToe game. Verifies over >= 5 sequential manager
        steps (without resets) that:
        1. Prompt length does not grow by > 100 chars (Markovian: no appended history)
        2. The action text from step t does NOT appear in the prompt at step t+1 (no history leak)
        """
        import importlib.util, sys

        # Load game module
        spec = importlib.util.spec_from_file_location(
            "_ttt_game_mgr_real",
            "agent_system/environments/env_package/vpr_games/tictactoe/game.py")
        game_mod = importlib.util.module_from_spec(spec)
        sys.modules["_ttt_game_mgr_real"] = game_mod
        spec.loader.exec_module(game_mod)

        # Load manager module (base_manager is already loaded as _testenvs_base_mgr)
        spec2 = importlib.util.spec_from_file_location(
            "_ttt_mgr_real",
            "agent_system/environments/env_package/vpr_games/tictactoe/manager.py")
        mgr_mod = importlib.util.module_from_spec(spec2)
        sys.modules["_ttt_mgr_real"] = mgr_mod
        spec2.loader.exec_module(mgr_mod)

        TicTacToeEnvironmentManager = mgr_mod.TicTacToeEnvironmentManager

        # Stateful mock pool: wraps a real TicTacToeGame but never terminates early
        # (by playing only valid non-winning moves)
        class MockTicTacToePool:
            def __init__(self):
                self._g = game_mod.TicTacToeGame(opponent="random", seed=42, max_steps=20)
                self._obs, self._info = None, None

            def reset(self):
                obs, info = self._g.reset(seed=42)
                self._obs, self._info = obs, info
                self._info["observation"] = obs
                return [obs], [dict(self._info)]

            def step(self, actions):
                raw = actions[0] if actions else ""
                # Parse action from raw text
                action_text = ""
                import re
                m = re.search(r'<action>(.*?)</action>', raw, re.DOTALL)
                if m:
                    action_text = m.group(1).strip()
                obs, reward, done, info = self._g.step(
                    action_text=action_text, parse_ok=bool(action_text), raw_action=raw)
                info["observation"] = obs
                return [obs], np.array([float(reward)], dtype=np.float32), np.array([done], dtype=bool), [info]

            def close(self):
                pass

        pool = MockTicTacToePool()
        config = SimpleNamespace(env=SimpleNamespace(history_length=0))
        mgr = TicTacToeEnvironmentManager(pool, lambda x: (x, [True]*len(x)), config)

        prompts = []
        actions_used = []

        # Initial observation from manager reset (not pool.reset())
        init_obs, init_infos = mgr.reset()
        prompts.append(init_obs["text"][0])

        cells = [1, 2, 3, 5, 7, 8, 4, 6, 9]  # 9 legal cells to ensure non-termination
        for cell in cells:
            action_text = f"<action>{cell}</action>"
            actions_used.append(action_text)
            obs_dict, rewards, dones, infos = mgr.step([action_text])
            prompts.append(obs_dict["text"][0])
            if len(prompts) >= 6:  # 1 initial + 5 steps
                break

        assert len(prompts) >= 5, f"Expected >= 5 prompts, got {len(prompts)}"

        # 1. Prompt length bounded (Markovian: no history appended)
        for i in range(1, len(prompts)):
            growth = len(prompts[i]) - len(prompts[0])
            assert growth <= 100, \
                f"Step {i} prompt grew {growth} chars (step0={len(prompts[0])}, step{i}={len(prompts[i])})"

        # 2. No prior action text in next prompt (Markovian: observation-only)
        for i in range(len(actions_used)):
            if i + 1 < len(prompts):
                act = actions_used[i][:50]  # check first 50 chars of action
                if len(act) >= 8:  # skip trivially short actions
                    assert act not in prompts[i+1], \
                        f"Action '{act}' from step {i} found in prompt at step {i+1} (history leak)"


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

    def test_sudoku_grouped_reset_exact_blanks_hard_seeds(self):
        """Through the real Ray builder, group replicas at hard seeds (4, 8) share an
        identical puzzle with exactly 40 blanks."""
        self._init_ray()
        from agent_system.environments.env_package.vpr_games.sudoku.envs import (
            build_sudoku_envs
        )
        for seed in [4, 8]:
            envs = build_sudoku_envs(seed=seed, env_num=1, group_n=2)
            obs_list, info_list = envs.reset()
            assert obs_list[0] == obs_list[1], (
                f"seed={seed} group replicas must start identically")
            for info in info_list:
                assert info["num_blanks_remaining"] == 40, (
                    f"seed={seed} produced {info['num_blanks_remaining']} blanks via builder")
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


# ---------------------------------------------------------------------------
# Shared outcome-reward helper + Sudoku outcome mode
# ---------------------------------------------------------------------------

class TestOutcomeRewardHelper:
    def _fn(self):
        from agent_system.environments.env_package.vpr_games.common.rewards import outcome_reward
        return outcome_reward

    def test_mapping(self):
        f = self._fn()
        assert f(False, None, None) == 0.0           # non-terminal
        assert f(True, True, "complete") == 1.0       # win
        assert f(True, False, "mine_hit") == -1.0     # loss
        assert f(True, False, "wrong_digit") == -1.0  # loss
        assert f(True, False, "invalid_action") == -1.0
        assert f(True, False, "timeout") == 0.0       # neutral
        assert f(True, False, None) == 0.0            # neutral (unspecified)


class TestSudokuOutcomeReward:
    def _w(self, clues=40, reward_mode="outcome", seed=0):
        return _su_envs_mod.SudokuWorker(seed=seed, n=3, clues=clues, max_turns=100,
                                         reward_mode=reward_mode)

    def test_invalid_reward_mode_raises(self):
        with pytest.raises(ValueError):
            _su_envs_mod.SudokuWorker(seed=0, n=3, clues=40, reward_mode="bogus")

    def test_outcome_wrong_digit_is_loss(self):
        w = self._w()
        w.reset(seed=42)
        r_str, c_str = w._blank_cells()[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        wrong = (correct % 9) + 1
        _, reward, done, info = w.step(f"<action>{r+1} {c+1} {wrong}</action>")
        assert done and reward == -1.0

    def test_outcome_correct_nonterminal_is_zero(self):
        w = self._w(clues=40)
        w.reset(seed=42)
        r_str, c_str = w._blank_cells()[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        _, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert not done and reward == 0.0

    def test_outcome_solve_is_win(self):
        # A single-blank puzzle: filling the one correct digit completes it → win → +1.
        w = self._w(clues=1)
        w.reset(seed=0)
        blanks = w._blank_cells()
        assert len(blanks) == 1
        r_str, c_str = blanks[0].split()
        r, c = int(r_str) - 1, int(c_str) - 1
        correct = w._env.full_grid[r][c]
        _, reward, done, info = w.step(f"<action>{r+1} {c+1} {correct}</action>")
        assert done and info["terminal_success"] and reward == 1.0

    def test_partial_outcome_target_rewards_only_on_twelfth_correct_fill(self):
        w = _su_envs_mod.SudokuWorker(
            seed=0,
            n=3,
            clues=40,
            max_turns=15,
            terminate_on_wrong_digit=False,
            reward_mode="outcome",
            outcome_success_correct_fills=12,
        )
        w.reset(seed=42)
        for expected_count in range(1, 13):
            r_str, c_str = w._blank_cells()[0].split()
            r, c = int(r_str) - 1, int(c_str) - 1
            correct = w._env.full_grid[r][c]
            _, reward, done, info = w.step(
                f"<action>{r + 1} {c + 1} {correct}</action>"
            )
            assert info["correct_fills"] == expected_count
            if expected_count < 12:
                assert not done and reward == 0.0
            else:
                assert done and reward == 1.0
                assert info["terminal_success"] is True
                assert info["terminal_reason"] == "outcome_target"
                assert info["outcome_target_completion_rate"] == 1.0

    def test_partial_outcome_target_snapshot_restores_progress(self):
        w = _su_envs_mod.SudokuWorker(
            seed=0,
            n=3,
            clues=40,
            max_turns=3,
            terminate_on_wrong_digit=False,
            reward_mode="outcome",
            outcome_success_correct_fills=2,
        )
        w.reset(seed=42)
        first = w._blank_cells()[0].split()
        row, col = int(first[0]) - 1, int(first[1]) - 1
        correct = w._env.full_grid[row][col]
        _, first_reward, first_done, _ = w.step(
            f"<action>{row + 1} {col + 1} {correct}</action>"
        )
        assert not first_done and first_reward == 0.0
        snapshot = w.snapshot_state()

        second = w._blank_cells()[0].split()
        row, col = int(second[0]) - 1, int(second[1]) - 1
        correct = w._env.full_grid[row][col]
        _, second_reward, second_done, _ = w.step(
            f"<action>{row + 1} {col + 1} {correct}</action>"
        )
        assert second_done and second_reward == 1.0

        _, restored_info = w.restore_state(snapshot)
        assert restored_info["correct_fills"] == 1
        assert restored_info["terminal_success"] is None

    def test_partial_outcome_timeout_is_neutral(self):
        w = _su_envs_mod.SudokuWorker(
            seed=0,
            n=3,
            clues=40,
            max_turns=15,
            terminate_on_wrong_digit=False,
            reward_mode="outcome",
            outcome_success_correct_fills=12,
        )
        w.reset(seed=42)
        for step in range(15):
            r_str, c_str = w._blank_cells()[0].split()
            r, c = int(r_str) - 1, int(c_str) - 1
            correct = w._env.full_grid[r][c]
            wrong = (correct % 9) + 1
            _, reward, done, info = w.step(
                f"<action>{r + 1} {c + 1} {wrong}</action>"
            )
            if step < 14:
                assert not done and reward == 0.0
        assert done and reward == 0.0
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "timeout"
        assert info["correct_fills"] == 0

    @pytest.mark.parametrize("target", [0, -1, 1.5, True, 16, 41])
    def test_partial_outcome_target_rejects_invalid_values(self, target):
        with pytest.raises(ValueError, match="outcome_success_correct_fills"):
            _su_envs_mod.SudokuWorker(
                seed=0,
                n=3,
                clues=40,
                max_turns=15,
                reward_mode="outcome",
                outcome_success_correct_fills=target,
            )

    def test_partial_outcome_target_requires_outcome_mode(self):
        with pytest.raises(ValueError, match="requires reward_mode='outcome'"):
            _su_envs_mod.SudokuWorker(
                seed=0,
                n=3,
                clues=40,
                max_turns=15,
                reward_mode="oracle",
                outcome_success_correct_fills=12,
            )
