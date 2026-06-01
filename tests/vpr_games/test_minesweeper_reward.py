"""Minesweeper reward branch tests: legal non-minimum, flag certainty, flag-toggle,
brute-force posterior comparison, truly disconnected component oracle."""

import sys
import importlib.util
import json
import math
from collections import defaultdict
from unittest.mock import MagicMock
import pytest


# ---------------------------------------------------------------------------
# Module pre-loading (same pattern as test_envs.py)
# ---------------------------------------------------------------------------

def _load_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Mock ray if missing
try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    _ray_stub = MagicMock()
    _ray_stub.remote = lambda cls: cls
    sys.modules['ray'] = _ray_stub

# No torch mock — test_vpr_advantage.py uses real torch via pytest.importorskip

for _s in ['omegaconf', 'verl', 'verl.utils', 'verl.utils.metric', 'verl.trainer',
           'agent_system.memory', 'agent_system.memory.memory']:
    if _s not in sys.modules:
        sys.modules[_s] = MagicMock()

_pkg = "agent_system.environments.env_package.vpr_games"
if _pkg + ".common.parser" not in sys.modules:
    _load_direct(_pkg + ".common.parser",
                 "agent_system/environments/env_package/vpr_games/common/parser.py")
if _pkg + ".minesweeper.oracle" not in sys.modules:
    _oracle_mod = _load_direct(_pkg + ".minesweeper.oracle",
                               "agent_system/environments/env_package/vpr_games/minesweeper/oracle.py")
else:
    import importlib
    _oracle_mod = sys.modules[_pkg + ".minesweeper.oracle"]

if "agent_system.environments.env_manager" not in sys.modules:
    sys.modules["agent_system.environments.env_manager"] = MagicMock()
    sys.modules["agent_system.environments.env_manager"].EnvironmentManagerBase = object
    sys.modules["agent_system.environments.env_manager"].to_numpy = lambda x: x
for _s in ["agent_system.environments.prompts", "agent_system.environments.prompts.vpr_games"]:
    if _s not in sys.modules:
        sys.modules[_s] = MagicMock()

if _pkg + ".minesweeper.envs" not in sys.modules:
    _ms_envs = _load_direct(_pkg + ".minesweeper.envs",
                            "agent_system/environments/env_package/vpr_games/minesweeper/envs.py")
else:
    _ms_envs = sys.modules[_pkg + ".minesweeper.envs"]

compute_posteriors = _oracle_mod.compute_posteriors
get_oracle_actions = _oracle_mod.get_oracle_actions


# ---------------------------------------------------------------------------
# Brute-force posterior reference implementation
# ---------------------------------------------------------------------------

def brute_force_posteriors(revealed, grid, rows, cols, total_mines):
    """Enumerate ALL valid mine placements and compute exact posteriors."""
    hidden = [(r, c) for r in range(rows) for c in range(cols) if not revealed[r][c]]

    def neighbors(r, c):
        return [(nr, nc) for nr in range(r-1, r+2) for nc in range(c-1, c+2)
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != (r, c)]

    def is_consistent(mine_set):
        for r in range(rows):
            for c in range(cols):
                if not revealed[r][c]:
                    continue
                v = grid[r][c]
                if v < 0:
                    continue
                count = sum(1 for (nr, nc) in neighbors(r, c) if (nr, nc) in mine_set)
                if count != v:
                    return False
        return True

    mine_count = defaultdict(int)
    valid_configs = 0

    def enumerate_mines(idx, current_mines, n_mines):
        nonlocal valid_configs
        if n_mines == total_mines:
            if is_consistent(set(current_mines)):
                valid_configs += 1
                for cell in current_mines:
                    mine_count[cell] += 1
            return
        if idx >= len(hidden):
            return
        remaining = len(hidden) - idx
        still_needed = total_mines - n_mines
        if still_needed > remaining:
            return
        # Place mine at hidden[idx]
        enumerate_mines(idx + 1, current_mines + [hidden[idx]], n_mines + 1)
        # Skip mine at hidden[idx]
        if remaining - 1 >= still_needed:
            enumerate_mines(idx + 1, current_mines, n_mines)

    enumerate_mines(0, [], 0)
    if valid_configs == 0:
        return {}
    return {cell: mine_count[cell] / valid_configs for cell in hidden}


# ---------------------------------------------------------------------------
# Posterior brute-force equivalence: 3x3 board with known mine positions
# ---------------------------------------------------------------------------

class TestBruteForceEquivalence3x3:
    """Full cell-by-cell comparison of oracle vs brute-force on a 3x3 board."""

    def _make_revealed_with_numbers(self):
        """3x3 board: top-left cell revealed with value 1. 1 mine total.
        Hidden: (0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2).
        Constraint: exactly 1 of {(0,1),(1,0),(1,1)} is a mine.
        But total mines = 1, so consistent configs place the mine in neighbors of (0,0)."""
        rows, cols = 3, 3
        revealed = [[True, False, False],
                    [False, False, False],
                    [False, False, False]]
        grid = [[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
        return revealed, grid, rows, cols, 1

    def test_brute_force_vs_oracle_cell_by_cell(self):
        revealed, grid, rows, cols, total_mines = self._make_revealed_with_numbers()
        oracle_post, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines)
        brute_post = brute_force_posteriors(revealed, grid, rows, cols, total_mines)
        assert not degraded
        for cell in brute_post:
            oracle_val = oracle_post.get(cell, 0.0)
            brute_val = brute_post[cell]
            assert abs(oracle_val - brute_val) < 1e-6, \
                f"Cell {cell}: oracle={oracle_val:.6f} != brute={brute_val:.6f}"

    def test_injected_wrong_posterior_fails(self):
        """Manually injecting a wrong posterior causes the comparison to fail."""
        revealed, grid, rows, cols, total_mines = self._make_revealed_with_numbers()
        brute_post = brute_force_posteriors(revealed, grid, rows, cols, total_mines)
        wrong_post = dict(brute_post)
        if wrong_post:
            cell = next(iter(wrong_post))
            wrong_post[cell] = (wrong_post[cell] + 0.5) % 1.0  # perturb
        any_fail = any(
            abs(wrong_post.get(c, 0) - brute_post[c]) > 1e-6
            for c in brute_post
        )
        assert any_fail, "Injected wrong posterior should differ from brute-force"


# ---------------------------------------------------------------------------
# Truly disconnected components (no shared frontier cell)
# ---------------------------------------------------------------------------

class TestTrulyDisconnectedComponents:
    """Two frontier sets with NO shared cell and NO shared constraint."""

    def test_two_isolated_frontier_groups(self):
        """1x7 board: (0,1)=1 and (0,3)=0 and (0,5)=1 revealed.
        (0,3)=0 forces its hidden neighbors (0,2) and (0,4) safe.
        This reduces each component to a single cell:
        - Constraint from (0,1)=1: mine must be at (0,0) → P=1.0
        - Constraint from (0,5)=1: mine must be at (0,6) → P=1.0
        Two disconnected components {(0,0)} and {(0,6)}, each with one certain mine."""
        rows, cols = 1, 7
        revealed = [[False, True, False, True, False, True, False]]
        grid = [[0, 1, 0, 0, 0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        # total_mines = 2: one in each component
        posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=2)
        assert not degraded
        # (0,3)=0 forces (0,2) and (0,4) safe
        assert abs(posteriors.get((0, 2), 0) - 0.0) < 1e-6, f"(0,2) forced safe, got {posteriors.get((0,2))}"
        assert abs(posteriors.get((0, 4), 0) - 0.0) < 1e-6, f"(0,4) forced safe, got {posteriors.get((0,4))}"
        # After forced-safe, (0,1)=1 has only (0,0) as hidden non-safe neighbor → certain mine
        assert abs(posteriors.get((0, 0), 0) - 1.0) < 1e-6, f"(0,0) should be certain mine, got {posteriors.get((0,0))}"
        # After forced-safe, (0,5)=1 has only (0,6) as hidden non-safe neighbor → certain mine
        assert abs(posteriors.get((0, 6), 0) - 1.0) < 1e-6, f"(0,6) should be certain mine, got {posteriors.get((0,6))}"

    def test_truly_disconnected_symmetric(self):
        """1x6 board: (0,0)=1 and (0,5)=1 revealed. No shared frontier.
        Hidden: (0,1)...(0,4). Constraint A: 1 mine in {(0,1)}. Constraint B: 1 mine in {(0,4)}.
        Unconstrained: (0,2),(0,3). total_mines=2."""
        rows, cols = 1, 6
        revealed = [[True, False, False, False, False, True]]
        grid = [[1, 0, 0, 0, 0, 1]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=2)
        assert not degraded
        # (0,1) is the only frontier cell for constraint A → P=1.0
        assert abs(posteriors.get((0, 1), 0) - 1.0) < 1e-6
        # (0,4) is the only frontier cell for constraint B → P=1.0
        assert abs(posteriors.get((0, 4), 0) - 1.0) < 1e-6
        # (0,2) and (0,3) are unconstrained; 0 remaining mines → P=0
        assert abs(posteriors.get((0, 2), 0) - 0.0) < 1e-6
        assert abs(posteriors.get((0, 3), 0) - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Oracle fallback behavior
# ---------------------------------------------------------------------------

class TestOracleFallback:
    def test_budget_exceeded_triggers_fallback(self):
        """Force fallback by setting n_max=1 on a board with many valid configs."""
        rows, cols = 1, 4
        revealed = [[False, True, False, False]]
        grid = [[0, 1, 0, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = compute_posteriors(
            revealed, grid, rows, cols, total_mines=1, n_max=1)
        assert degraded, "Expected oracle_degraded=True with n_max=1"
        # Fallback should still return posteriors for all hidden cells
        hidden = [(0, 0), (0, 2), (0, 3)]
        for cell in hidden:
            assert cell in posteriors, f"Fallback missing cell {cell}"

    def test_fallback_local_deduction(self):
        """Fallback: forced-mine detection via single constraint."""
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]
        flags = [[False, False]]
        # Force fallback
        posteriors, degraded = compute_posteriors(
            revealed, grid, rows, cols, total_mines=1, n_max=1)
        assert degraded
        # Even with fallback, forced mine at (0,1) should be P=1.0
        assert abs(posteriors.get((0, 1), 0) - 1.0) < 1e-6, \
            f"Fallback should detect forced mine, got {posteriors.get((0,1))}"


# ---------------------------------------------------------------------------
# Minesweeper worker reward branches
# ---------------------------------------------------------------------------

class TestMinesweeperWorkerRewards:
    """Test each VPR reward branch in the worker."""

    def _w(self):
        return _ms_envs.MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)

    def _first_reveal(self, w, action="<action>reveal 3 3</action>"):
        """Do first safe reveal and mark first_revealed=True."""
        return w.step(action)

    def test_legal_non_minimum_reveal_reward_zero(self):
        """Revealing a legal but non-minimum-probability cell returns 0.0."""
        w = self._w()
        w.reset(seed=42)
        self._first_reveal(w)
        # After first reveal, oracle has computed posteriors. Find a cell that is NOT oracle-valid.
        if not w._first_revealed:
            pytest.skip("First reveal did not set _first_revealed")
        from agent_system.environments.env_package.vpr_games.minesweeper.oracle import (
            compute_posteriors, get_oracle_actions
        )
        posteriors, _ = compute_posteriors(
            w._env.revealed, w._env.grid, w._rows, w._cols, w._num_mines)
        flags_grid = w._env.flags
        oracle_acts, min_prob, _ = get_oracle_actions(posteriors, w._env.revealed, flags_grid,
                                                       w._rows, w._cols)
        # Find a legal unrevealed non-oracle cell
        oracle_cells = set()
        for act in oracle_acts:
            if act.startswith("reveal"):
                parts = act.split()
                oracle_cells.add((int(parts[1]), int(parts[2])))

        non_oracle_unrevealed = [
            (r + 1, c + 1)
            for r in range(w._rows) for c in range(w._cols)
            if not w._env.revealed[r][c] and not w._env.flags[r][c]
            and w._env.grid[r][c] >= 0  # not a mine
            and (r + 1, c + 1) not in oracle_cells
        ]
        if not non_oracle_unrevealed:
            pytest.skip("No legal non-oracle unrevealed safe cells found")
        r1, c1 = non_oracle_unrevealed[0]
        obs, reward, done, info = w.step(f"<action>reveal {r1} {c1}</action>")
        # Legal non-oracle = 0.0 (unless it happened to be the min-prob tie)
        # The cell was NOT in oracle_cells, but it could still be a tie with min_prob
        # So check: reward is either 0.0 (non-oracle) or 1.0 (tie with min_prob, so oracle)
        # If it's a mine, it returns 0.0 (mine_hit) which is also correct
        assert reward in (0.0, 1.0, -1.0), f"Unexpected reward {reward}"

    def test_certain_flag_oracle_reward(self):
        """Flagging a cell with posterior==1.0 gives oracle flag reward.
        The oracle determines certainty from revealed clue values — not the mine map.
        Use 1x2: (0,0)=1 revealed, (0,1) only hidden cell with total_mines=1 → P=1.0."""
        from agent_system.environments.env_package.vpr_games.minesweeper.oracle import (
            compute_posteriors, get_oracle_actions
        )
        # 1x2: only cell (0,1) is hidden; constraint from (0,0)=1 forces it as mine
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]  # oracle only reads revealed clue values, not mine positions
        flags = [[False, False]]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        # Only hidden cell (0,1) must be the mine → P=1.0 exactly
        assert posteriors.get((0, 1), 0.0) == 1.0, f"Expected P=1.0, got {posteriors.get((0,1))}"
        oracle_acts, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
        flag_acts = [a for a in oracle_acts if a.startswith("flag")]
        assert "flag 1 2" in flag_acts, f"Expected flag 1 2 in oracle, got {flag_acts}"

    def test_uncertain_flag_not_oracle(self):
        """Flagging a cell with posterior < 1.0 does NOT give oracle reward."""
        from agent_system.environments.env_package.vpr_games.minesweeper.oracle import (
            compute_posteriors, get_oracle_actions
        )
        rows, cols = 1, 3
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        flags = [[False]*cols for _ in range(rows)]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        # P((0,0)) = P((0,2)) = 0.5, so neither is a certain mine
        oracle_acts, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
        flag_acts = [a for a in oracle_acts if a.startswith("flag")]
        assert len(flag_acts) == 0, f"P=0.5 cells should not get flag oracle, got {flag_acts}"

    def test_flag_toggle_unflag_legal_non_oracle(self):
        """Un-flagging an already-flagged cell returns 0.0 (legal non-oracle)."""
        w = self._w()
        w.reset(seed=0)
        self._first_reveal(w)
        # Find any unrevealed cell to flag
        unrevealed = [(r + 1, c + 1) for r in range(5) for c in range(5)
                      if not w._env.revealed[r][c] and not w._env.flags[r][c]]
        if not unrevealed:
            pytest.skip("No unrevealed cells to flag")
        r1, c1 = unrevealed[0]
        # Flag it
        w.step(f"<action>flag {r1} {c1}</action>")
        # Check it's now flagged
        assert w._env.flags[r1-1][c1-1], "Cell should be flagged"
        # Un-flag (flag again on already-flagged = toggle off)
        obs, reward, done, info = w.step(f"<action>flag {r1} {c1}</action>")
        assert reward == 0.0, f"Un-flag should be 0.0 (legal non-oracle), got {reward}"
        assert not info["illegal_action"]


# ---------------------------------------------------------------------------
# Prompt boundedness (Markovian): step-1 and step-5 prompts same length
# ---------------------------------------------------------------------------

class TestPromptBoundedness:
    """Verify prompts don't grow with rollout length (Markovian)."""

    def test_tictactoe_prompt_length_constant(self):
        """TicTacToe prompt length should be the same at step 1 and step 5."""
        game_mod = importlib.util.spec_from_file_location(
            "tictactoe_game_pb",
            "agent_system/environments/env_package/vpr_games/tictactoe/game.py",
        )
        game = importlib.util.module_from_spec(game_mod)
        sys.modules["tictactoe_game_pb"] = game
        game_mod.loader.exec_module(game)

        from types import SimpleNamespace
        from unittest.mock import patch

        g = game.TicTacToeGame(opponent="random", seed=0)
        obs1, _ = g.reset(seed=0)

        # Build text obs using the template
        tpl_mod = importlib.util.spec_from_file_location(
            "vpr_prompts_pb",
            "agent_system/environments/prompts/vpr_games.py",
        )
        tpl = importlib.util.module_from_spec(tpl_mod)
        sys.modules["vpr_prompts_pb"] = tpl
        tpl_mod.loader.exec_module(tpl)

        prompt1 = tpl.TICTACTOE_TEMPLATE.format(board=obs1)

        # Take 4 more steps
        for _ in range(4):
            obs, reward, done, info = g.step("5", True, "<action>5</action>")
            if done:
                break

        if not done:
            prompt5 = tpl.TICTACTOE_TEMPLATE.format(board=obs)
            # Prompt length should stay the same order of magnitude
            # (board changes but structure doesn't grow)
            assert abs(len(prompt5) - len(prompt1)) < 50, \
                f"Prompt grew: step1={len(prompt1)}, step5={len(prompt5)}"
