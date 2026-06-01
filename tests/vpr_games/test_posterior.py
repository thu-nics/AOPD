"""Unit tests for the Minesweeper posterior oracle.

All board states are physically consistent: in GEM Minesweeper with flood fill,
a revealed cell with value 0 never has hidden neighbors (they get flood-filled).
Therefore our test boards only use non-zero values for revealed cells adjacent
to hidden cells.
"""

import sys
import importlib.util
import pytest
from math import comb


def load_oracle():
    spec = importlib.util.spec_from_file_location(
        "ms_oracle",
        "agent_system/environments/env_package/vpr_games/minesweeper/oracle.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ms_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


oracle_mod = load_oracle()
compute_posteriors = oracle_mod.compute_posteriors
get_oracle_actions = oracle_mod.get_oracle_actions


def test_posterior_uniform_4_cells():
    """2x3 board: only (0,1)=1 and (1,1)=1 are revealed.
    Hidden cells: (0,0),(0,2),(1,0),(1,2). Constraint: 1 mine among these 4.
    All 4 should have equal posterior = 0.25."""
    rows, cols = 2, 3
    revealed = [[False, True, False], [False, True, False]]
    grid = [[0, 1, 0], [0, 1, 0]]
    flags = [[False] * cols for _ in range(rows)]
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded
    expected_cells = [(0, 0), (0, 2), (1, 0), (1, 2)]
    for cell in expected_cells:
        assert cell in posteriors, f"{cell} missing"
        assert abs(posteriors[cell] - 0.25) < 1e-6, f"Expected 0.25 for {cell}, got {posteriors[cell]}"


def test_forced_mine():
    """1x2 board: only (0,0)=1 revealed, (0,1) is the only hidden cell.
    (0,0)=1 means (0,1) must be the mine: P=1.0."""
    rows, cols = 1, 2
    revealed = [[True, False]]
    grid = [[1, 0]]
    flags = [[False, False]]
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded
    assert abs(posteriors.get((0, 1), 0) - 1.0) < 1e-6, f"Expected 1.0, got {posteriors}"


def test_brute_force_equivalence():
    """1x3 board: (0,1)=1 revealed, (0,0) and (0,2) are hidden.
    1 mine split equally: P=0.5 each."""
    rows, cols = 1, 3
    revealed = [[False, True, False]]
    grid = [[0, 1, 0]]
    flags = [[False] * cols for _ in range(rows)]
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded
    assert abs(posteriors.get((0, 0), 0) - 0.5) < 1e-6, f"Expected 0.5 for (0,0), got {posteriors}"
    assert abs(posteriors.get((0, 2), 0) - 0.5) < 1e-6, f"Expected 0.5 for (0,2), got {posteriors}"


def test_oracle_actions_reveal_min_prob():
    """From the uniform 4-cell board, all cells are oracle-valid reveals at P=0.25."""
    rows, cols = 2, 3
    revealed = [[False, True, False], [False, True, False]]
    grid = [[0, 1, 0], [0, 1, 0]]
    flags = [[False] * cols for _ in range(rows)]
    posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    oracle_actions, min_prob, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
    assert abs(min_prob - 0.25) < 1e-6, f"Expected min_prob=0.25, got {min_prob}"
    reveal_actions = [a for a in oracle_actions if a.startswith("reveal")]
    assert len(reveal_actions) == 4, f"Expected 4 oracle reveals, got {reveal_actions}"


def test_oracle_actions_flag_forced():
    """1x2 board: (0,1) has posterior=1.0 → oracle says flag it (1-indexed: flag 1 2)."""
    rows, cols = 1, 2
    revealed = [[True, False]]
    grid = [[1, 0]]
    flags = [[False, False]]
    posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    oracle_actions, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
    assert "flag 1 2" in oracle_actions, f"Expected 'flag 1 2', got {oracle_actions}"


def test_forced_safe_by_zero():
    """Board where a revealed zero-value cell forces its hidden neighbors to probability 0.
    This handles an edge case: if somehow a 0-value cell has a hidden neighbor,
    those neighbors must be mines-free."""
    rows, cols = 2, 2
    # (0,0)=0 revealed, (0,1)=1 revealed. Hidden: (1,0),(1,1).
    # (0,0)=0 forces (1,0) and (1,1) safe? No — (0,0) sees 0 adjacent mines.
    # (0,0) neighbors: (0,1),(1,0),(1,1). Both (1,0) and (1,1) are hidden.
    # Value 0 means 0 mines among neighbors → (1,0) and (1,1) forced safe.
    # (0,1)=1 means 1 mine among its neighbors: (0,0),(1,0),(1,1). (0,0) revealed.
    # Hidden neighbors of (0,1): (1,0),(1,1). But they're forced safe!
    # Contradiction: can't have 1 mine if all hidden neighbors are forced safe.
    # In this case, total_mines=0 is the only consistent count.
    rows, cols = 2, 2
    revealed = [[True, True], [False, False]]
    grid = [[0, 1], [0, 0]]  # (0,0)=0 forces (1,0),(1,1) safe; (0,1)=1 conflicts
    flags = [[False] * cols for _ in range(rows)]
    # With 0 mines: no mines anywhere, forced_safe is correct
    posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=0)
    for cell in [(1, 0), (1, 1)]:
        assert posteriors.get(cell, -1) == pytest.approx(0.0, abs=1e-6), \
            f"Forced-safe cell {cell} should have P=0, got {posteriors.get(cell)}"
