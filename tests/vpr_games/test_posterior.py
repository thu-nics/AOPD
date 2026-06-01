"""Unit tests for the Minesweeper posterior oracle."""

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


def make_3x3_board():
    """3x3 board with 1 mine at (2,2) — 0-indexed."""
    rows, cols = 3, 3
    # revealed: top-left 2x2 plus (0,2)
    revealed = [
        [True, True, True],
        [True, True, False],
        [False, False, False],
    ]
    # grid values (after first reveal — only matter for revealed cells)
    grid = [
        [0, 0, 0],
        [0, 1, 0],  # (1,1) sees 1 mine
        [0, 0, 0],
    ]
    flags = [[False] * cols for _ in range(rows)]
    return rows, cols, revealed, grid, flags


def test_posterior_3x3():
    """With 1 mine in unrevealed region [(1,2),(2,0),(2,1),(2,2)] and cell (1,1)=1,
    the mine can only be at (1,2), (2,0), (2,1), or (2,2).
    Given constraint: (1,1)=1 means exactly one of its hidden neighbors is a mine.
    Hidden neighbors of (1,1): (1,2), (2,0), (2,1), (2,2).
    So all 4 cells have equal probability = 1/4 = 0.25."""
    rows, cols, revealed, grid, flags = make_3x3_board()
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded, "Should not degrade for 3x3"
    hidden = [(1, 2), (2, 0), (2, 1), (2, 2)]
    for cell in hidden:
        assert cell in posteriors, f"Cell {cell} missing from posteriors"
        assert abs(posteriors[cell] - 0.25) < 1e-6, f"Expected 0.25 for {cell}, got {posteriors[cell]}"


def test_forced_mine():
    """Board where only 1 hidden cell can be the mine."""
    rows, cols = 2, 2
    revealed = [[True, True], [True, False]]
    grid = [[0, 0], [0, 1]]  # (1,1) sees 1 mine, but (1,1) is hidden
    # Hidden: only (1,1). Constraint: (1,0)? No...
    # Let's make (0,1)=1: its only hidden neighbor is (1,1)
    grid = [[0, 1], [0, 0]]
    revealed = [[True, True], [True, False]]
    flags = [[False] * cols for _ in range(rows)]
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded
    assert abs(posteriors.get((1, 1), 0) - 1.0) < 1e-6, f"Expected 1.0, got {posteriors}"


def test_brute_force_equivalence():
    """Compare oracle with brute force on a tiny board."""
    rows, cols = 2, 3
    # 1 mine, 2 hidden cells at (1,0) and (1,2)
    revealed = [[True, True, True], [False, True, False]]
    grid = [[0, 0, 0], [0, 1, 0]]  # (1,1)=1 sees exactly 1 mine among (1,0) and (1,2)
    flags = [[False] * cols for _ in range(rows)]
    posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    assert not degraded
    # Brute force: 2 valid configs, each puts mine in one of (1,0) or (1,2)
    # P(mine at (1,0)) = 1/2, P(mine at (1,2)) = 1/2
    assert abs(posteriors.get((1, 0), 0) - 0.5) < 1e-6
    assert abs(posteriors.get((1, 2), 0) - 0.5) < 1e-6


def test_oracle_actions_reveal_min_prob():
    rows, cols, revealed, grid, flags = make_3x3_board()
    posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    oracle_actions, min_prob, _ = get_oracle_actions(
        posteriors, revealed, flags, rows, cols
    )
    # All cells have equal prob=0.25, so all are oracle-valid for reveal
    assert min_prob == pytest.approx(0.25, abs=1e-6)
    reveal_actions = [a for a in oracle_actions if a.startswith("reveal")]
    assert len(reveal_actions) == 4  # all 4 hidden cells


def test_oracle_actions_flag_forced():
    """Forced mine should appear in oracle flag actions."""
    rows, cols = 2, 2
    revealed = [[True, True], [True, False]]
    grid = [[0, 1], [0, 0]]
    flags = [[False] * cols for _ in range(rows)]
    posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
    oracle_actions, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
    # (1,1) has posterior=1.0, so flagging it is oracle-valid
    assert "flag 2 2" in oracle_actions  # 1-indexed: row=2, col=2
