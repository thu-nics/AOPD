"""Minesweeper posterior mine probability oracle.

Uses backtracking enumeration over hidden cell assignments, constrained by
revealed cell values and the global mine count. Flags are NOT used as evidence.

Enumerates configurations in lexicographic (row, col) order over hidden cells.
Stops after N_MAX_CONFIGS attempts and falls back to local single-constraint
deduction, exposing oracle_degraded=True.
"""

from __future__ import annotations

from math import comb
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

N_MAX_CONFIGS = 50_000


def _neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    return [
        (nr, nc)
        for nr in range(r - 1, r + 2)
        for nc in range(c - 1, c + 2)
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != (r, c)
    ]


def compute_posteriors(
    revealed: List[List[bool]],
    grid: List[List[int]],
    rows: int,
    cols: int,
    total_mines: int,
    n_max: int = N_MAX_CONFIGS,
) -> Tuple[Dict[Tuple[int, int], float], bool]:
    """Compute posterior mine probabilities for all hidden cells.

    Args:
        revealed: 2D bool grid — True means cell has been revealed.
        grid: 2D int grid — value of revealed cell (count of adjacent mines);
              only accessed for revealed cells (ground-truth hidden from oracle).
        rows, cols: board dimensions.
        total_mines: total mine count for the board.
        n_max: maximum configurations to enumerate before falling back.

    Returns:
        (posteriors, oracle_degraded) where posteriors maps (r,c) -> P(mine).
        oracle_degraded=True when the fallback was used.
    """
    # Hidden cells in lexicographic (row, col) order
    hidden = sorted(
        (r, c) for r in range(rows) for c in range(cols) if not revealed[r][c]
    )
    if not hidden:
        return {}, False

    # Build constraints from revealed cells that have at least one hidden neighbor
    # constraint: (required_mine_count, [hidden_neighbors])
    constraints: List[Tuple[int, List[Tuple[int, int]]]] = []
    for r in range(rows):
        for c in range(cols):
            if not revealed[r][c]:
                continue
            v = grid[r][c]
            if v <= 0:
                continue
            hidden_nbrs = [(nr, nc) for (nr, nc) in _neighbors(r, c, rows, cols) if not revealed[nr][nc]]
            if hidden_nbrs:
                constraints.append((v, hidden_nbrs))

    # Index: for each hidden cell, which constraints involve it
    cell_to_constraints = defaultdict(list)
    for ci, (v, nbrs) in enumerate(constraints):
        for cell in nbrs:
            cell_to_constraints[cell].append(ci)

    # Backtracking enumeration in lexicographic order
    mine_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_valid = 0
    steps_taken = [0]
    budget_exceeded = [False]

    # Track how many mines each constraint has assigned so far
    constraint_mines = [0] * len(constraints)
    constraint_total = [len(nbrs) for _, nbrs in constraints]

    assignment: Dict[Tuple[int, int], int] = {}  # 1=mine, 0=safe

    def backtrack(idx: int, mines_placed: int) -> None:
        if budget_exceeded[0]:
            return
        if steps_taken[0] >= n_max:
            budget_exceeded[0] = True
            return

        if idx == len(hidden):
            if mines_placed != total_mines:
                return
            # Check all constraints are exactly satisfied
            for ci, (v, _) in enumerate(constraints):
                if constraint_mines[ci] != v:
                    return
            total_valid_ref = total_valid  # will reassign below
            return  # handled after the call to backtrack returns True

        cell = hidden[idx]
        steps_taken[0] += 1

        for is_mine in (0, 1):
            # Pruning: don't exceed total mines
            if is_mine and mines_placed + 1 > total_mines:
                continue
            # Pruning: can't place enough remaining mines
            remaining_cells = len(hidden) - idx - 1
            if not is_mine and (mines_placed + remaining_cells) < total_mines:
                continue

            # Update constraint counters
            affected = cell_to_constraints[cell]
            feasible = True
            for ci in affected:
                constraint_mines[ci] += is_mine
                v_req = constraints[ci][0]
                placed_so_far = constraint_mines[ci]
                total_in_constraint = constraint_total[ci]
                assigned_so_far = idx + 1  # number of hidden cells processed
                # If constraint already exceeded
                if placed_so_far > v_req:
                    feasible = False
                    break

            if feasible:
                assignment[cell] = is_mine
                backtrack(idx + 1, mines_placed + is_mine)
                del assignment[cell]

            for ci in affected:
                constraint_mines[ci] -= is_mine

    # We need a mutable total_valid for the inner function
    # Use a list as a mutable container
    valid_configs: List[Tuple[Dict, int]] = []

    def backtrack2(idx: int, mines_placed: int) -> None:
        if budget_exceeded[0]:
            return
        if steps_taken[0] >= n_max:
            budget_exceeded[0] = True
            return

        if idx == len(hidden):
            if mines_placed != total_mines:
                return
            for ci, (v, _) in enumerate(constraints):
                if constraint_mines[ci] != v:
                    return
            valid_configs.append(dict(assignment))
            return

        cell = hidden[idx]
        steps_taken[0] += 1

        for is_mine in (0, 1):
            if is_mine and mines_placed + 1 > total_mines:
                continue
            remaining_cells = len(hidden) - idx - 1
            if not is_mine and (mines_placed + remaining_cells) < total_mines:
                continue

            affected = cell_to_constraints[cell]
            feasible = True
            for ci in affected:
                constraint_mines[ci] += is_mine
                if constraint_mines[ci] > constraints[ci][0]:
                    feasible = False
                    break

            if feasible:
                assignment[cell] = is_mine
                backtrack2(idx + 1, mines_placed + is_mine)
                del assignment[cell]

            for ci in affected:
                constraint_mines[ci] -= is_mine

    backtrack2(0, 0)

    if budget_exceeded[0] or not valid_configs:
        return _local_fallback(hidden, constraints, total_mines), True

    # Aggregate
    total = len(valid_configs)
    mine_counts = defaultdict(int)
    for cfg in valid_configs:
        for cell, is_mine in cfg.items():
            if is_mine:
                mine_counts[cell] += 1

    posteriors = {cell: mine_counts[cell] / total for cell in hidden}
    return posteriors, False


def _local_fallback(
    hidden: List[Tuple[int, int]],
    constraints: List[Tuple[int, List[Tuple[int, int]]]],
    total_mines: int,
) -> Dict[Tuple[int, int], float]:
    """Simple local deduction: cells forced safe or forced mine by single constraints."""
    forced_mine: set[Tuple[int, int]] = set()
    forced_safe: set[Tuple[int, int]] = set()

    for v, nbrs in constraints:
        hidden_in_constraint = [cell for cell in nbrs]
        if v == len(hidden_in_constraint):
            forced_mine.update(hidden_in_constraint)
        elif v == 0:
            forced_safe.update(hidden_in_constraint)

    uncertain = [cell for cell in hidden if cell not in forced_mine and cell not in forced_safe]
    n_uncertain = len(uncertain)
    remaining = total_mines - len(forced_mine)
    default_prob = remaining / n_uncertain if n_uncertain > 0 else 0.5

    posteriors = {}
    for cell in hidden:
        if cell in forced_mine:
            posteriors[cell] = 1.0
        elif cell in forced_safe:
            posteriors[cell] = 0.0
        else:
            posteriors[cell] = default_prob
    return posteriors


def get_oracle_actions(
    posteriors: Dict[Tuple[int, int], float],
    revealed: List[List[bool]],
    flags: List[List[bool]],
    rows: int,
    cols: int,
    eps: float = 1e-9,
) -> Tuple[List[str], float, float]:
    """Determine oracle-valid reveal and flag actions.

    Returns (oracle_valid_actions, min_prob, flag_threshold=1.0).
    oracle_valid_actions: list of action strings (1-indexed)
    min_prob: minimum posterior probability for reveal oracle
    """
    unrevealed_unflagged = [
        cell for cell in posteriors
        if not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]
    ]
    if not unrevealed_unflagged:
        return [], 0.0, 1.0

    probs = {cell: posteriors[cell] for cell in unrevealed_unflagged}
    min_prob = min(probs.values())

    oracle_actions = []
    # Reveal oracle: cells with minimum posterior probability
    for cell, p in probs.items():
        if abs(p - min_prob) < eps:
            oracle_actions.append(f"reveal {cell[0]+1} {cell[1]+1}")  # convert to 1-indexed

    # Flag oracle: cells with posterior == 1.0 (exact integer comparison via fraction)
    for cell, p in posteriors.items():
        if not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]:
            if p >= 1.0 - eps:
                oracle_actions.append(f"flag {cell[0]+1} {cell[1]+1}")

    return oracle_actions, min_prob, 1.0
