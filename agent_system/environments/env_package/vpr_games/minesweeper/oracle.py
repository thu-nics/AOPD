"""Minesweeper posterior mine probability oracle.

Uses frontier decomposition: only enumerates hidden cells adjacent to revealed
numbered cells (the frontier), then aggregates unconstrained hidden cells
analytically using the global mine count. Flags are NOT used as evidence.

Enumeration is in lexicographic (row, col) order over frontier cells.
Stops after N_MAX_CONFIGS configurations and falls back to local single-
constraint deduction, exposing oracle_degraded=True.
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

    Uses frontier decomposition: enumerates only frontier cells (hidden cells
    adjacent to at least one revealed numbered cell). Unconstrained hidden cells
    are aggregated analytically using comb(n_unconstrained, remaining_mines).

    Args:
        revealed: 2D bool grid — True means cell has been revealed.
        grid: 2D int grid — value for revealed cells; ignored for hidden.
        rows, cols: board dimensions.
        total_mines: total mine count for the board.
        n_max: maximum configurations to enumerate before falling back.

    Returns:
        (posteriors, oracle_degraded) where posteriors maps (r,c) -> P(mine).
        oracle_degraded=True when the local fallback was used.
    """
    hidden_set = set(
        (r, c) for r in range(rows) for c in range(cols) if not revealed[r][c]
    )
    if not hidden_set:
        return {}, False

    # Build constraints from revealed cells with at least one hidden neighbor
    constraints: List[Tuple[int, List[Tuple[int, int]]]] = []
    for r in range(rows):
        for c in range(cols):
            if not revealed[r][c]:
                continue
            v = grid[r][c]
            if v <= 0:
                continue
            hidden_nbrs = [cell for cell in _neighbors(r, c, rows, cols) if cell in hidden_set]
            if hidden_nbrs:
                constraints.append((v, hidden_nbrs))

    # Frontier = hidden cells appearing in at least one constraint
    frontier_set: set = set()
    for _, nbrs in constraints:
        frontier_set.update(nbrs)
    frontier = sorted(frontier_set)  # lexicographic

    # Unconstrained = hidden cells not in any constraint
    unconstrained = [cell for cell in sorted(hidden_set) if cell not in frontier_set]
    n_unconstrained = len(unconstrained)

    # Index: for each frontier cell, which constraints involve it
    cell_to_ci: Dict = defaultdict(list)
    for ci, (_, nbrs) in enumerate(constraints):
        for cell in nbrs:
            if cell in frontier_set:
                cell_to_ci[cell].append(ci)

    # Per-constraint mine counters during backtracking
    constraint_mines = [0] * len(constraints)
    constraint_req = [v for v, _ in constraints]

    valid_configs = []
    steps_taken = [0]
    budget_exceeded = [False]
    assignment: Dict = {}

    def backtrack(idx: int, frontier_mines: int) -> None:
        if budget_exceeded[0]:
            return
        if steps_taken[0] >= n_max:
            budget_exceeded[0] = True
            return

        if idx == len(frontier):
            for ci, req in enumerate(constraint_req):
                if constraint_mines[ci] != req:
                    return
            remaining = total_mines - frontier_mines
            if remaining < 0 or remaining > n_unconstrained:
                return
            valid_configs.append((dict(assignment), frontier_mines))
            return

        cell = frontier[idx]
        steps_taken[0] += 1
        affected = cell_to_ci[cell]

        for is_mine in (0, 1):
            if is_mine and frontier_mines + 1 > total_mines:
                continue
            remaining_frontier = len(frontier) - idx - 1
            min_mines_needed = max(total_mines - frontier_mines - int(is_mine) - n_unconstrained, 0)
            if min_mines_needed > remaining_frontier:
                continue

            feasible = True
            for ci in affected:
                constraint_mines[ci] += is_mine
                if constraint_mines[ci] > constraint_req[ci]:
                    feasible = False
                    break

            if feasible:
                assignment[cell] = is_mine
                backtrack(idx + 1, frontier_mines + is_mine)
                del assignment[cell]

            for ci in affected:
                constraint_mines[ci] -= is_mine

    backtrack(0, 0)

    if budget_exceeded[0] or not valid_configs:
        return _local_fallback(sorted(hidden_set), constraints, total_mines), True

    # Aggregate configurations, weighting by C(n_unconstrained, remaining)
    total_weight = 0
    mine_weight: Dict = defaultdict(float)

    for cfg, frontier_mines in valid_configs:
        remaining = total_mines - frontier_mines
        w = comb(n_unconstrained, remaining)
        total_weight += w
        for cell, is_mine in cfg.items():
            if is_mine:
                mine_weight[cell] += w
        if n_unconstrained > 0 and remaining > 0:
            unc_contrib = w * remaining / n_unconstrained
            for cell in unconstrained:
                mine_weight[cell] += unc_contrib

    if total_weight == 0:
        return _local_fallback(sorted(hidden_set), constraints, total_mines), True

    posteriors: Dict = {}
    for cell in hidden_set:
        posteriors[cell] = mine_weight[cell] / total_weight
    return posteriors, False


def _local_fallback(
    hidden: List[Tuple[int, int]],
    constraints: List[Tuple[int, List[Tuple[int, int]]]],
    total_mines: int,
) -> Dict[Tuple[int, int], float]:
    """Single-constraint local deduction: forced-mine and forced-safe cells."""
    forced_mine: set = set()
    forced_safe: set = set()

    for v, nbrs in constraints:
        hidden_in = list(nbrs)
        if v == len(hidden_in):
            forced_mine.update(hidden_in)
        elif v == 0:
            forced_safe.update(hidden_in)

    uncertain = [cell for cell in hidden if cell not in forced_mine and cell not in forced_safe]
    n_uncertain = len(uncertain)
    remaining = total_mines - len(forced_mine)
    default_prob = remaining / n_uncertain if n_uncertain > 0 else 0.5

    posteriors: Dict = {}
    for cell in hidden:
        if cell in forced_mine:
            posteriors[cell] = 1.0
        elif cell in forced_safe:
            posteriors[cell] = 0.0
        else:
            posteriors[cell] = max(0.0, min(1.0, default_prob))
    return posteriors


def get_oracle_actions(
    posteriors: Dict[Tuple[int, int], float],
    revealed: List[List[bool]],
    flags: List[List[bool]],
    rows: int,
    cols: int,
    eps: float = 1e-9,
) -> Tuple[List[str], float, float]:
    """Determine oracle-valid reveal and flag actions (1-indexed output)."""
    unrevealed_unflagged = [
        cell for cell in posteriors
        if not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]
    ]
    if not unrevealed_unflagged:
        return [], 0.0, 1.0

    probs = {cell: posteriors[cell] for cell in unrevealed_unflagged}
    min_prob = min(probs.values())

    oracle_actions: List[str] = []
    for cell, p in probs.items():
        if abs(p - min_prob) < eps:
            oracle_actions.append(f"reveal {cell[0]+1} {cell[1]+1}")

    # Flag oracle: posterior == 1.0 (exact, within eps)
    for cell, p in posteriors.items():
        if not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]:
            if p >= 1.0 - eps:
                oracle_actions.append(f"flag {cell[0]+1} {cell[1]+1}")

    return oracle_actions, min_prob, 1.0
