"""Minesweeper posterior mine probability oracle.

Uses connected-component frontier decomposition: the frontier (hidden cells
adjacent to revealed numbered cells) is partitioned into independent connected
components. Each component is enumerated separately under a shared N_MAX_CONFIGS
budget, then combined across components via DP-style mine-count convolution,
weighted by C(n_unconstrained, remaining_mines_after_frontier).

Exact integer counts are maintained throughout for frontier cells, enabling an
exact flag-certainty test (mine_count == total_weight).

Falls back to local single-constraint deduction when the budget is exceeded.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from math import comb
from typing import Dict, List, Tuple

N_MAX_CONFIGS = 50_000


def _neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    return [
        (nr, nc)
        for nr in range(r - 1, r + 2)
        for nc in range(c - 1, c + 2)
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != (r, c)
    ]


def _find_components(
    frontier: List[Tuple[int, int]],
    constraints: List[Tuple[int, List[Tuple[int, int]]]],
) -> List[List[Tuple[int, int]]]:
    """Partition frontier cells into connected components.

    Two frontier cells are in the same component if they share at least one
    constraint (i.e., appear together in a revealed cell's neighborhood).
    Uses union-find for efficiency.
    """
    frontier_set = set(frontier)
    parent: Dict[Tuple, Tuple] = {c: c for c in frontier}

    def find(x: Tuple) -> Tuple:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: Tuple, y: Tuple) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _, nbrs in constraints:
        front_in_constraint = [c for c in nbrs if c in frontier_set]
        for i in range(1, len(front_in_constraint)):
            union(front_in_constraint[0], front_in_constraint[i])

    # Group by root
    groups: Dict[Tuple, List] = defaultdict(list)
    for cell in frontier:
        groups[find(cell)].append(cell)

    return [sorted(cells) for cells in groups.values()]


def _enumerate_component(
    comp: List[Tuple[int, int]],
    constraints: List[Tuple[int, List[Tuple[int, int]]]],
    frontier_set: set,
    steps_taken: List[int],
    n_max: int,
) -> Optional[List[Tuple[int, Dict[Tuple, int]]]]:
    """Enumerate all valid mine assignments for a single frontier component.

    Args:
        comp: sorted list of frontier cells in this component.
        constraints: all constraints (only those involving comp cells are used).
        frontier_set: full set of frontier cells (to filter constraints).
        steps_taken: shared mutable counter for budget tracking.
        n_max: maximum enumeration steps.

    Returns:
        List of (mine_count, assignment) pairs, or None if budget exceeded.
        assignment maps cell -> 0 or 1.
    """
    comp_set = set(comp)

    # Constraints relevant to this component
    relevant = [
        (v, [c for c in nbrs if c in comp_set])
        for v, nbrs in constraints
        if any(c in comp_set for c in nbrs)
    ]
    # Separate: constraint_req is total mines required; constraint_nbrs are the subset
    # of the constraint's neighbors that belong to this component.
    # Constraints may also span cells NOT in this component (other frontier cells or
    # already assigned cells). We only track the sub-count within this component.
    #
    # To handle cross-component constraints correctly, we store the full-nbr list and
    # component-nbr list separately, and only verify when the component is fully assigned.
    # Note: since components are CONNECTED, a constraint either involves only this
    # component's cells or also cells in other components. For now we enumerate this
    # component's cells and record partial counts; cross-component constraints are
    # verified at the aggregation stage (they're rare and only matter for global feasibility).
    #
    # Simpler: treat each component's constraints as self-contained (cells outside the
    # component contribute 0 mines to this component's sub-constraint). This is valid
    # when components are truly disconnected — shared constraints would have merged them.

    # Build index: comp cell -> which relevant constraints involve it
    cell_to_ci = defaultdict(list)
    for ci, (_, nbrs) in enumerate(relevant):
        for cell in nbrs:
            cell_to_ci[cell].append(ci)

    constraint_mines = [0] * len(relevant)
    constraint_req = [v for v, _ in relevant]

    results: List[Tuple[int, Dict]] = []
    assignment: Dict[Tuple, int] = {}
    budget_exceeded = [False]

    def backtrack(idx: int, mine_count: int) -> None:
        if budget_exceeded[0]:
            return
        steps_taken[0] += 1
        if steps_taken[0] > n_max:
            budget_exceeded[0] = True
            return

        if idx == len(comp):
            # Verify all constraints satisfied
            for ci, req in enumerate(constraint_req):
                if constraint_mines[ci] != req:
                    return
            results.append((mine_count, dict(assignment)))
            return

        cell = comp[idx]
        affected = cell_to_ci[cell]
        for is_mine in (0, 1):
            feasible = True
            n_inc = 0
            for ci in affected:
                constraint_mines[ci] += is_mine
                n_inc += 1
                if constraint_mines[ci] > constraint_req[ci]:
                    feasible = False
                    break
            if feasible:
                assignment[cell] = is_mine
                backtrack(idx + 1, mine_count + is_mine)
                del assignment[cell]
            for ci in affected[:n_inc]:
                constraint_mines[ci] -= is_mine

    backtrack(0, 0)
    if budget_exceeded[0]:
        return None
    return results


def compute_posteriors(
    revealed: List[List[bool]],
    grid: List[List[int]],
    rows: int,
    cols: int,
    total_mines: int,
    n_max: int = N_MAX_CONFIGS,
) -> Tuple[Dict[Tuple[int, int], float], bool]:
    """Compute posterior mine probabilities for all hidden cells.

    Uses connected-component frontier decomposition with exact integer counts.

    Returns:
        (posteriors, oracle_degraded) where posteriors maps (r,c) -> P(mine).
        oracle_degraded=True when local fallback was used.
    """
    hidden_set = {
        (r, c) for r in range(rows) for c in range(cols) if not revealed[r][c]
    }
    if not hidden_set:
        return {}, False

    # Zero-value cells force hidden neighbors to be safe
    forced_safe_by_zero: set = set()
    constraints: List[Tuple[int, List[Tuple[int, int]]]] = []
    for r in range(rows):
        for c in range(cols):
            if not revealed[r][c]:
                continue
            v = grid[r][c]
            if v < 0:
                continue
            hidden_nbrs = [cell for cell in _neighbors(r, c, rows, cols) if cell in hidden_set]
            if not hidden_nbrs:
                continue
            if v == 0:
                forced_safe_by_zero.update(hidden_nbrs)
            else:
                constraints.append((v, hidden_nbrs))

    active_hidden = hidden_set - forced_safe_by_zero

    # Frontier: hidden cells appearing in at least one constraint
    frontier_set: set = set()
    for _, nbrs in constraints:
        frontier_set.update(c for c in nbrs if c in active_hidden)
    frontier = sorted(frontier_set)

    unconstrained = sorted(c for c in active_hidden if c not in frontier_set)
    n_unconstrained = len(unconstrained)

    if not frontier:
        # No frontier — all hidden cells are unconstrained
        if n_unconstrained == 0 or total_mines == 0:
            posteriors = {c: 0.0 for c in hidden_set}
        else:
            p = min(1.0, total_mines / n_unconstrained)
            posteriors = {c: p for c in unconstrained}
        for c in forced_safe_by_zero:
            posteriors[c] = 0.0
        return posteriors, False

    # Connected component decomposition
    components = _find_components(frontier, constraints)

    # Per-component enumeration with shared budget
    steps_taken = [0]
    comp_results: List[List[Tuple[int, Dict]]] = []
    for comp in components:
        result = _enumerate_component(comp, constraints, frontier_set, steps_taken, n_max)
        if result is None:
            return _local_fallback(sorted(active_hidden), constraints, total_mines,
                                   forced_safe_by_zero), True
        comp_results.append(result)

    if not all(comp_results):
        return _local_fallback(sorted(active_hidden), constraints, total_mines,
                               forced_safe_by_zero), True

    # Cross-component aggregation: integer exact counts for frontier cells
    # For each combination of per-component assignments, weight by C(n_unc, remaining)
    mine_count_int: Dict[Tuple, int] = defaultdict(int)   # exact for frontier cells
    mine_count_float: Dict[Tuple, float] = defaultdict(float)  # approx for unconstrained
    total_weight: int = 0

    for combo in itertools.product(*comp_results):
        # combo[i] = (mine_count_i, assignment_i) for component i
        frontier_mines = sum(mc for mc, _ in combo)
        remaining = total_mines - frontier_mines
        if remaining < 0 or remaining > n_unconstrained:
            continue
        w = comb(n_unconstrained, remaining)  # exact integer
        total_weight += w
        for _, assignment in combo:
            for cell, is_mine in assignment.items():
                if is_mine:
                    mine_count_int[cell] += w
        if n_unconstrained > 0 and remaining > 0:
            # Unconstrained cells each carry equal probability
            unc_mine_per_cell = w * remaining  # will divide by n_unconstrained at end
            for cell in unconstrained:
                mine_count_float[cell] += unc_mine_per_cell

    if total_weight == 0:
        return _local_fallback(sorted(active_hidden), constraints, total_mines,
                               forced_safe_by_zero), True

    posteriors: Dict[Tuple, float] = {}
    for cell in frontier_set:
        posteriors[cell] = mine_count_int[cell] / total_weight
    for cell in unconstrained:
        posteriors[cell] = mine_count_float[cell] / (total_weight * n_unconstrained)
    for cell in forced_safe_by_zero:
        posteriors[cell] = 0.0
    return posteriors, False


def _local_fallback(
    hidden: List[Tuple[int, int]],
    constraints: List[Tuple[int, List[Tuple[int, int]]]],
    total_mines: int,
    externally_forced_safe: set = None,
) -> Dict[Tuple[int, int], float]:
    """Single-constraint local deduction fallback."""
    forced_mine: set = set()
    forced_safe: set = set(externally_forced_safe or [])

    for v, nbrs in constraints:
        hidden_in = [c for c in nbrs if c not in forced_safe]
        if v == len(hidden_in):
            forced_mine.update(hidden_in)
        elif v == 0:
            forced_safe.update(hidden_in)

    uncertain = [c for c in hidden if c not in forced_mine and c not in forced_safe]
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
    if externally_forced_safe:
        for cell in externally_forced_safe:
            if cell not in posteriors:
                posteriors[cell] = 0.0
    return posteriors


def get_oracle_actions(
    posteriors: Dict[Tuple[int, int], float],
    revealed: List[List[bool]],
    flags: List[List[bool]],
    rows: int,
    cols: int,
) -> Tuple[List[str], float, float]:
    """Determine oracle-valid reveal and flag actions (1-indexed output).

    Flag oracle uses exact float equality (== 1.0). Frontier-cell posteriors
    are computed via exact integer division, so p == 1.0 iff mine_count == total_weight.
    """
    unrevealed_unflagged = [
        cell for cell in posteriors
        if not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]
    ]
    if not unrevealed_unflagged:
        return [], 0.0, 1.0

    probs = {cell: posteriors[cell] for cell in unrevealed_unflagged}
    min_prob = min(probs.values())

    oracle_actions: List[str] = []
    eps = 1e-9
    for cell, p in probs.items():
        if abs(p - min_prob) < eps:
            oracle_actions.append(f"reveal {cell[0]+1} {cell[1]+1}")

    # Flag oracle: exact certainty using float equality (exact because posteriors
    # for frontier cells are computed as mine_count_int / total_weight with integer ops)
    for cell in unrevealed_unflagged:
        if posteriors.get(cell, 0.0) == 1.0:
            oracle_actions.append(f"flag {cell[0]+1} {cell[1]+1}")

    return oracle_actions, min_prob, 1.0
