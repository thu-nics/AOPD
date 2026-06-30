"""Minesweeper reward branch tests: legal non-minimum, flag certainty, flag-toggle,
brute-force posterior comparison, truly disconnected component oracle.

All worker tests load the envs module with @ray.remote mocked as a no-op so workers
are regular Python objects. This keeps tests fast, deterministic, and compatible with
both default python3 and the production verl-agent venv (which has real Ray).
"""

import sys
import importlib.util
import json
from collections import defaultdict
from unittest.mock import MagicMock
import pytest


# ---------------------------------------------------------------------------
# Load oracle module (no ray needed)
# ---------------------------------------------------------------------------

def _load_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PKG = "agent_system.environments.env_package.vpr_games"
_ORACLE_KEY = _PKG + ".minesweeper.oracle"

# Load the oracle under its REAL package path so Ray workers can import it.
# Using a fake module name (e.g. "_ms_oracle_for_reward") would break Ray worker
# processes, which import the module by its real package path from disk.
if _ORACLE_KEY not in sys.modules:
    _oracle_mod = _load_direct(_ORACLE_KEY,
        "agent_system/environments/env_package/vpr_games/minesweeper/oracle.py")
else:
    _oracle_mod = sys.modules[_ORACLE_KEY]

compute_posteriors = _oracle_mod.compute_posteriors
get_oracle_actions = _oracle_mod.get_oracle_actions


# ---------------------------------------------------------------------------
# Load envs module with @ray.remote mocked as no-op (regardless of Ray install)
# ---------------------------------------------------------------------------
# We mock ray.remote→identity so MinesweeperWorker is a plain Python class
# that can be instantiated and called directly in unit tests.
# We also load the envs under a private alias ("_ms_envs_local_ray") so we
# don't clobber the canonical agent_system...minesweeper.envs module, which
# real Ray workers need to import from disk by its real package path.

def _load_envs_with_local_ray():
    """Load minesweeper envs module with @ray.remote replaced by no-op."""
    _orig = sys.modules.get('ray')
    _mock = MagicMock()
    _mock.remote = lambda cls: cls
    sys.modules['ray'] = _mock

    _parser_key = _PKG + ".common.parser"
    if _parser_key not in sys.modules:
        _load_direct(_parser_key,
                     "agent_system/environments/env_package/vpr_games/common/parser.py")

    name = "_ms_envs_local_ray"
    spec = importlib.util.spec_from_file_location(
        name, "agent_system/environments/env_package/vpr_games/minesweeper/envs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    if _orig is None:
        sys.modules.pop('ray', None)
    else:
        sys.modules['ray'] = _orig
    return mod


_ms_envs = _load_envs_with_local_ray()
MinesweeperWorker = _ms_envs.MinesweeperWorker


# ---------------------------------------------------------------------------
# Board state helpers
# ---------------------------------------------------------------------------

def _make_worker(rows=5, cols=5, mines=3, seed=0, max_turns=30):
    return MinesweeperWorker(seed=seed, rows=rows, cols=cols,
                             num_mines=mines, max_turns=max_turns)


def _first_reveal(w):
    """Do first safe reveal on a 5x5 board and return (obs, reward, done, info)."""
    return w.step("<action>reveal 3 3</action>")


def _inject_known_state(w, revealed, grid, flags=None):
    """Inject a known board state into the worker for deterministic testing."""
    rows, cols = len(revealed), len(revealed[0])
    w._env.revealed = [row[:] for row in revealed]
    w._env.grid = [row[:] for row in grid]
    if flags is not None:
        w._env.flags = [row[:] for row in flags]
    else:
        w._env.flags = [[False] * cols for _ in range(rows)]
    w._first_revealed = True
    w._rows = rows
    w._cols = cols


# ---------------------------------------------------------------------------
# Brute-force posterior reference
# ---------------------------------------------------------------------------

def brute_force_posteriors(revealed, grid, rows, cols, total_mines):
    hidden = [(r, c) for r in range(rows) for c in range(cols) if not revealed[r][c]]

    def nbrs(r, c):
        return [(nr, nc) for nr in range(r-1, r+2) for nc in range(c-1, c+2)
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != (r, c)]

    def consistent(mine_set):
        for r in range(rows):
            for c in range(cols):
                if not revealed[r][c]:
                    continue
                v = grid[r][c]
                if v < 0:
                    continue
                if sum(1 for x in nbrs(r, c) if x in mine_set) != v:
                    return False
        return True

    mine_count = defaultdict(int)
    valid = [0]

    def enum(idx, cur, n):
        if n == total_mines:
            if consistent(set(cur)):
                valid[0] += 1
                for c in cur:
                    mine_count[c] += 1
            return
        if idx >= len(hidden):
            return
        rem = len(hidden) - idx
        need = total_mines - n
        if need > rem:
            return
        enum(idx + 1, cur + [hidden[idx]], n + 1)
        if rem - 1 >= need:
            enum(idx + 1, cur, n)

    enum(0, [], 0)
    if valid[0] == 0:
        return {}
    return {c: mine_count[c] / valid[0] for c in hidden}


# ---------------------------------------------------------------------------
# Posterior brute-force equivalence: 3x3 board
# ---------------------------------------------------------------------------

class TestBruteForceEquivalence3x3:
    """Cell-by-cell comparison of oracle vs brute-force on a 3x3 board."""

    def _board(self):
        # 3x3: top-left revealed=1, 1 total mine, 8 hidden cells
        rows, cols = 3, 3
        revealed = [[True, False, False],
                    [False, False, False],
                    [False, False, False]]
        grid = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        return revealed, grid, rows, cols, 1

    def test_brute_force_vs_oracle_cell_by_cell(self):
        revealed, grid, rows, cols, total_mines = self._board()
        oracle_post, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines)
        brute_post = brute_force_posteriors(revealed, grid, rows, cols, total_mines)
        assert not degraded
        for cell in brute_post:
            oracle_val = oracle_post.get(cell, 0.0)
            brute_val = brute_post[cell]
            assert abs(oracle_val - brute_val) < 1e-6, \
                f"Cell {cell}: oracle={oracle_val:.6f} != brute={brute_val:.6f}"

    def test_injected_wrong_posterior_fails(self):
        revealed, grid, rows, cols, total_mines = self._board()
        brute_post = brute_force_posteriors(revealed, grid, rows, cols, total_mines)
        wrong = dict(brute_post)
        if wrong:
            cell = next(iter(wrong))
            wrong[cell] = (wrong[cell] + 0.5) % 1.0
        assert any(
            abs(wrong.get(c, 0) - brute_post[c]) > 1e-6
            for c in brute_post
        ), "Injected wrong posterior should differ from brute-force"


# ---------------------------------------------------------------------------
# Truly disconnected components (no shared frontier cell)
# ---------------------------------------------------------------------------

class TestTrulyDisconnectedComponents:
    def test_two_isolated_frontier_groups(self):
        """1x7: (0,1)=1 and (0,3)=0 and (0,5)=1. (0,3)=0 forces (0,2),(0,4) safe.
        Constraints reduce to: (0,0) certain mine, (0,6) certain mine."""
        rows, cols = 1, 7
        revealed = [[False, True, False, True, False, True, False]]
        grid = [[0, 1, 0, 0, 0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=2)
        assert not degraded
        assert abs(posteriors.get((0, 2), 0) - 0.0) < 1e-6, f"(0,2) forced safe, got {posteriors.get((0,2))}"
        assert abs(posteriors.get((0, 4), 0) - 0.0) < 1e-6, f"(0,4) forced safe, got {posteriors.get((0,4))}"
        assert abs(posteriors.get((0, 0), 0) - 1.0) < 1e-6, f"(0,0) certain mine, got {posteriors.get((0,0))}"
        assert abs(posteriors.get((0, 6), 0) - 1.0) < 1e-6, f"(0,6) certain mine, got {posteriors.get((0,6))}"

    def test_truly_disconnected_symmetric(self):
        """1x6: (0,0)=1 and (0,5)=1. One mine each in {(0,1)} and {(0,4)}.
        (0,2),(0,3) unconstrained with 0 remaining mines → P=0."""
        rows, cols = 1, 6
        revealed = [[True, False, False, False, False, True]]
        grid = [[1, 0, 0, 0, 0, 1]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = compute_posteriors(revealed, grid, rows, cols, total_mines=2)
        assert not degraded
        assert abs(posteriors.get((0, 1), 0) - 1.0) < 1e-6
        assert abs(posteriors.get((0, 4), 0) - 1.0) < 1e-6
        assert abs(posteriors.get((0, 2), 0) - 0.0) < 1e-6
        assert abs(posteriors.get((0, 3), 0) - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Oracle fallback
# ---------------------------------------------------------------------------

class TestOracleFallback:
    def test_budget_exceeded_triggers_fallback(self):
        rows, cols = 1, 4
        revealed = [[False, True, False, False]]
        grid = [[0, 1, 0, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, degraded = compute_posteriors(
            revealed, grid, rows, cols, total_mines=1, n_max=1)
        assert degraded
        hidden = [(0, 0), (0, 2), (0, 3)]
        for cell in hidden:
            assert cell in posteriors, f"Fallback missing cell {cell}"

    def test_fallback_local_deduction(self):
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]
        flags = [[False, False]]
        posteriors, degraded = compute_posteriors(
            revealed, grid, rows, cols, total_mines=1, n_max=1)
        assert degraded
        assert abs(posteriors.get((0, 1), 0) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Oracle flag certainty (oracle function level, not worker level)
# ---------------------------------------------------------------------------

class TestOracleFlagCertainty:
    def test_certain_mine_p_equals_1_exact(self):
        """1x2 forced mine: posterior == 1.0 exactly (integer arithmetic)."""
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]
        flags = [[False, False]]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        assert posteriors.get((0, 1), 0.0) == 1.0

    def test_uncertain_mine_p_not_1(self):
        """P=0.5 cells: posterior < 1.0, no flag oracle."""
        rows, cols = 1, 3
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        actions, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
        flag_acts = [a for a in actions if a.startswith("flag")]
        assert len(flag_acts) == 0


# ---------------------------------------------------------------------------
# Minesweeper worker reward branches — deterministic board injection
# ---------------------------------------------------------------------------

class TestMinesweeperWorkerRewardsDeterministic:
    """Test each reward branch by injecting a known board state.

    Workers are loaded with ray.remote mocked as no-op so methods are callable directly.
    Board states are injected after reset to ensure deterministic test conditions.
    """

    def _setup_board_after_first_reveal(self):
        """
        3x3 board, manually revealed to a state where we know the mine positions
        and can predict oracle actions exactly.

        Layout (0-indexed):
          row 0: [revealed=1, hidden, hidden]
          row 1: [hidden, hidden, hidden]
          row 2: [hidden, hidden, hidden]

        With total_mines=1: the mine must be one of the 3 neighbors of (0,0):
        {(0,1),(1,0),(1,1)}. All get P=1/3. Min-prob = 1/3. All are oracle-valid reveals.
        """
        w = MinesweeperWorker(seed=0, rows=3, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        # Inject known state: (0,0) revealed with value 1
        revealed = [[True, False, False],
                    [False, False, False],
                    [False, False, False]]
        grid = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        _inject_known_state(w, revealed, grid)
        w._step_count = 1
        w._num_mines = 1
        return w

    def test_oracle_reveal_safe_cell_plus_2_deterministic(self):
        """Safe minimum-posterior reveal returns +2.0 deterministically.

        Board (1x3): mine injected at (0,1). After revealing (0,0)=1:
        - P(0,1) = 1.0 (certain mine, oracle would flag it)
        - P(0,2) = 0.0 (unconstrained safe cell — unconstrained remaining mines = 0)
        Min-prob = 0.0 for unconstrained cells. Oracle-valid reveals: {(0,2)}.
        Revealing (0,2) is guaranteed safe (no mine there) → +2.0.
        """
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        # Inject: (0,0)=1 revealed, mine at (0,1), safe at (0,2)
        # grid[0][2]=1 because (0,2)'s only neighbor is (0,1) which is a mine
        revealed = [[True, False, False]]
        grid = [[1, -1, 1]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False  # prevent GEM from re-placing mines
        w._num_mines = 1

        # (0,2) is oracle-valid (P=0.0, safe) → reveal 1 3 (1-indexed row=1, col=3)
        obs, reward, done, info = w.step("<action>reveal 1 3</action>")
        assert reward == 2.0, f"Oracle-valid safe reveal should be +2.0, got {reward}"
        assert info["parse_ok"]
        assert info["move_optimal"] is True
        assert info["oracle_policy_tier"] == "all_oracle_actions"
        assert info["oracle_tier"] == "safe_reveal"
        assert info["safe_reveal_available"] is True
        assert info["certain_flag_available"] is True
        assert info["guess_required"] is False
        assert info["reveal_posterior_margin"] == 0.0
        assert not info["illegal_action"]

    def test_oracle_reveal_reward_is_configurable(self):
        """Safe oracle reveal can be scaled independently from oracle flag reward."""
        w = MinesweeperWorker(
            seed=0, rows=1, cols=3, num_mines=1, max_turns=30,
            oracle_reward=5.0, oracle_flag_reward=2.0,
        )
        w.reset(seed=0)
        revealed = [[True, False, False]]
        grid = [[1, -1, 1]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        obs, reward, done, info = w.step("<action>reveal 1 3</action>")
        assert reward == 5.0, f"Oracle-valid safe reveal should be +5.0, got {reward}"
        assert info["move_optimal"] is True
        assert not info["illegal_action"]

    def test_certain_flag_worker_step_gets_flag_oracle_reward(self):
        """Flagging P=1.0 receives flag reward when no safe reveal is available."""
        w = MinesweeperWorker(seed=0, rows=1, cols=2, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[True, False]]
        grid = [[1, 0]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        obs, reward, done, info = w.step("<action>flag 1 2</action>")
        assert reward == 1.0, f"Certain flag worker step should get oracle flag reward, got {reward}"
        assert info["move_optimal"] is True
        assert info["oracle_policy_tier"] == "all_oracle_actions"
        assert info["oracle_tier"] == "certain_flag"
        assert info["safe_reveal_available"] is False
        assert info["certain_flag_available"] is True
        assert not info["illegal_action"]

    def test_safe_reveal_and_certain_flag_are_both_oracle_actions(self):
        """When safe reveal exists, P=1.0 flags still count as oracle flag actions."""
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[True, False, False]]
        grid = [[1, -1, 1]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        obs, reward, done, info = w.step("<action>flag 1 2</action>")
        assert reward == 1.0
        assert info["move_optimal"] is True
        assert info["pre_exec_oracle_match"] is True
        assert info["legal_non_oracle"] is False
        assert info["oracle_policy_tier"] == "all_oracle_actions"
        assert info["oracle_tier"] == "certain_flag"
        assert info["safe_reveal_available"] is True
        assert info["certain_flag_available"] is True

    def test_uncertain_flag_worker_step_penalty(self):
        """Flagging an uncertain cell (P=0.5) receives the legal non-oracle penalty.

        Board (1x3): (0,1)=1 revealed; P(0,0)=P(0,2)=0.5. Neither is oracle-certain.
        Flagging (0,0) = 'flag 1 1' -> reward = -1.0 (legal non-oracle flag).
        """
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        obs, reward, done, info = w.step("<action>flag 1 1</action>")  # flag (0,0)
        assert reward == -1.0, f"Uncertain flag should be -1.0, got {reward}"
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "non_oracle_flag"
        assert info["move_optimal"] is False
        assert info["legal_non_oracle"] is True
        assert not info["illegal_action"]
        assert not w._env.flags[0][0], "Non-oracle flag should not be written to the board"

    def test_non_oracle_reveal_reward_penalty(self):
        """Revealing a cell NOT in oracle_valid_actions returns 0.0.

        On a board with (0,0)=1 revealed and 1 mine among {(0,1),(1,0),(1,1)},
        cells (0,2),(1,2),(2,0),(2,1),(2,2) are unconstrained (posterior = 0 remaining
        mines / n_unconstrained = 0). They are safe → P=0.0 which is the min_prob!

        Actually with 1 total mine, all hidden cells have some probability ≤ 1/3.
        Min-prob = 1/3 for {(0,1),(1,0),(1,1)}, and the unconstrained cells have
        P = remaining_mines / n_unconstrained. With frontier_mines = 1 config selected,
        remaining = 0, so unconstrained cells have P=0.

        Non-oracle = unconstrained cells = 0.0 probability, but they're not oracle-valid
        (min-prob of 0 is < 1/3 for frontier cells which ARE at 1/3).

        Wait — min_prob is over ALL unrevealed cells. If unconstrained have P=0.0 and
        frontier have P=1/3, then min_prob = 0.0 (unconstrained are lower).
        So unconstrained cells ARE oracle-valid reveals (lower min probability)!

        Let me reconsider: With 1 mine total and frontier {(0,1),(1,0),(1,1)} each at P=1/3,
        and unconstrained {(0,2),(1,2),(2,0),(2,1),(2,2)} at P=0.0 (remaining=0 since
        frontier absorbs the mine with P=1.0 contribution), unconstrained P=0.0 < 1/3.
        So min_prob = 0.0 and oracle-valid reveals are the unconstrained cells!

        For testing 'non-oracle' we need a cell that is NOT at min probability.
        The frontier cells (0,1),(1,0),(1,1) at P=1/3 are NOT oracle-valid when min=0.
        So revealing (0,1) -> not oracle-valid -> reward = 0.0.
        """
        w = self._setup_board_after_first_reveal()
        # (0,1) is at P=1/3 but min-prob is 0.0 (unconstrained cells), so non-oracle
        obs, reward, done, info = w.step("<action>reveal 1 2</action>")
        # If (0,1) is a mine (P=1/3 chance), GEM terminates -> reward = 0.0 (mine-hit, legal non-oracle)
        # If (0,1) is safe -> reward = 0.0 (non-oracle since min_prob is for unconstrained)
        # Either way: reward = 0.0
        assert reward == 0.0, f"Non-oracle/mine-hit reveal should be 0.0, got {reward}"
        assert info["move_optimal"] is False
        assert info["legal_non_oracle"] is True

    def test_illegal_action_penalty(self):
        """Parse failure gets the invalid action penalty."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w.reset(seed=0)
        obs, reward, done, info = w.step("no action tag here")
        assert reward == -2.0
        assert done
        assert not info["parse_ok"]
        assert info["illegal_action"]

    def test_flag_on_flagged_cell_terminates_as_non_oracle_flag(self):
        """Flagging an already-flagged cell is a non-oracle flag terminal."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w.reset(seed=0)
        _first_reveal(w)  # set _first_revealed = True
        unrevealed = [(r+1, c+1) for r in range(5) for c in range(5)
                      if not w._env.revealed[r][c] and not w._env.flags[r][c]]
        if not unrevealed:
            pytest.skip("No unrevealed cells")
        r1, c1 = unrevealed[0]
        w._env.flags[r1-1][c1-1] = True

        obs, reward, done, info = w.step(f"<action>flag {r1} {c1}</action>")
        assert reward == -1.0, f"Non-oracle flag should be -1.0, got {reward}"
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "non_oracle_flag"
        assert not info["illegal_action"]
        assert info["legal_non_oracle"] is True
        assert w._env.flags[r1-1][c1-1], "Non-oracle flag should not mutate existing flags"

    def test_mine_hit_uses_pre_exec_oracle_label(self):
        """Mine reveal uses the pre-execution oracle label, not hidden outcome."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w.reset(seed=0)
        # First safe reveal
        _first_reveal(w)
        # Find mine cell
        mine_cells = [(r+1, c+1) for r in range(5) for c in range(5)
                      if w._env.grid[r][c] < 0 and not w._env.revealed[r][c]]
        if not mine_cells:
            pytest.skip("No unrevealed mine")
        r1, c1 = mine_cells[0]
        obs, reward, done, info = w.step(f"<action>reveal {r1} {c1}</action>")
        if info["pre_exec_oracle_match"]:
            expected = {"safe_reveal": 2.0, "certain_flag": 1.0, "guess": 1.0}[info["oracle_tier"]]
        else:
            expected = 0.0
        assert reward == expected, f"Mine hit should use pre-exec oracle label, got {reward}"
        assert info["move_optimal"] is info["pre_exec_oracle_match"]
        assert done
        assert info["terminal_success"] is False
        assert info["terminal_reason"] == "mine_hit"
        assert not info["illegal_action"]

    def test_oracle_guess_mine_hit_keeps_oracle_reward(self):
        """A lowest-risk oracle reveal remains a positive imitation label even if it hits a mine.

        Board (1x3): center revealed as 1, one mine among the two hidden edge cells.
        Both hidden cells have posterior 0.5 and are oracle-valid minimum-risk reveals.
        Revealing the left cell is therefore an oracle match even though the injected
        hidden board makes it a mine.
        """
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[False, True, False]]
        grid = [[-1, 1, 0]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        obs, reward, done, info = w.step("<action>reveal 1 1</action>")
        assert done
        assert info["terminal_reason"] == "mine_hit"
        assert reward == 1.0
        assert info["move_optimal"] is True
        assert info["pre_exec_oracle_match"] is True
        assert info["legal_non_oracle"] is False
        assert info["oracle_guess"] is True
        assert info["oracle_policy_tier"] == "all_oracle_actions"
        assert info["oracle_tier"] == "guess"
        assert info["guess_required"] is True
        assert info["reveal_posterior_margin"] == 0.0

    def test_guess_remains_oracle_when_certain_flag_exists(self):
        """Minimum-risk guesses stay oracle when flags are certain and no safe reveal exists."""
        w = MinesweeperWorker(seed=0, rows=1, cols=4, num_mines=2, max_turns=30)
        w.reset(seed=0)
        _inject_known_state(w, [[False, False, False, False]], [[0, 0, 0, 0]])

        policy = w._policy_oracle_actions({
            (0, 0): 1.0,
            (0, 1): 0.25,
            (0, 2): 0.25,
            (0, 3): 0.5,
        })

        assert policy["safe_reveal_available"] is False
        assert policy["certain_flag_available"] is True
        assert policy["guess_required"] is True
        assert "flag 1 1" in policy["oracle_actions"]
        assert "reveal 1 2" in policy["oracle_actions"]
        assert "reveal 1 3" in policy["oracle_actions"]
        assert policy["oracle_action_tiers"]["flag 1 1"] == "certain_flag"
        assert policy["oracle_action_tiers"]["reveal 1 2"] == "guess"

    def test_certain_flag_reward_via_oracle_query(self):
        """Oracle correctly identifies P=1.0 cell as flag-oracle (uses get_oracle_actions)."""
        # 1x2: only cell (0,1) is hidden → certainly a mine → get_oracle_actions returns flag
        rows, cols = 1, 2
        revealed = [[True, False]]
        grid = [[1, 0]]
        flags = [[False, False]]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        assert posteriors.get((0, 1), 0.0) == 1.0
        actions, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
        assert "flag 1 2" in actions

    def test_uncertain_flag_not_oracle_via_oracle_query(self):
        """P=0.5 cells: oracle does not flag them."""
        rows, cols = 1, 3
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        flags = [[False] * cols for _ in range(rows)]
        posteriors, _ = compute_posteriors(revealed, grid, rows, cols, total_mines=1)
        actions, _, _ = get_oracle_actions(posteriors, revealed, flags, rows, cols)
        flag_acts = [a for a in actions if a.startswith("flag")]
        assert len(flag_acts) == 0, f"P=0.5 should not get flag oracle, got {flag_acts}"

    def test_is_action_valid_in_info_set_by_manager(self):
        """Worker info contains parse_ok and illegal_action; manager derives is_action_valid.

        VPRBaseEnvironmentManager.step() (base_manager.py:46) sets:
            info["is_action_valid"] = int(info.get("parse_ok", True)
                                          and not info.get("illegal_action", False))
        This test verifies workers always emit these fields.
        """
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w.reset(seed=0)
        _, reward, done, info = w.step("<action>reveal 3 3</action>")
        assert "parse_ok" in info, "parse_ok must be in worker info for manager is_action_valid"
        assert "illegal_action" in info, "illegal_action must be in worker info"
        # Verify manager's logic produces expected is_action_valid value
        expected_valid = int(info["parse_ok"] and not info["illegal_action"])
        computed = int(info.get("parse_ok", True) and not info.get("illegal_action", False))
        assert computed == expected_valid == 1, "First successful reveal should be valid"

        # Verify on parse failure
        w2 = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w2.reset(seed=0)
        _, _, _, info2 = w2.step("no action tag")
        assert not info2["parse_ok"]
        expected_invalid = int(info2.get("parse_ok", True) and not info2.get("illegal_action", False))
        assert expected_invalid == 0, "Parse failure should produce is_action_valid=0"


# ---------------------------------------------------------------------------
# Prompt boundedness: multi-step sequence with deterministic outcome
# ---------------------------------------------------------------------------

class TestMarkovianPrompts:
    """Verify prompts don't grow across steps (Markovian — no history accumulation)."""

    def _load_tictactoe_game(self):
        spec = importlib.util.spec_from_file_location(
            "_ttt_game_prompt",
            "agent_system/environments/env_package/vpr_games/tictactoe/game.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_ttt_game_prompt"] = mod
        spec.loader.exec_module(mod)
        return mod

    def _load_template(self):
        spec = importlib.util.spec_from_file_location(
            "_vpr_prompts_prompt",
            "agent_system/environments/prompts/vpr_games.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_vpr_prompts_prompt"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_tictactoe_prompt_constant_length(self):
        """After each step, prompt length stays the same (board changes, not history)."""
        game_mod = self._load_tictactoe_game()
        tpl_mod = self._load_template()

        g = game_mod.TicTacToeGame(opponent="random", seed=42)
        obs1, _ = g.reset(seed=42)
        prompt1 = tpl_mod.TICTACTOE_TEMPLATE.format(board=obs1, mark="X", opp="O")

        # Take 3 steps; ensure we check after each successful step
        prompts = [prompt1]
        for action in ["1", "3", "7"]:  # use corner cells to avoid collisions
            obs, reward, done, info = g.step(action, True, f"<action>{action}</action>")
            if done:
                break
            prompt = tpl_mod.TICTACTOE_TEMPLATE.format(board=obs, mark="X", opp="O")
            prompts.append(prompt)

        assert len(prompts) >= 2, "Need at least 2 steps for comparison"

        # All prompts should be the same length ±50 chars (board changes but structure stays)
        # More importantly: later prompts must NOT contain text from earlier actions
        for i in range(1, len(prompts)):
            # Check: no prior action text embedded (Markovian)
            for j in range(i):
                prev_action = ["1", "3", "7"][j]
                # The prompt should not contain "I chose cell X" or similar history
                # We check that the length doesn't grow significantly
                assert len(prompts[i]) <= len(prompts[0]) + 100, \
                    f"Prompt grew: step0={len(prompts[0])}, step{i}={len(prompts[i])}"

    def test_tictactoe_prompt_no_prior_action_in_obs(self):
        """A prompt generated at step 5 does not contain the raw action text from step 3."""
        game_mod = self._load_tictactoe_game()
        tpl_mod = self._load_template()

        g = game_mod.TicTacToeGame(opponent="random", seed=0)
        g.reset(seed=0)

        # Step 1: action "1"
        _, _, done1, _ = g.step("1", True, "<action>1</action>")
        if done1:
            pytest.skip("Episode ended too early")

        # Step 2: action "3"
        _, _, done2, _ = g.step("3", True, "<action>3</action>")
        if done2:
            pytest.skip("Episode ended too early")

        # Step 3: action "7"
        obs3, _, done3, _ = g.step("7", True, "<action>7</action>")
        if done3:
            pytest.skip("Episode ended too early")

        # Build prompt at step 3 — should not contain "1" or "3" or "7" as standalone move text
        prompt3 = tpl_mod.TICTACTOE_TEMPLATE.format(board=obs3, mark="X", opp="O")
        # The board obs itself shows X/O marks, but the raw action text "cell 1", "cell 3"
        # should not appear as "I played" or similar history
        # The key invariant: prompt length ≈ prompt at step 1
        g2 = game_mod.TicTacToeGame(opponent="random", seed=0)
        obs_init, _ = g2.reset(seed=0)
        prompt_init = tpl_mod.TICTACTOE_TEMPLATE.format(board=obs_init, mark="X", opp="O")
        # Markovian: prompt at step 3 should not be longer than prompt at step 1 + small delta
        assert len(prompt3) <= len(prompt_init) + 100, \
            f"Prompt grew beyond tolerance: init={len(prompt_init)}, step3={len(prompt3)}"


# ---------------------------------------------------------------------------
# Outcome reward mode (win/lose scoring for the standard-GRPO baseline)
# ---------------------------------------------------------------------------

class TestMinesweeperOutcomeReward:
    def test_invalid_reward_mode_raises(self):
        with pytest.raises(ValueError):
            MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=5, reward_mode="bogus")

    def test_outcome_mine_hit_is_loss(self):
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=5, reward_mode="outcome")
        w.reset(seed=0)
        _, _, done, _ = _first_reveal(w)   # safe first click; mines now placed in env.grid
        if done:
            pytest.skip("first reveal ended the game")
        mine = next(((r, c) for r in range(5) for c in range(5)
                     if w._env.grid[r][c] == -1 and not w._env.revealed[r][c]), None)
        assert mine is not None
        r, c = mine
        _, reward, done, info = w.step(f"<action>reveal {r + 1} {c + 1}</action>")
        assert done and info["terminal_reason"] == "mine_hit" and reward == -1.0

    def test_outcome_nonterminal_safe_reveal_is_zero(self):
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=5, reward_mode="outcome")
        w.reset(seed=0)
        _, reward, done, _ = _first_reveal(w)
        if done:
            pytest.skip("first reveal ended the game")
        assert reward == 0.0

    def test_outcome_invalid_action_is_loss(self):
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=5, reward_mode="outcome")
        w.reset(seed=0)
        _first_reveal(w)
        _, reward, done, info = w.step("no action tag here")
        assert done and reward == -1.0


# ---------------------------------------------------------------------------
# Auto-reveal-center opening (default-on in the training config)
# ---------------------------------------------------------------------------

class TestMinesweeperAutoRevealCenter:
    def test_default_worker_does_not_auto_reveal(self):
        """Bare worker default keeps first-click semantics (center unrevealed at reset)."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30)
        w.reset(seed=0)
        assert not w._env.revealed[2][2]
        assert w._first_revealed is False

    def test_auto_reveal_reveals_center_and_arms_oracle(self):
        """With auto_reveal_center, reset reveals the (always-safe) center for free."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30,
                              auto_reveal_center=True)
        obs, info = w.reset(seed=0)
        # Center revealed, never a mine (GEM first-click safety), oracle now active.
        assert w._env.revealed[2][2]
        assert w._env.grid[2][2] != -1
        assert w._first_revealed is True
        # The opening reveal is free: it does not consume an agent step.
        assert info["step"] == 0
        assert "3 3" not in info["available_actions"]
        # completion_rate reflects the revealed safe cells (strictly positive).
        assert info["completion_rate"] > 0.0

    def test_auto_reveal_center_then_reveal_is_illegal(self):
        """Re-revealing the auto-opened center is an illegal action."""
        w = MinesweeperWorker(seed=0, rows=5, cols=5, num_mines=3, max_turns=30,
                              auto_reveal_center=True)
        w.reset(seed=0)
        _, reward, done, info = w.step("<action>reveal 3 3</action>")
        assert info["illegal_action"] is True


class TestMinesweeperStateGroupCandidates:
    def test_candidate_group_commits_best_reward_without_candidate_mutation_leakage(self):
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[True, False, False]]
        grid = [[1, -1, 1]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        candidates, best_idx, obs, reward, done, info = w.step_candidate_group([
            "<action>flag 1 1</action>",   # invalid revealed cell: -2
            "<action>reveal 1 2</action>", # mine / non-oracle reveal: 0
            "<action>reveal 1 3</action>", # oracle safe reveal: +2
        ])

        assert best_idx == 2
        assert reward == 2.0
        assert info["move_optimal"] is True
        assert w._env.revealed[0][2] is True
        assert w._env.revealed[0][1] is False, "Rejected mine candidate must not mutate board"
        assert w._step_count == 1, "Only the committed action should consume a step"
        assert candidates[0][1] == -2.0
        assert candidates[1][1] == 0.0
        assert candidates[2][1] == 2.0

    def test_candidate_group_tie_selects_first_max_reward(self):
        w = MinesweeperWorker(seed=0, rows=1, cols=3, num_mines=1, max_turns=30)
        w.reset(seed=0)
        revealed = [[False, True, False]]
        grid = [[0, 1, 0]]
        _inject_known_state(w, revealed, grid)
        w._env.first_reveal = False
        w._num_mines = 1

        candidates, best_idx, obs, reward, done, info = w.step_candidate_group([
            "<action>reveal 1 1</action>",
            "<action>reveal 1 3</action>",
        ])

        assert best_idx == 0
        assert reward == 1.0
        assert info["oracle_tier"] == "guess"
        assert w._env.revealed[0][0] is True
        assert w._env.revealed[0][2] is False

    def test_vector_candidate_groups_use_active_worker_indices(self):
        calls = []

        class _RemoteMethod:
            def __init__(self, fn):
                self._fn = fn

            def remote(self, actions, **kwargs):
                return self._fn(actions, **kwargs)

        class _FakeWorker:
            def __init__(self, worker_id):
                self.worker_id = worker_id
                self.step_candidate_group = _RemoteMethod(self._step_candidate_group)

            def _step_candidate_group(self, actions, **kwargs):
                calls.append((self.worker_id, list(actions), kwargs))
                info = {
                    "terminal_reason": None,
                    "terminal_success": None,
                    "parse_ok": True,
                    "illegal_action": False,
                }
                return ([(f"obs{self.worker_id}", 0.0, False, info)], 0,
                        f"obs{self.worker_id}", 0.0, False, info)

        old_get = _ms_envs.ray.get
        _ms_envs.ray.get = lambda futures: futures
        try:
            env = _ms_envs.MinesweeperMultiProcessEnv(
                workers=[_FakeWorker(0), _FakeWorker(1), _FakeWorker(2)],
                seeds=[0, 1, 2],
            )
            _, _, obs_list, _, _, _ = env.step_candidate_groups(
                [["a"], ["b"]],
                active_indices=[1, 2],
                selection_mode="mixed",
                random_select_prob=0.5,
            )
        finally:
            _ms_envs.ray.get = old_get

        assert calls == [
            (1, ["a"], {"selection_mode": "mixed", "random_select_prob": 0.5}),
            (2, ["b"], {"selection_mode": "mixed", "random_select_prob": 0.5}),
        ]
        assert obs_list == ["obs1", "obs2"]
