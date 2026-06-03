"""OpenSpiel MCTS opponent for VPR TicTacToe.

OpenSpiel (`pyspiel`) is imported lazily here so it is only a dependency when the
`mcts` opponent is actually selected. The opponent keeps a pyspiel `tic_tac_toe`
state in sync with the Python board: the game applies every move (the policy
model's and the opponent's) to this state as it happens, so the MCTS bot always
acts on the true current position and we never have to reconstruct move order.

Cell numbering: the VPR board uses 1-indexed cells (1..9); 0-indexed cell `idx`
maps directly to the OpenSpiel tic_tac_toe action id `idx` (= row*3 + col), with
player 0 = 'x' (X) and player 1 = 'o' (O).
"""

from __future__ import annotations

from typing import Optional


class OpenSpielMCTSOpponent:
    def __init__(self, max_simulations: int = 1000, uct_c: float = 2.0,
                 rollout_count: int = 1, seed: int = 0):
        try:
            import pyspiel  # noqa: F401
            from open_spiel.python.algorithms import mcts  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError(
                "opponent='mcts' requires OpenSpiel. Install it with: "
                "pip install open_spiel"
            ) from e
        import pyspiel
        self._pyspiel = pyspiel
        self._mcts = mcts
        self._max_simulations = max_simulations
        self._uct_c = uct_c
        self._rollout_count = rollout_count
        self._game = pyspiel.load_game("tic_tac_toe")
        self._bot = None
        self._state = None
        self.reset(seed)

    def _build_bot(self, seed: int):
        import numpy as np
        rs = np.random.RandomState(seed)
        evaluator = self._mcts.RandomRolloutEvaluator(self._rollout_count, rs)
        return self._mcts.MCTSBot(
            self._game, self._uct_c, self._max_simulations, evaluator,
            solve=True, random_state=rs,
        )

    def reset(self, seed: int = 0) -> None:
        """Start a fresh game and reseed the bot for deterministic play."""
        self._bot = self._build_bot(seed)
        self._state = self._game.new_initial_state()

    def apply_cell(self, idx: int) -> None:
        """Apply a move (0-indexed cell == pyspiel action id) to the synced state."""
        self._state.apply_action(int(idx))

    def choose(self) -> Optional[int]:
        """Return the MCTS-chosen 0-indexed cell for the current player, or None."""
        if self._state.is_terminal():
            return None
        return int(self._bot.step(self._state))
