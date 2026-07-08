"""TicTacToe game logic with exact minimax oracle.

Cells are 1-indexed (1..9), laid out row-major:
  1 | 2 | 3
  4 | 5 | 6
  7 | 8 | 9

The agent always plays as 'X'; the opponent plays as 'O'.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

_EMPTY = ""
_AGENT = "X"
_OPPONENT = "O"

_LINES: List[Tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def _check_winner(board: List[str]) -> Optional[str]:
    for a, b, c in _LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def _is_full(board: List[str]) -> bool:
    return all(c != _EMPTY for c in board)


def _minimax(board: List[str], is_agent_turn: bool, alpha: int, beta: int,
             agent_mark: str = _AGENT, opp_mark: str = _OPPONENT) -> int:
    """Exact minimax value from the agent's perspective (+1 agent win, -1 loss, 0 draw).

    `agent_mark`/`opp_mark` make the side configurable; they default to X (agent) / O
    (opponent) so existing callers are unchanged.
    """
    winner = _check_winner(board)
    if winner == agent_mark:
        return 1
    if winner == opp_mark:
        return -1
    if _is_full(board):
        return 0

    if is_agent_turn:
        best = -2
        for i in range(9):
            if board[i] == _EMPTY:
                board[i] = agent_mark
                val = _minimax(board, False, alpha, beta, agent_mark, opp_mark)
                board[i] = _EMPTY
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        return best
    else:
        best = 2
        for i in range(9):
            if board[i] == _EMPTY:
                board[i] = opp_mark
                val = _minimax(board, True, alpha, beta, agent_mark, opp_mark)
                board[i] = _EMPTY
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best


def oracle_valid_actions(board: List[str], agent_mark: str = _AGENT,
                         opp_mark: str = _OPPONENT) -> List[str]:
    """Return 1-indexed cell indices of minimax-optimal moves for `agent_mark`."""
    empty_cells = [i for i in range(9) if board[i] == _EMPTY]
    if not empty_cells:
        return []

    scores = []
    for i in empty_cells:
        board[i] = agent_mark
        s = _minimax(board, False, -2, 2, agent_mark, opp_mark)
        board[i] = _EMPTY
        scores.append(s)

    best = max(scores)
    return [str(empty_cells[j] + 1) for j, s in enumerate(scores) if s == best]


def _random_opponent_move(board: List[str], rng: random.Random) -> Optional[int]:
    """Return 0-indexed cell for opponent's random move, or None if no moves."""
    choices = [i for i in range(9) if board[i] == _EMPTY]
    return rng.choice(choices) if choices else None


class TicTacToeGame:
    """Single-instance TicTacToe game with minimax oracle and configurable opponent."""

    def __init__(self, opponent: str = "random", invalid_action_terminates: bool = True,
                 max_steps: int = 9, invalid_penalty: float = -1.0, seed: int = 0,
                 reward_mode: str = "oracle", agent_player: str = "X",
                 mcts_max_simulations: int = 1000, mcts_uct_c: float = 2.0,
                 mcts_rollout_count: int = 1):
        # reward_mode:
        #   "oracle"  — dense per-step VPR oracle reward (minimax-optimal move +1,
        #               legal non-optimal 0); used by VPR per-turn advantage.
        #   "outcome" — sparse win/lose reward: legal non-terminal moves get 0, and the
        #               terminal step gets +1 (agent win) / -1 (loss) / 0 (draw or
        #               step-limit). Use this for a standard-GRPO outcome-reward baseline.
        # agent_player: which mark the policy model controls ("X" moves first, "O" second).
        # opponent: "random" or "mcts" (OpenSpiel C++ MCTS; imported lazily).
        if reward_mode not in ("oracle", "outcome"):
            raise ValueError(f"reward_mode must be 'oracle' or 'outcome', got {reward_mode!r}")
        if agent_player not in ("X", "O"):
            raise ValueError(f"agent_player must be 'X' or 'O', got {agent_player!r}")
        if opponent not in ("random", "mcts"):
            raise ValueError(f"opponent must be 'random' or 'mcts', got {opponent!r}")
        self._opponent = opponent
        self._invalid_terminates = invalid_action_terminates
        self._max_steps = max_steps
        self._invalid_penalty = invalid_penalty
        self._reward_mode = reward_mode
        self._agent_mark = agent_player
        self._opponent_mark = "O" if agent_player == "X" else "X"
        self._seed = seed
        self._rng = random.Random(seed)
        self._board: List[str] = [_EMPTY] * 9
        self._step_count: int = 0
        self._done: bool = False
        self._game_result: str = "ongoing"
        self._last_opponent_action: Optional[str] = None
        # OpenSpiel MCTS opponent (lazy: only constructed/imported when selected).
        self._mcts = None
        if opponent == "mcts":
            from agent_system.environments.env_package.vpr_games.tictactoe.mcts_opponent import (
                OpenSpielMCTSOpponent,
            )
            self._mcts = OpenSpielMCTSOpponent(
                max_simulations=mcts_max_simulations, uct_c=mcts_uct_c,
                rollout_count=mcts_rollout_count, seed=seed,
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict]:
        s = seed if seed is not None else self._seed
        self._rng = random.Random(s)
        self._board = [_EMPTY] * 9
        self._step_count = 0
        self._done = False
        self._game_result = "ongoing"
        self._last_opponent_action = None
        if self._mcts is not None:
            self._mcts.reset(s)
        # If the policy model plays second (O), the opponent (X) makes the opening move.
        if self._agent_mark == "O":
            self._opponent_move()
        obs = self._render()
        info = self._build_info(
            raw_action="", parsed_action=None, parse_ok=True,
            illegal_action=False, vpr_reward=0.0,
            terminal_success=None, terminal_reason=None,
        )
        return obs, info


    def _sync_mcts_from_board(self) -> None:
        """Rebuild the OpenSpiel state from the Python board after restore()."""
        if self._mcts is None:
            return
        self._mcts.reset(self._seed)
        board = list(self._board)
        while True:
            x_cells = [i for i, mark in enumerate(board) if mark == "X"]
            o_cells = [i for i, mark in enumerate(board) if mark == "O"]
            if not x_cells and not o_cells:
                break
            next_mark = "X" if len(x_cells) > len(o_cells) else "O"
            cells = x_cells if next_mark == "X" else o_cells
            if not cells:
                # Fallback for externally-mutated test boards that are not strictly reachable.
                cells = x_cells or o_cells
            idx = cells[0]
            self._mcts.apply_cell(idx)
            board[idx] = _EMPTY

    def current_observation_info(self) -> Tuple[str, dict]:
        obs = self._render()
        terminal_success = (self._game_result == "win") if self._done else None
        terminal_reason = self._game_result if self._done else None
        info = self._build_info(
            raw_action="", parsed_action=None, parse_ok=True,
            illegal_action=False, vpr_reward=0.0,
            terminal_success=terminal_success, terminal_reason=terminal_reason,
        )
        return obs, info

    def snapshot_state(self) -> dict:
        return {
            "board": list(self._board),
            "step_count": int(self._step_count),
            "done": bool(self._done),
            "game_result": self._game_result,
            "last_opponent_action": self._last_opponent_action,
            "rng_state": self._rng.getstate(),
        }

    def restore_state(self, state: dict) -> Tuple[str, dict]:
        self._board = list(state["board"])
        self._step_count = int(state["step_count"])
        self._done = bool(state["done"])
        self._game_result = str(state["game_result"])
        self._last_opponent_action = state.get("last_opponent_action")
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        self._sync_mcts_from_board()
        return self.current_observation_info()

    def _sync_agent_move(self, idx: int) -> None:
        """Mirror the agent's move onto the synced pyspiel state (mcts opponent only)."""
        if self._mcts is not None:
            self._mcts.apply_cell(idx)

    def _opponent_move(self) -> Optional[int]:
        """Opponent plays one move; updates the board and the synced pyspiel state."""
        if self._opponent == "mcts":
            idx = self._mcts.choose()
            if idx is None:
                return None
            self._mcts.apply_cell(idx)
        else:
            idx = _random_opponent_move(self._board, self._rng)
            if idx is None:
                return None
        self._board[idx] = self._opponent_mark
        self._last_opponent_action = str(idx + 1)
        return idx

    def step(self, action_text: Optional[str], parse_ok: bool, raw_action: str) -> Tuple[str, float, bool, dict]:
        """Execute one agent step.

        Returns (obs, reward, done, info).
        action_text: the parsed action string (1-indexed cell number as string), or None on parse failure.
        """
        if self._done:
            obs = self._render()
            info = self._build_info(
                raw_action=raw_action, parsed_action=action_text, parse_ok=parse_ok,
                illegal_action=True, vpr_reward=0.0,
                terminal_success=(self._game_result == "win"),
                terminal_reason="already_done",
            )
            return obs, 0.0, True, info

        self._step_count += 1
        illegal = False
        vpr_reward = 0.0
        parsed_action = action_text

        # Parse failure
        if not parse_ok or action_text is None:
            illegal = True
            vpr_reward = self._invalid_penalty
            if self._invalid_terminates:
                self._done = True
                self._game_result = "ongoing"
                obs = self._render()
                info = self._build_info(
                    raw_action=raw_action, parsed_action=None, parse_ok=False,
                    illegal_action=True, vpr_reward=vpr_reward,
                    terminal_success=False, terminal_reason="invalid_action",
                )
                return obs, vpr_reward, True, info
            obs = self._render()
            info = self._build_info(
                raw_action=raw_action, parsed_action=None, parse_ok=False,
                illegal_action=True, vpr_reward=vpr_reward,
                terminal_success=None, terminal_reason=None,
            )
            return obs, vpr_reward, self._done, info

        # Parse cell number
        try:
            cell = int(action_text.strip())
        except (ValueError, AttributeError):
            cell = -1

        if cell < 1 or cell > 9:
            illegal = True
            vpr_reward = self._invalid_penalty
            if self._invalid_terminates:
                self._done = True
                obs = self._render()
                info = self._build_info(
                    raw_action=raw_action, parsed_action=action_text, parse_ok=True,
                    illegal_action=True, vpr_reward=vpr_reward,
                    terminal_success=False, terminal_reason="invalid_action",
                )
                return obs, vpr_reward, True, info
            obs = self._render()
            info = self._build_info(
                raw_action=raw_action, parsed_action=action_text, parse_ok=True,
                illegal_action=True, vpr_reward=vpr_reward,
                terminal_success=None, terminal_reason=None,
            )
            return obs, vpr_reward, self._done, info

        idx = cell - 1  # 0-indexed
        if self._board[idx] != _EMPTY:
            illegal = True
            vpr_reward = self._invalid_penalty
            if self._invalid_terminates:
                self._done = True
                obs = self._render()
                info = self._build_info(
                    raw_action=raw_action, parsed_action=action_text, parse_ok=True,
                    illegal_action=True, vpr_reward=vpr_reward,
                    terminal_success=False, terminal_reason="invalid_action",
                )
                return obs, vpr_reward, True, info
            obs = self._render()
            info = self._build_info(
                raw_action=raw_action, parsed_action=action_text, parse_ok=True,
                illegal_action=True, vpr_reward=vpr_reward,
                terminal_success=None, terminal_reason=None,
            )
            return obs, vpr_reward, self._done, info

        # Legal move — evaluate minimax optimality on the pre-move board (used both for
        # the oracle reward and for the move_optimal metric flag, regardless of mode).
        # In oracle mode the reward is the optimality bit; in outcome mode the move
        # itself earns 0 (the terminal step below carries the win/lose reward).
        oracle = oracle_valid_actions(self._board, self._agent_mark, self._opponent_mark)
        move_optimal = action_text.strip() in oracle
        if self._reward_mode == "outcome":
            vpr_reward = 0.0
        else:
            vpr_reward = 1.0 if move_optimal else 0.0

        # Place agent's move (and mirror onto the synced pyspiel state for mcts)
        self._board[idx] = self._agent_mark
        self._sync_agent_move(idx)
        winner = _check_winner(self._board)
        self._last_opponent_action = None

        if winner == self._agent_mark:
            self._done = True
            self._game_result = "win"
        elif _is_full(self._board):
            self._done = True
            self._game_result = "draw"
        elif self._step_count >= self._max_steps:
            self._done = True
            self._game_result = "ongoing"
        else:
            # Opponent's move (random or OpenSpiel MCTS)
            opp_idx = self._opponent_move()
            if opp_idx is not None:
                opp_winner = _check_winner(self._board)
                if opp_winner == self._opponent_mark:
                    self._done = True
                    self._game_result = "loss"
                elif _is_full(self._board):
                    self._done = True
                    self._game_result = "draw"

        terminal_success = None
        terminal_reason = None
        if self._done:
            terminal_success = (self._game_result == "win")
            terminal_reason = self._game_result

        # Outcome-reward baseline: the episode's only nonzero signal is the terminal
        # game result (+1 win / -1 loss / 0 draw or step-limit).
        if self._done and self._reward_mode == "outcome":
            vpr_reward = {"win": 1.0, "loss": -1.0}.get(self._game_result, 0.0)

        obs = self._render()
        info = self._build_info(
            raw_action=raw_action, parsed_action=action_text, parse_ok=True,
            illegal_action=illegal, vpr_reward=vpr_reward,
            terminal_success=terminal_success, terminal_reason=terminal_reason,
            move_optimal=move_optimal,
        )
        return obs, vpr_reward, self._done, info

    def _render(self) -> str:
        def cell(i: int) -> str:
            return self._board[i] if self._board[i] else str(i + 1)

        legal = [str(i + 1) for i in range(9) if self._board[i] == _EMPTY]
        lines = [
            f" {cell(0)} | {cell(1)} | {cell(2)} ",
            "---+---+---",
            f" {cell(3)} | {cell(4)} | {cell(5)} ",
            "---+---+---",
            f" {cell(6)} | {cell(7)} | {cell(8)} ",
        ]
        board_str = "\n".join(lines)
        return (f"TicTacToe board (you={self._agent_mark}, opponent={self._opponent_mark}):\n"
                f"{board_str}\nLegal cells: {', '.join(legal) if legal else 'none'}")

    def _build_info(self, *, raw_action: str, parsed_action: Optional[str],
                    parse_ok: bool, illegal_action: bool, vpr_reward: float,
                    terminal_success: Optional[bool], terminal_reason: Optional[str],
                    move_optimal: Optional[bool] = None) -> dict:
        oracle = (oracle_valid_actions(self._board, self._agent_mark, self._opponent_mark)
                  if not self._done else [])
        legal = [str(i + 1) for i in range(9) if self._board[i] == _EMPTY]
        return {
            "env_name": "vpr_tictactoe",
            "step": self._step_count,
            "max_steps": self._max_steps,
            "raw_action": raw_action,
            "parsed_action": parsed_action,
            "parse_ok": parse_ok,
            "illegal_action": illegal_action,
            "available_actions": legal,
            "vpr_reward": vpr_reward,
            "terminal_success": terminal_success,
            "terminal_reason": terminal_reason,
            "game_result": self._game_result,
            "oracle_valid_actions": oracle,
            "opponent_action": self._last_opponent_action,
            "agent_player": self._agent_mark,
            # Whether the agent's move was minimax-optimal (only set on legal moves;
            # None on illegal / parse-failure / already-done steps).
            "move_optimal": move_optimal,
        }
