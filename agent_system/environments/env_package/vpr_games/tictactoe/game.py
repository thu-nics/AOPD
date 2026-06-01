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


def _minimax(board: List[str], is_agent_turn: bool, alpha: int, beta: int) -> int:
    winner = _check_winner(board)
    if winner == _AGENT:
        return 1
    if winner == _OPPONENT:
        return -1
    if _is_full(board):
        return 0

    if is_agent_turn:
        best = -2
        for i in range(9):
            if board[i] == _EMPTY:
                board[i] = _AGENT
                val = _minimax(board, False, alpha, beta)
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
                board[i] = _OPPONENT
                val = _minimax(board, True, alpha, beta)
                board[i] = _EMPTY
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best


def oracle_valid_actions(board: List[str]) -> List[str]:
    """Return 1-indexed cell indices of minimax-optimal moves for the agent."""
    empty_cells = [i for i in range(9) if board[i] == _EMPTY]
    if not empty_cells:
        return []

    scores = []
    for i in empty_cells:
        board[i] = _AGENT
        s = _minimax(board, False, -2, 2)
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
                 max_steps: int = 9, invalid_penalty: float = -1.0, seed: int = 0):
        self._opponent = opponent
        self._invalid_terminates = invalid_action_terminates
        self._max_steps = max_steps
        self._invalid_penalty = invalid_penalty
        self._rng = random.Random(seed)
        self._board: List[str] = [_EMPTY] * 9
        self._step_count: int = 0
        self._done: bool = False
        self._game_result: str = "ongoing"
        self._last_opponent_action: Optional[str] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        self._board = [_EMPTY] * 9
        self._step_count = 0
        self._done = False
        self._game_result = "ongoing"
        self._last_opponent_action = None
        obs = self._render()
        info = self._build_info(
            raw_action="", parsed_action=None, parse_ok=True,
            illegal_action=False, vpr_reward=0.0,
            terminal_success=None, terminal_reason=None,
        )
        return obs, info

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

        # Legal move — compute oracle reward before placing
        oracle = oracle_valid_actions(self._board)
        vpr_reward = 1.0 if action_text.strip() in oracle else 0.0

        # Place agent's move
        self._board[idx] = _AGENT
        winner = _check_winner(self._board)
        self._last_opponent_action = None

        if winner == _AGENT:
            self._done = True
            self._game_result = "win"
        elif _is_full(self._board):
            self._done = True
            self._game_result = "draw"
        elif self._step_count >= self._max_steps:
            self._done = True
            self._game_result = "ongoing"
        else:
            # Opponent's move
            opp_idx = _random_opponent_move(self._board, self._rng)
            if opp_idx is not None:
                self._board[opp_idx] = _OPPONENT
                self._last_opponent_action = str(opp_idx + 1)
                opp_winner = _check_winner(self._board)
                if opp_winner == _OPPONENT:
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

        obs = self._render()
        info = self._build_info(
            raw_action=raw_action, parsed_action=action_text, parse_ok=True,
            illegal_action=illegal, vpr_reward=vpr_reward,
            terminal_success=terminal_success, terminal_reason=terminal_reason,
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
        return f"TicTacToe board (X=you, O=opponent):\n{board_str}\nLegal cells: {', '.join(legal) if legal else 'none'}"

    def _build_info(self, *, raw_action: str, parsed_action: Optional[str],
                    parse_ok: bool, illegal_action: bool, vpr_reward: float,
                    terminal_success: Optional[bool], terminal_reason: Optional[str]) -> dict:
        oracle = oracle_valid_actions(self._board) if not self._done else []
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
        }
