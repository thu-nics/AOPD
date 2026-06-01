"""Unit tests for TicTacToe game logic and oracle."""

import sys
import importlib.util
import pytest


def load_game():
    spec = importlib.util.spec_from_file_location(
        "tictactoe_game",
        "agent_system/environments/env_package/vpr_games/tictactoe/game.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tictactoe_game"] = mod
    spec.loader.exec_module(mod)
    return mod


ttt = load_game()
TicTacToeGame = ttt.TicTacToeGame
oracle_valid_actions = ttt.oracle_valid_actions


def test_oracle_empty_board_all_valid():
    board = [""] * 9
    orac = oracle_valid_actions(board)
    assert len(orac) > 0
    assert all(str(i) in orac for i in range(1, 10))


def test_oracle_single_optimal():
    # X has cells 0,1 (indices), needs 2 to win; O has 3,4
    board = ["X", "X", "", "O", "O", "", "", "", ""]
    orac = oracle_valid_actions(board)
    assert orac == ["3"], f"Expected ['3'], got {orac}"


def test_optimal_move_reward():
    game = TicTacToeGame(opponent="random", seed=0, invalid_action_terminates=False)
    game.reset(seed=0)
    # Place X in middle of first row (winning next if we set up correctly)
    game._board = ["X", "X", "", "O", "O", "", "", "", ""]
    game._done = False
    game._step_count = 4
    obs, reward, done, info = game.step("3", True, "<action>3</action>")
    assert reward == 1.0, f"Expected +1.0 for optimal move, got {reward}"


def test_legal_non_optimal_reward():
    game = TicTacToeGame(opponent="random", seed=0, invalid_action_terminates=False)
    game.reset(seed=0)
    game._board = ["X", "X", "", "O", "O", "", "", "", ""]
    game._done = False
    game._step_count = 4
    obs, reward, done, info = game.step("7", True, "<action>7</action>")
    assert reward == 0.0, f"Expected 0.0 for legal non-optimal, got {reward}"


def test_invalid_cell_penalty():
    game = TicTacToeGame(opponent="random", seed=1, invalid_action_terminates=False)
    game.reset(seed=1)
    obs, reward, done, info = game.step("10", True, "<action>10</action>")
    assert reward == -1.0, f"Expected penalty, got {reward}"
    assert info["illegal_action"]


def test_occupied_cell_penalty():
    game = TicTacToeGame(opponent="random", seed=2, invalid_action_terminates=False)
    game.reset(seed=2)
    game._board[4] = "O"
    obs, reward, done, info = game.step("5", True, "<action>5</action>")
    assert reward == -1.0
    assert info["illegal_action"]


def test_parse_failure_penalty():
    game = TicTacToeGame(opponent="random", seed=3, invalid_action_terminates=False)
    game.reset(seed=3)
    obs, reward, done, info = game.step(None, False, "garbage")
    assert reward == -1.0
    assert not info["parse_ok"]


def test_seeded_reset_determinism():
    game = TicTacToeGame(opponent="random", seed=42)
    obs1, info1 = game.reset(seed=42)
    obs2, info2 = game.reset(seed=42)
    assert obs1 == obs2


def test_info_schema():
    game = TicTacToeGame(opponent="random", seed=0)
    obs, info = game.reset(seed=0)
    _, _, _, step_info = game.step("5", True, "<action>5</action>")
    required = [
        "env_name", "step", "max_steps", "raw_action", "parsed_action",
        "parse_ok", "illegal_action", "available_actions",
        "vpr_reward", "terminal_success", "terminal_reason",
        "game_result", "oracle_valid_actions", "opponent_action"
    ]
    for field in required:
        assert field in step_info, f"Missing info field: {field}"
    import json
    # JSON-serializable check (with None handling)
    json.dumps({k: v for k, v in step_info.items() if v is not None})
