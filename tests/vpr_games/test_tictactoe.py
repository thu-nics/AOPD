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
        "game_result", "oracle_valid_actions", "opponent_action",
        "move_optimal",
    ]
    for field in required:
        assert field in step_info, f"Missing info field: {field}"
    import json
    # JSON-serializable check (with None handling)
    json.dumps({k: v for k, v in step_info.items() if v is not None})


# ── Outcome reward mode (win/lose scoring for the standard-GRPO baseline) ─────

def test_invalid_reward_mode_raises():
    with pytest.raises(ValueError):
        TicTacToeGame(opponent="random", reward_mode="bogus")


def test_outcome_mode_win_reward():
    """Completing three-in-a-row → terminal win → +1.0 in outcome mode."""
    g = TicTacToeGame(opponent="random", seed=0, reward_mode="outcome")
    g.reset(seed=0)
    g._board = ["X", "X", "", "O", "O", "", "", "", ""]
    g._done = False
    g._step_count = 4
    _, reward, done, info = g.step("3", True, "<action>3</action>")  # completes row 0
    assert done and info["game_result"] == "win"
    assert reward == 1.0


def test_outcome_mode_loss_reward():
    """Opponent's only remaining move completes its line → terminal loss → -1.0."""
    g = TicTacToeGame(opponent="random", seed=0, reward_mode="outcome")
    g.reset(seed=0)
    # X to move; after X plays cell 9 (idx 8, non-winning), the only empty cell is
    # idx 5, which completes O's middle row → opponent wins deterministically.
    g._board = ["X", "O", "X", "O", "O", "", "X", "O", ""]
    g._done = False
    g._step_count = 0
    _, reward, done, info = g.step("9", True, "<action>9</action>")
    assert done and info["game_result"] == "loss"
    assert reward == -1.0


def test_outcome_mode_draw_reward():
    """Filling the last cell with no winner → terminal draw → 0.0."""
    g = TicTacToeGame(opponent="random", seed=0, reward_mode="outcome")
    g.reset(seed=0)
    g._board = ["X", "O", "X", "X", "O", "O", "O", "X", ""]
    g._done = False
    g._step_count = 8
    _, reward, done, info = g.step("9", True, "<action>9</action>")
    assert done and info["game_result"] == "draw"
    assert reward == 0.0


def test_outcome_mode_nonterminal_is_zero_even_when_optimal():
    """A legal, minimax-optimal, but non-terminal move earns +1 in oracle mode but 0 in
    outcome mode (outcome only pays at the terminal step)."""
    board = ["X", "", "", "O", "O", "", "", "", ""]  # block the O threat at cell 6 (idx5)
    g_oracle = TicTacToeGame(opponent="random", seed=0, reward_mode="oracle")
    g_oracle.reset(seed=0)
    g_oracle._board = list(board); g_oracle._done = False; g_oracle._step_count = 2
    _, r_oracle, _, _ = g_oracle.step("6", True, "<action>6</action>")
    assert r_oracle == 1.0  # blocking the only threat is the unique optimal move

    g_out = TicTacToeGame(opponent="random", seed=0, reward_mode="outcome")
    g_out.reset(seed=0)
    g_out._board = list(board); g_out._done = False; g_out._step_count = 2
    _, r_out, done_out, _ = g_out.step("6", True, "<action>6</action>")
    assert r_out == 0.0 and not done_out


# ── Configurable agent side (X / O) ──────────────────────────────────────────

def test_invalid_agent_player_raises():
    with pytest.raises(ValueError):
        TicTacToeGame(opponent="random", agent_player="Z")


def test_agent_as_O_opponent_moves_first_on_reset():
    g = TicTacToeGame(opponent="random", seed=0, agent_player="O")
    obs, info = g.reset(seed=0)
    assert info["agent_player"] == "O"
    assert info["opponent_action"] is not None, "opponent (X) should have opened"
    assert sum(1 for c in g._board if c) == 1


def test_agent_as_O_oracle_winning_move():
    """Minimax oracle is computed for the agent's mark (O); an immediate O win is optimal."""
    g = TicTacToeGame(opponent="random", seed=0, agent_player="O",
                      invalid_action_terminates=False)
    g.reset(seed=0)
    g._board = ["O", "O", "", "X", "X", "", "", "", ""]  # O to move; cell 3 completes O's row
    g._done = False
    g._step_count = 4
    assert oracle_valid_actions(g._board, "O", "X") == ["3"]
    _, reward, done, info = g.step("3", True, "<action>3</action>")
    assert done and info["game_result"] == "win" and reward == 1.0


# ── OpenSpiel MCTS opponent (skipped if OpenSpiel not installed) ──────────────

def test_mcts_opponent_plays_legal_and_terminates():
    pytest.importorskip("pyspiel")
    g = TicTacToeGame(opponent="mcts", seed=0, agent_player="X", mcts_max_simulations=30)
    obs, info = g.reset(seed=0)
    done, steps = False, 0
    while not done and steps < 9:
        legal = info["available_actions"]
        if not legal:
            break
        cell = legal[0]
        obs, reward, done, info = g.step(cell, True, f"<action>{cell}</action>")
        steps += 1
    assert done, "game with mcts opponent must terminate"


def test_mcts_opponent_deterministic_first_move_for_agent_O():
    pytest.importorskip("pyspiel")
    g1 = TicTacToeGame(opponent="mcts", seed=7, agent_player="O", mcts_max_simulations=30)
    _, i1 = g1.reset(seed=7)
    g2 = TicTacToeGame(opponent="mcts", seed=7, agent_player="O", mcts_max_simulations=30)
    _, i2 = g2.reset(seed=7)
    assert i1["opponent_action"] is not None
    assert i1["opponent_action"] == i2["opponent_action"]


# ── State-group / Vine snapshot helpers ──────────────────────────────────────

def test_snapshot_restore_round_trip_preserves_board_and_rng():
    g = TicTacToeGame(opponent="random", seed=11, invalid_action_terminates=False)
    obs0, info0 = g.reset(seed=11)
    snap = g.snapshot_state()
    obs1, reward1, done1, info1 = g.step("5", True, "<action>5</action>")
    assert g._board != snap["board"]
    restored_obs, restored_info = g.restore_state(snap)
    assert g._board == snap["board"]
    assert restored_obs == obs0

    # Restoring the same snapshot should replay the same random opponent move.
    _, _, _, first = g.step("5", True, "<action>5</action>")
    g.restore_state(snap)
    _, _, _, second = g.step("5", True, "<action>5</action>")
    assert first["opponent_action"] == second["opponent_action"]


def test_snapshot_restore_after_candidate_like_eval_keeps_original_state():
    g = TicTacToeGame(opponent="random", seed=13, invalid_action_terminates=False)
    obs0, _ = g.reset(seed=13)
    snap = g.snapshot_state()
    for action in ["1", "2", "3"]:
        g.restore_state(snap)
        g.step(action, True, f"<action>{action}</action>")
    restored_obs, _ = g.restore_state(snap)
    assert restored_obs == obs0
    assert g._board == snap["board"]
