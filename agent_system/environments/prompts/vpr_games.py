"""Prompt templates for VPR game environments."""

from agent_system.environments.env_package.vpr_games.common.parser import (
    normalize_action_format,
)

TICTACTOE_TEMPLATE = """\
You are playing Tic-Tac-Toe. You play as {mark}, your opponent plays as {opp}.

{board}

Choose one of the legal cells listed above.
Respond with your chosen cell number inside <action> tags.
You may reason briefly in <think>...</think> before your answer.
Your final answer must be: <action>CELL_NUMBER</action>
"""

SUDOKU_TEMPLATE = """\
You are solving a Sudoku puzzle. Fill in the blank cells (shown as .) using digits 1-9.
Each row, column, and 3×3 box must contain digits 1-9 exactly once.

Current grid:
{grid}

Blank cells: {blank_cells}

Choose one blank cell and one digit to fill it.
Format: <action>ROW COL DIGIT</action>  (rows and columns are 1-indexed, e.g. <action>3 5 7</action>)
You may reason briefly in <think>...</think> before your answer.
"""

MINESWEEPER_TEMPLATE = """\
You are playing Minesweeper on a {rows}×{cols} board with {mines} mines.
Legend: . = hidden, F = flagged, numbers = revealed (count of adjacent mines)

Current board:
{board}

Unrevealed cells: {unrevealed_cells}
Flagged cells: {flagged_cells}

Choose one action:
  reveal ROW COL  — reveal a hidden cell
  flag ROW COL    — toggle flag on a hidden cell
Rows and columns are 1-indexed.

Format: <action>ACTION ROW COL</action>  (e.g. <action>reveal 2 3</action> or <action>flag 1 4</action>)
Aliases: open/click → reveal, mark → flag
You may reason briefly in <think>...</think> before your answer.
"""

SOKOBAN_TEMPLATE = """\
You are solving a Sokoban puzzle with {num_boxes} box(es).
Legend: # = wall, _ = floor, O = target, X = box, P = player, √ = box on target, S = player on target

Current board:
{board}

Choose one move:
  up
  down
  left
  right

Format: <action>DIRECTION</action>  (e.g. <action>up</action>)
You may reason briefly in <think>...</think> before your answer.
"""

SUDOKU_BOXED_TEMPLATE = """\
You are solving a Sudoku puzzle. Fill in the blank cells (shown as .) using digits 1-9.
Each row, column, and 3×3 box must contain digits 1-9 exactly once.

Current grid:
{grid}

Blank cells: {blank_cells}

Choose one blank cell and one digit to fill it.
You may reason briefly in <think>...</think> before your answer.
Your final answer must be: \\boxed{{ROW COL DIGIT}}
Rows and columns are 1-indexed, e.g. \\boxed{{3 5 7}}.
"""

MINESWEEPER_BOXED_TEMPLATE = """\
You are playing Minesweeper on a {rows}×{cols} board with {mines} mines.
Legend: . = hidden, F = flagged, numbers = revealed (count of adjacent mines)

Current board:
{board}

Unrevealed cells: {unrevealed_cells}
Flagged cells: {flagged_cells}

Choose one action:
  reveal ROW COL  — reveal a hidden cell
  flag ROW COL    — toggle flag on a hidden cell
Rows and columns are 1-indexed.

Aliases: open/click → reveal, mark → flag
You may reason briefly in <think>...</think> before your answer.
Your final answer must be one action, e.g. \\boxed{{reveal 2 3}} or \\boxed{{flag 1 4}}.
"""

SOKOBAN_BOXED_TEMPLATE = """\
You are solving a Sokoban puzzle with {num_boxes} box(es).
Legend: # = wall, _ = floor, O = target, X = box, P = player, √ = box on target, S = player on target

Current board:
{board}

Choose one move:
  up
  down
  left
  right

You may reason briefly in <think>...</think> before your answer.
Your final answer must be one move, e.g. \\boxed{{up}}.
"""

_GAME_TEMPLATES = {
    "sokoban": {
        "action_tag": SOKOBAN_TEMPLATE,
        "boxed": SOKOBAN_BOXED_TEMPLATE,
    },
    "sudoku": {
        "action_tag": SUDOKU_TEMPLATE,
        "boxed": SUDOKU_BOXED_TEMPLATE,
    },
    "minesweeper": {
        "action_tag": MINESWEEPER_TEMPLATE,
        "boxed": MINESWEEPER_BOXED_TEMPLATE,
    },
}


def get_vpr_game_template(game: str, action_format: str = "action_tag") -> str:
    try:
        templates = _GAME_TEMPLATES[str(game).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported VPR game template: {game!r}") from exc
    return templates[normalize_action_format(action_format)]
