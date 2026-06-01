"""Prompt templates for VPR game environments."""

TICTACTOE_TEMPLATE = """\
You are playing Tic-Tac-Toe. You play as X, your opponent plays as O.

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
