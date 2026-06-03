"""Shared reward helpers for VPR game environments."""

from __future__ import annotations

from typing import Optional

# Terminal reasons that are neither a win nor a loss (no reward sign): the episode
# ended without a decisive outcome (ran out of steps, already finished, drew, etc.).
NEUTRAL_TERMINAL_REASONS = frozenset({
    "timeout", "already_done", "draw", "ongoing", "step_limit", None,
})


def outcome_reward(done: bool, terminal_success: Optional[bool],
                   terminal_reason: Optional[str]) -> float:
    """Win/lose outcome reward, shared by the env `reward_mode="outcome"` baselines.

    +1.0 on a successful terminal step, -1.0 on a losing terminal step (any non-neutral
    terminal reason: mine hit, wrong digit, invalid/illegal action, ...), and 0.0 for
    non-terminal steps or neutral terminal endings (timeout / step-limit / draw).
    """
    if not done:
        return 0.0
    if terminal_success:
        return 1.0
    if terminal_reason in NEUTRAL_TERMINAL_REASONS:
        return 0.0
    return -1.0
