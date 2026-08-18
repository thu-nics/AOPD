"""Shared teacher-match reward modes for semantic agent training."""

from __future__ import annotations

import math

TEACHER_REWARD_MODES = {"appearance", "frequency_weighted"}
DEFAULT_TEACHER_REWARD_MODE = "frequency_weighted"
DEFAULT_FREQUENCY_BONUS_SCALE = 0.5


def validate_teacher_reward_config(
    mode: str,
    frequency_bonus_scale: float,
) -> tuple[str, float]:
    mode = str(mode)
    scale = float(frequency_bonus_scale)
    if mode not in TEACHER_REWARD_MODES:
        raise ValueError(f"unsupported teacher reward mode: {mode!r}")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("frequency_bonus_scale must be finite and non-negative")
    return mode, scale


def teacher_match_reward(
    match_count: int,
    *,
    teacher_sample_count: int,
    mode: str = DEFAULT_TEACHER_REWARD_MODE,
    frequency_bonus_scale: float = DEFAULT_FREQUENCY_BONUS_SCALE,
) -> float:
    """Convert an ordered teacher-multiset match count into semantic reward."""
    mode, scale = validate_teacher_reward_config(mode, frequency_bonus_scale)
    if isinstance(match_count, bool) or int(match_count) != match_count or match_count < 0:
        raise ValueError("teacher match count must be a non-negative integer")
    if isinstance(teacher_sample_count, bool) or int(teacher_sample_count) != teacher_sample_count or teacher_sample_count < 0:
        raise ValueError("teacher sample count must be a non-negative integer")
    count = int(match_count)
    sample_count = int(teacher_sample_count)
    if count > sample_count:
        raise ValueError("teacher match count cannot exceed teacher sample count")
    if count == 0:
        return 0.0
    if mode == "appearance" or sample_count <= 1:
        return 1.0
    return 1.0 + scale * (count - 1) / (sample_count - 1)
