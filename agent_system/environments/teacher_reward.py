"""Shared teacher-match reward modes for semantic agent training."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

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


def select_with_appearance_counterfactual(
    selection_scores: Sequence[float],
    appearance_scores: Sequence[float],
    rng: random.Random,
) -> tuple[int, int]:
    """Select the real action and a no-side-effect appearance counterfactual.

    Both selections start from the same RNG state. The real selection consumes
    exactly the same RNG draw as the previous uniform-argmax implementation;
    the counterfactual uses a cloned generator and therefore cannot perturb the
    rollout trajectory or later task scheduling.
    """
    if len(selection_scores) == 0:
        raise ValueError("cannot select from an empty candidate group")
    if len(selection_scores) != len(appearance_scores):
        raise ValueError("real and appearance selection scores must align")

    def argmax_indices(scores: Sequence[float]) -> list[int]:
        maximum = max(scores)
        return [index for index, score in enumerate(scores) if score == maximum]

    counterfactual_rng = random.Random(0)
    counterfactual_rng.setstate(rng.getstate())
    selected_index = rng.choice(argmax_indices(selection_scores))
    appearance_index = counterfactual_rng.choice(argmax_indices(appearance_scores))
    return selected_index, appearance_index


def teacher_selection_diagnostics(
    candidate_episodes: Sequence[Sequence[Mapping[str, Any]]],
    selected_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    """Summarize teacher matching and frequency-sensitive advancement."""
    candidate_rows = [row for episode in candidate_episodes for row in episode]
    selected_rows = [row for episode in selected_episodes for row in episode]

    def candidate_mean(kind: str) -> float:
        values = [float(row.get("teacher_frequency", 0) or 0) for row in candidate_rows if row.get("action_kind") == kind]
        return sum(values) / len(values) if values else 0.0

    valid_selected = [row for row in selected_rows if row.get("action_kind") in {"tool", "message", "invalid"}]
    denominator = len(valid_selected)

    def selected_rate(predicate) -> float:
        if not denominator:
            return 0.0
        return sum(bool(predicate(row)) for row in valid_selected) / denominator

    return {
        "tool_candidate_teacher_match_count_mean": candidate_mean("tool"),
        "message_candidate_teacher_match_count_mean": candidate_mean("message"),
        "selected_tool_action_rate": selected_rate(lambda row: row.get("action_kind") == "tool"),
        "selected_message_action_rate": selected_rate(lambda row: row.get("action_kind") == "message"),
        "appearance_counterfactual_selected_tool_rate": selected_rate(lambda row: row.get("appearance_counterfactual_action_kind") == "tool"),
        "appearance_counterfactual_selected_message_rate": selected_rate(lambda row: row.get("appearance_counterfactual_action_kind") == "message"),
        "frequency_changed_selection_rate": selected_rate(lambda row: row.get("frequency_changed_selection", False)),
        "frequency_changed_selection_to_tool_rate": selected_rate(lambda row: row.get("frequency_changed_selection_to_tool", False)),
        "frequency_changed_selection_to_message_rate": selected_rate(lambda row: row.get("frequency_changed_selection_to_message", False)),
    }
