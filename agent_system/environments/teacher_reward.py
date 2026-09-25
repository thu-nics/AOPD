"""Shared teacher-match reward modes for semantic agent training."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
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

    metrics = {
        "tool_argument_normalized_match_count": float(sum(int(row.get("tool_argument_normalized_match_count", 0)) for row in candidate_rows if not row.get("is_padding", False))),
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
    metrics.update(_stopping_selection_diagnostics(candidate_episodes))
    metrics.update(_repeated_tool_diagnostics(selected_episodes))
    metrics.update(_rollout_progress_diagnostics(candidate_episodes, selected_episodes))
    return metrics


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _teacher_multiset(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("teacher_multiset") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _candidate_groups(
    candidate_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    for episode_index, episode in enumerate(candidate_episodes):
        keyed: dict[str, list[Mapping[str, Any]]] = {}
        for row in episode:
            group_id = row.get("state_group_uid")
            if group_id is None:
                # Focused unit tests and old callers may provide one implicit
                # candidate group without rollout metadata.
                group_id = f"implicit-{episode_index}"
            keyed.setdefault(str(group_id), []).append(row)
        groups.extend(keyed.values())
    return groups


def _rollout_progress_diagnostics(
    candidate_episodes: Sequence[Sequence[Mapping[str, Any]]],
    selected_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    selected_rows = [row for episode in selected_episodes for row in episode if row.get("action_kind") in {"tool", "message", "invalid"}]
    selected_count = len(selected_rows)
    candidate_groups = _candidate_groups(candidate_episodes)
    candidate_rows = [row for group in candidate_groups for row in group if row.get("action_kind") in {"tool", "message", "invalid"}]
    repeat_candidates = [row for row in candidate_rows if bool(row.get("prospective_no_progress_repeat", False))]
    capped_candidates = [row for row in candidate_rows if bool(row.get("repeat_reward_capped", False))]
    capped_groups = [group for group in candidate_groups if any(bool(row.get("repeat_reward_capped", False)) for row in group)]

    top_repeat_groups = []
    top_repeat_teacher_groups = []
    for group in candidate_groups:
        if not group:
            continue
        maximum = max(_selection_score(row) for row in group)
        top = [row for row in group if math.isclose(_selection_score(row), maximum, abs_tol=1e-8)]
        repeated_top = [row for row in top if bool(row.get("prospective_no_progress_repeat", False))]
        if repeated_top:
            top_repeat_groups.append(group)
            if any(int(row.get("teacher_frequency", 0) or 0) > 0 for row in repeated_top):
                top_repeat_teacher_groups.append(group)

    repeat_terminations = []
    for episode in selected_episodes:
        terminal = next(
            (row for row in reversed(episode) if row.get("terminal_reason") == "no_progress_repeat_limit"),
            None,
        )
        if terminal is not None:
            repeat_terminations.append(terminal)
    termination_turns = [float(row.get("turn_index", row.get("step", 0)) or 0) + (1.0 if row.get("turn_index") is not None else 0.0) for row in repeat_terminations]
    valid_termination_outcomes = [bool(row["terminal_success"]) for row in repeat_terminations if row.get("terminal_success") is not None]

    return {
        "nonrepeat_argmax_available_rate": _safe_rate(
            sum(bool(row.get("nonrepeat_alternative_available", False)) for row in selected_rows),
            selected_count,
        ),
        "nonrepeat_commit_rate": _safe_rate(
            sum(bool(row.get("nonrepeat_preference_applied", False)) for row in selected_rows),
            selected_count,
        ),
        "repeat_reward_capped_candidate_rate": _safe_rate(len(capped_candidates), len(candidate_rows)),
        "repeat_reward_capped_group_rate": _safe_rate(len(capped_groups), len(candidate_groups)),
        "no_progress_repeat_candidate_teacher_match_rate": _safe_rate(
            sum(int(row.get("teacher_frequency", 0) or 0) > 0 for row in repeat_candidates),
            len(repeat_candidates),
        ),
        "top_reward_no_progress_repeat_group_rate": _safe_rate(len(top_repeat_groups), len(candidate_groups)),
        "top_reward_no_progress_repeat_teacher_match_rate": _safe_rate(len(top_repeat_teacher_groups), len(top_repeat_groups)),
        "repeat_limit_termination_count": float(len(repeat_terminations)),
        "repeat_limit_termination_rate": _safe_rate(len(repeat_terminations), len(selected_episodes)),
        "repeat_limit_termination_turn_mean": (sum(termination_turns) / len(termination_turns) if termination_turns else 0.0),
        "repeat_limit_terminal_success_rate": _safe_rate(sum(valid_termination_outcomes), len(valid_termination_outcomes)),
    }


def _selection_score(row: Mapping[str, Any]) -> float:
    value = row.get("selection_score")
    if value is None:
        value = row.get("rewards", row.get("reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _turn_bucket(row: Mapping[str, Any]) -> str:
    try:
        turn = int(row.get("turn_index", 0)) + 1
    except (TypeError, ValueError):
        turn = 1
    if turn <= 5:
        return "turn_1_5"
    if turn <= 10:
        return "turn_6_10"
    if turn <= 20:
        return "turn_11_20"
    return "turn_21_plus"


def _stopping_selection_diagnostics(
    candidate_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    groups = []
    for rows in _candidate_groups(candidate_episodes):
        action_rows = [row for row in rows if str(row.get("action_kind") or "") in {"tool", "message", "invalid"}]
        teacher = _teacher_multiset(action_rows[0]) if action_rows else []
        if action_rows and teacher:
            groups.append((action_rows, teacher))

    candidate_count = sum(len(rows) for rows, _ in groups)
    candidate_messages = sum(str(row.get("action_kind") or "") == "message" for rows, _ in groups for row in rows)
    student_message_groups = 0
    teacher_message_groups = 0
    teacher_message_student_missing = 0
    teacher_message_unmatched_candidate = 0
    teacher_message_matched_selected = 0
    teacher_message_matched_not_selected = 0
    teacher_all_tool_groups = 0
    teacher_all_tool_student_message = 0
    comparable_message_candidates = 0
    matched_message_candidates = 0
    comparable_message_groups = 0
    all_unmatched_message_groups = 0
    matched_message_groups = 0
    matched_message_not_selected = 0
    matched_message_lower_score = 0
    matched_message_tied_tool_selected = 0
    matched_message_tied_duplicate_tool_bias = 0
    unmatched_message_groups = 0
    unmatched_message_tool_selected = 0
    turn_counts: dict[str, Counter[str]] = {}

    for rows, teacher in groups:
        messages = [row for row in rows if row.get("action_kind") == "message"]
        tools = [row for row in rows if row.get("action_kind") == "tool"]
        selected = next(
            (row for row in rows if bool(row.get("state_group_selected", False))),
            None,
        )
        teacher_kinds = [str(item.get("kind") or "") for item in teacher]
        teacher_has_message = "message" in teacher_kinds
        teacher_all_tool = bool(teacher_kinds) and all(kind == "tool" for kind in teacher_kinds)
        student_has_message = bool(messages)
        matched_messages = [row for row in messages if int(row.get("teacher_frequency", 0) or 0) > 0]
        selected_is_matched_message = bool(selected is not None and selected.get("action_kind") == "message" and int(selected.get("teacher_frequency", 0) or 0) > 0)

        student_message_groups += int(student_has_message)
        teacher_message_groups += int(teacher_has_message)
        teacher_message_student_missing += int(teacher_has_message and not student_has_message)
        teacher_all_tool_groups += int(teacher_all_tool)
        teacher_all_tool_student_message += int(teacher_all_tool and student_has_message)

        if teacher_has_message:
            teacher_message_unmatched_candidate += int(student_has_message and not matched_messages)
            teacher_message_matched_selected += int(selected_is_matched_message)
            teacher_message_matched_not_selected += int(bool(matched_messages) and not selected_is_matched_message)
            comparable_message_candidates += len(messages)
            matched_message_candidates += len(matched_messages)
            if student_has_message:
                comparable_message_groups += 1
                all_unmatched_message_groups += int(not matched_messages)

        if student_has_message and not matched_messages:
            unmatched_message_groups += 1
            unmatched_message_tool_selected += int(selected is not None and selected.get("action_kind") == "tool")

        if matched_messages:
            matched_message_groups += 1
            if not selected_is_matched_message:
                matched_message_not_selected += 1
                selected_score = _selection_score(selected or {})
                best_message_score = max(_selection_score(row) for row in matched_messages)
                if best_message_score < selected_score and not math.isclose(best_message_score, selected_score):
                    matched_message_lower_score += 1
                elif selected is not None and selected.get("action_kind") == "tool" and math.isclose(best_message_score, selected_score):
                    matched_message_tied_tool_selected += 1
                    top_tools = [row for row in tools if math.isclose(_selection_score(row), selected_score)]
                    top_messages = [row for row in messages if math.isclose(_selection_score(row), selected_score)]
                    canonical_counts = Counter(str(row.get("parsed_action") or "") for row in top_tools)
                    duplicate_tool = any(count > 1 for count in canonical_counts.values())
                    matched_message_tied_duplicate_tool_bias += int(duplicate_tool and len(top_tools) > len(top_messages))

        bucket = _turn_bucket(rows[0])
        counts = turn_counts.setdefault(bucket, Counter())
        counts["groups"] += 1
        counts["teacher_message"] += int(teacher_has_message)
        counts["student_message"] += int(student_has_message)
        counts["selected_message"] += int(selected is not None and selected.get("action_kind") == "message")

    group_count = len(groups)
    output = {
        "diagnostic_state_group_count": float(group_count),
        "candidate_action_count": float(candidate_count),
        "candidate_message_count": float(candidate_messages),
        "candidate_message_rate": _safe_rate(candidate_messages, candidate_count),
        "student_message_group_count": float(student_message_groups),
        "group_has_message_candidate_rate": _safe_rate(student_message_groups, group_count),
        "teacher_has_message_group_count": float(teacher_message_groups),
        "teacher_has_message_group_rate": _safe_rate(teacher_message_groups, group_count),
        "teacher_has_message_student_missing_rate": _safe_rate(teacher_message_student_missing, teacher_message_groups),
        "teacher_message_unmatched_candidate_rate": _safe_rate(teacher_message_unmatched_candidate, teacher_message_groups),
        "teacher_message_matched_selected_rate": _safe_rate(teacher_message_matched_selected, teacher_message_groups),
        "teacher_message_matched_not_selected_rate": _safe_rate(teacher_message_matched_not_selected, teacher_message_groups),
        "teacher_all_tool_student_message_rate": _safe_rate(teacher_all_tool_student_message, teacher_all_tool_groups),
        "message_candidate_match_rate_when_teacher_message": _safe_rate(matched_message_candidates, comparable_message_candidates),
        "message_candidate_all_unmatched_rate_when_teacher_message": _safe_rate(all_unmatched_message_groups, comparable_message_groups),
        "matched_message_group_count": float(matched_message_groups),
        "matched_message_not_selected_count": float(matched_message_not_selected),
        "matched_message_not_selected_rate": _safe_rate(matched_message_not_selected, matched_message_groups),
        "matched_message_lower_score_rate": _safe_rate(matched_message_lower_score, matched_message_not_selected),
        "matched_message_tied_tool_selected_rate": _safe_rate(matched_message_tied_tool_selected, matched_message_not_selected),
        "matched_message_tied_duplicate_tool_bias_rate": _safe_rate(
            matched_message_tied_duplicate_tool_bias,
            matched_message_tied_tool_selected,
        ),
        "unmatched_message_tool_selected_rate": _safe_rate(unmatched_message_tool_selected, unmatched_message_groups),
    }
    for bucket in (
        "turn_1_5",
        "turn_6_10",
        "turn_11_20",
        "turn_21_plus",
    ):
        counts = turn_counts.get(bucket, Counter())
        total = counts["groups"]
        output[f"{bucket}_teacher_message_rate"] = _safe_rate(counts["teacher_message"], total)
        output[f"{bucket}_state_group_count"] = float(total)
        output[f"{bucket}_student_message_candidate_rate"] = _safe_rate(counts["student_message"], total)
        output[f"{bucket}_selected_message_rate"] = _safe_rate(counts["selected_message"], total)
    return output


def _repeated_tool_diagnostics(
    selected_episodes: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    tool_to_tool_transitions = 0
    repeated_tool_transitions = 0
    repeated_tool_same_observation = 0
    repeated_tool_teacher_matches = 0
    repeated_tool_trajectories = 0
    max_streaks: list[int] = []

    for episode in selected_episodes:
        actions = [row for row in episode if str(row.get("action_kind") or "") in {"tool", "message", "invalid"}]
        first_is_tool = bool(actions and actions[0].get("action_kind") == "tool")
        max_streak = int(first_is_tool)
        current_streak = int(first_is_tool)
        for previous, current in zip(actions, actions[1:], strict=False):
            both_tools = previous.get("action_kind") == "tool" and current.get("action_kind") == "tool"
            if not both_tools:
                current_streak = int(current.get("action_kind") == "tool")
                continue
            tool_to_tool_transitions += 1
            same_action = str(previous.get("parsed_action") or "") == str(current.get("parsed_action") or "")
            if not same_action:
                current_streak = 1
                continue
            repeated_tool_transitions += 1
            same_observation = str(previous.get("observation") or "") == str(current.get("observation") or "")
            if same_observation:
                repeated_tool_same_observation += 1
                repeated_tool_teacher_matches += int(int(current.get("teacher_frequency", 0) or 0) > 0)
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        if actions:
            max_streaks.append(max_streak)
            repeated_tool_trajectories += int(max_streak >= 3)

    return {
        "tool_to_tool_transition_count": float(tool_to_tool_transitions),
        "repeated_tool_transition_count": float(repeated_tool_transitions),
        "repeated_tool_same_observation_count": float(repeated_tool_same_observation),
        "action_trajectory_count": float(len(max_streaks)),
        "consecutive_same_tool_call_rate": _safe_rate(repeated_tool_transitions, tool_to_tool_transitions),
        "consecutive_same_tool_same_observation_rate": _safe_rate(repeated_tool_same_observation, tool_to_tool_transitions),
        "repeat_streak_ge3_trajectory_rate": _safe_rate(repeated_tool_trajectories, len(max_streaks)),
        "repeat_streak_max_mean": (float(sum(max_streaks)) / len(max_streaks) if max_streaks else 0.0),
        "repeated_tool_teacher_match_rate": _safe_rate(repeated_tool_teacher_matches, repeated_tool_same_observation),
    }
