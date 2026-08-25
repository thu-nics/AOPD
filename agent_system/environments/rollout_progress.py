"""History-aware commit selection and deterministic no-progress tracking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HistoryAwareSelection:
    selected_index: int
    appearance_index: int
    nonrepeat_alternative_available: bool
    nonrepeat_preference_applied: bool
    selection_type: str


def _preferred_argmax_indices(
    scores: Sequence[float],
    canonical_actions: Sequence[str],
    previous_canonical_action: str | None,
) -> tuple[list[int], bool]:
    maximum = max(scores)
    top = [index for index, score in enumerate(scores) if score == maximum]
    if previous_canonical_action is None:
        return top, False
    repeated = [index for index in top if canonical_actions[index] == previous_canonical_action]
    alternatives = [index for index in top if canonical_actions[index] != previous_canonical_action]
    if repeated and alternatives:
        return alternatives, True
    return top, False


def select_history_aware_with_appearance_counterfactual(
    selection_scores: Sequence[float],
    appearance_scores: Sequence[float],
    canonical_actions: Sequence[str],
    previous_canonical_action: str | None,
    rng: random.Random,
) -> HistoryAwareSelection:
    """Uniformly choose tied maxima, preferring a non-repeated top action."""
    if not selection_scores:
        raise ValueError("cannot select from an empty candidate group")
    if not (len(selection_scores) == len(appearance_scores) == len(canonical_actions)):
        raise ValueError("selection scores and canonical actions must align")

    selected_pool, selected_preferred = _preferred_argmax_indices(
        selection_scores,
        canonical_actions,
        previous_canonical_action,
    )
    appearance_pool, _ = _preferred_argmax_indices(
        appearance_scores,
        canonical_actions,
        previous_canonical_action,
    )
    counterfactual_rng = random.Random(0)
    counterfactual_rng.setstate(rng.getstate())
    selected_index = rng.choice(selected_pool)
    appearance_index = counterfactual_rng.choice(appearance_pool)
    return HistoryAwareSelection(
        selected_index=selected_index,
        appearance_index=appearance_index,
        nonrepeat_alternative_available=selected_preferred,
        nonrepeat_preference_applied=selected_preferred,
        selection_type=("uniform_argmax_nonrepeat_preferred" if selected_preferred else "uniform_argmax"),
    )


def validate_no_progress_config(*, enabled: bool, max_rounds: int, min_repeat_streak: int) -> tuple[bool, int, int]:
    """Validate the deliberately bounded no-progress resampling protocol."""
    enabled = bool(enabled)
    max_rounds = int(max_rounds)
    min_repeat_streak = int(min_repeat_streak)
    if max_rounds not in {0, 1}:
        raise ValueError("no-progress resampling supports only max_rounds=0 or 1")
    if enabled != (max_rounds == 1):
        raise ValueError("no-progress resampling requires enabled=true exactly when max_rounds=1")
    if min_repeat_streak < 1:
        raise ValueError("no-progress minimum repeat streak must be positive")
    return enabled, max_rounds, min_repeat_streak


@dataclass
class NoProgressTracker:
    last_action_kind: str | None = None
    last_canonical_action: str | None = None
    last_observation: str | None = None
    repeat_streak: int = 0

    def reset(self) -> None:
        self.last_action_kind = None
        self.last_canonical_action = None
        self.last_observation = None
        self.repeat_streak = 0

    def record(
        self,
        *,
        action_kind: str,
        canonical_action: str,
        observation: str,
    ) -> None:
        if action_kind != "tool":
            self.reset()
            return
        repeated_without_progress = self.last_action_kind == "tool" and self.last_canonical_action == canonical_action and self.last_observation == observation
        self.repeat_streak = self.repeat_streak + 1 if repeated_without_progress else 1
        self.last_action_kind = "tool"
        self.last_canonical_action = canonical_action
        self.last_observation = observation

    def inspect_candidate_actions(
        self,
        *,
        action_kinds: Sequence[str],
        canonical_actions: Sequence[str],
        enabled: bool,
        min_repeat_streak: int,
    ) -> dict[str, object]:
        if len(action_kinds) != len(canonical_actions):
            raise ValueError("candidate action kinds and canonical actions must align")
        unique = len(set(canonical_actions))
        collapsed_repeat = bool(enabled and self.repeat_streak >= min_repeat_streak and action_kinds and all(kind == "tool" for kind in action_kinds) and unique == 1 and canonical_actions[0] == self.last_canonical_action)
        return {
            "trigger": collapsed_repeat,
            "repeat_streak": self.repeat_streak,
            "candidate_unique_action_count": unique,
            "repeated_canonical_action": (self.last_canonical_action if collapsed_repeat else None),
        }
