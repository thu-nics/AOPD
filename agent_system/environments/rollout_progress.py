"""History-aware commit selection and deterministic no-progress tracking."""

from __future__ import annotations

import math
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


def validate_progress_config(
    *,
    repeat_reward_cap_enabled: bool,
    repeat_reward_cap_min_streak: int,
    repeat_reward_cap_value: float,
    repeat_termination_enabled: bool,
    repeat_termination_max_streak: int,
) -> tuple[bool, int, float, bool, int]:
    """Validate the bounded no-progress intervention protocol."""
    cap_enabled = bool(repeat_reward_cap_enabled)
    cap_streak = int(repeat_reward_cap_min_streak)
    cap_value = float(repeat_reward_cap_value)
    termination_enabled = bool(repeat_termination_enabled)
    termination_streak = int(repeat_termination_max_streak)
    if cap_streak < 2:
        raise ValueError("repeat reward cap minimum streak must be at least two")
    if not math.isfinite(cap_value):
        raise ValueError("repeat reward cap value must be finite")
    if termination_streak < 2:
        raise ValueError("repeat termination maximum streak must be at least two")
    if cap_enabled and termination_enabled and termination_streak < cap_streak:
        raise ValueError("repeat termination streak must not precede the repeat reward cap")
    return cap_enabled, cap_streak, cap_value, termination_enabled, termination_streak


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

    def prospective_repeat_flags(
        self,
        *,
        action_kinds: Sequence[str],
        canonical_actions: Sequence[str],
        min_streak: int,
    ) -> list[bool]:
        """Mark candidates that would extend a proven no-progress repeat streak."""
        if len(action_kinds) != len(canonical_actions):
            raise ValueError("candidate action kinds and canonical actions must align")
        threshold = int(min_streak)
        if threshold < 2:
            raise ValueError("prospective repeat threshold must be at least two")
        active = self.repeat_streak + 1 >= threshold
        return [
            bool(active and kind == "tool" and canonical == self.last_canonical_action)
            for kind, canonical in zip(
                action_kinds,
                canonical_actions,
                strict=True,
            )
        ]

    def reached(self, max_streak: int) -> bool:
        threshold = int(max_streak)
        if threshold < 2:
            raise ValueError("repeat termination threshold must be at least two")
        return self.repeat_streak >= threshold
