"""Shared contract for task-scoped static-feasibility screening."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STATIC_FEASIBILITY_PROTOCOL_VERSION = 1
STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS = 32_768
STATIC_FEASIBILITY_MEMBERSHIP_RULE = "deterministic pass AND static feasibility judge label healthy; confidence is diagnostic only"
STATIC_FEASIBILITY_RETRYABLE_STATUS_REASONS = frozenset({"judge_infrastructure_exhausted"})
STATIC_FEASIBILITY_DECISION_RULES = """Return exactly one JSON object with:
- label: healthy, environment_or_verifier_failure, or uncertain
- confidence: integer 0 to 100, for diagnostics only
- rationale: concise string
- evidence: list of concise strings grounded in the supplied evidence

Make one task-scoped decision:
- Use healthy when at least one legal sequence of exposed native tool calls can satisfy
  every explicit task constraint and the verifier can reliably distinguish that outcome
  from a materially incorrect or no-action state.
- A legal workaround is acceptable only when it preserves every explicit constraint,
  required object identity, and temporal requirement.
- A defect in an unused tool, unrelated environment capability, or harmless schema or
  documentation oddity must not reject the task.
- Use environment_or_verifier_failure only for a concrete task-relevant defect:
  contradictory requirements, missing required initial data, an unreachable requested
  state, a required operation with no legal implementation, or a verifier that can reject
  a correct outcome or accept a materially incorrect outcome.
- Use uncertain only when the supplied static evidence is insufficient to decide.

Do not judge task difficulty or agent quality. There is deliberately no expert trajectory.
Confidence is diagnostic and never changes the membership rule."""
STATIC_FEASIBILITY_LABELS = frozenset(
    {
        "healthy",
        "environment_or_verifier_failure",
        "uncertain",
    }
)


def static_feasibility_instruction(
    *,
    environment_description: str,
    evidence_notes: str,
) -> str:
    """Combine environment-specific evidence semantics with shared decisions."""
    return environment_description.strip() + "\n\n" + evidence_notes.strip() + "\n\n" + STATIC_FEASIBILITY_DECISION_RULES


def static_feasibility_generation_settings(
    *,
    max_tokens: int = STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Return the generation settings shared by every feasibility judge."""
    if isinstance(max_tokens, bool) or int(max_tokens) <= 0:
        raise ValueError("static-feasibility max_tokens must be positive")
    return {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }


def static_feasibility_review_is_complete(review: Any) -> bool:
    """Return whether a review is reusable under the sole current protocol."""
    if not isinstance(review, Mapping):
        return False
    if review.get("static_feasibility_protocol_version") != STATIC_FEASIBILITY_PROTOCOL_VERSION:
        return False
    reason = str(review.get("status_reason") or "")
    if reason in STATIC_FEASIBILITY_RETRYABLE_STATUS_REASONS:
        return False
    judge = review.get("judge")
    if isinstance(judge, Mapping):
        return judge.get("protocol_version") == STATIC_FEASIBILITY_PROTOCOL_VERSION
    return reason == "static_evidence_failure"
