"""Strict reviewer judgments and conservative two-reviewer consensus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

JUDGMENT_PROTOCOL_VERSION = 1
REVIEW_SLOTS = frozenset({"A", "B"})
VERDICTS = frozenset(
    {
        "healthy_success",
        "confirmed_policy_failure",
        "task_infeasible",
        "environment_semantic_bug",
        "verifier_false_negative",
        "verifier_false_positive",
        "uncertain",
    }
)
PATH_RELEVANCE = frozenset({"required", "avoidable", "not_applicable", "uncertain"})
EXCLUSION_VERDICTS = frozenset(
    {
        "task_infeasible",
        "verifier_false_negative",
        "verifier_false_positive",
    }
)
MIN_EXCLUSION_CONFIDENCE = 0.90


def exclusion_recommended(verdict: str, path_relevance: str) -> bool:
    return verdict in EXCLUSION_VERDICTS or (verdict == "environment_semantic_bug" and path_relevance == "required")


def validate_judgment(
    value: Mapping[str, Any],
    *,
    task_id: str | None = None,
    slot: str | None = None,
    evidence_sha256: str | None = None,
    allowed_cohort_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one human/Codex-produced JSON judgment."""
    required = {
        "protocol_version",
        "task_id",
        "review_slot",
        "evidence_sha256",
        "verdict",
        "path_relevance",
        "confidence",
        "evidence_refs",
        "cohort_keys",
        "rationale",
        "required_path_explanation",
    }
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ValueError(f"judgment fields mismatch: missing={missing!r}, unknown={unknown!r}")
    if value["protocol_version"] != JUDGMENT_PROTOCOL_VERSION:
        raise ValueError("judgment protocol mismatch")
    actual_task_id = str(value["task_id"])
    actual_slot = str(value["review_slot"])
    actual_evidence_sha = str(value["evidence_sha256"])
    if task_id is not None and actual_task_id != task_id:
        raise ValueError("judgment task ID mismatch")
    if actual_slot not in REVIEW_SLOTS or (slot is not None and actual_slot != slot):
        raise ValueError("judgment review slot mismatch")
    if evidence_sha256 is not None and actual_evidence_sha != evidence_sha256:
        raise ValueError("judgment evidence hash mismatch")
    verdict = str(value["verdict"])
    relevance = str(value["path_relevance"])
    if verdict not in VERDICTS:
        raise ValueError(f"unknown judgment verdict: {verdict!r}")
    if relevance not in PATH_RELEVANCE:
        raise ValueError(f"unknown path relevance: {relevance!r}")
    if verdict == "environment_semantic_bug" and relevance not in {"required", "avoidable", "uncertain"}:
        raise ValueError("environment-semantic-bug path relevance must be required, avoidable, or uncertain")
    if verdict == "uncertain" and relevance != "uncertain":
        raise ValueError("an uncertain verdict requires uncertain path relevance")
    if verdict not in {"environment_semantic_bug", "uncertain"} and relevance != "not_applicable":
        raise ValueError("non-environment verdicts require not_applicable path relevance")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("judgment confidence must be numeric")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("judgment confidence must be in [0, 1]")
    evidence_refs = value["evidence_refs"]
    if not isinstance(evidence_refs, list) or not evidence_refs or any(not isinstance(item, str) or not item.strip() for item in evidence_refs):
        raise ValueError("judgment requires non-empty string evidence_refs")
    cohort_keys = value["cohort_keys"]
    if not isinstance(cohort_keys, list) or any(not isinstance(item, str) or not item.strip() for item in cohort_keys):
        raise ValueError("judgment cohort_keys must be a list of strings")
    if len(cohort_keys) != len(set(cohort_keys)):
        raise ValueError("judgment cohort_keys must be unique")
    if allowed_cohort_keys is not None and not set(cohort_keys) <= set(allowed_cohort_keys):
        raise ValueError("judgment cites an unknown cohort key")
    rationale = str(value["rationale"]).strip()
    explanation = str(value["required_path_explanation"]).strip()
    if not rationale or not explanation:
        raise ValueError("judgment rationale and required-path explanation are required")
    if exclusion_recommended(verdict, relevance) and not cohort_keys:
        raise ValueError("an exclusion judgment must cite at least one packet cohort key")
    return {
        "protocol_version": JUDGMENT_PROTOCOL_VERSION,
        "task_id": actual_task_id,
        "review_slot": actual_slot,
        "evidence_sha256": actual_evidence_sha,
        "verdict": verdict,
        "path_relevance": relevance,
        "confidence": confidence,
        "evidence_refs": list(evidence_refs),
        "cohort_keys": list(cohort_keys),
        "rationale": rationale,
        "required_path_explanation": explanation,
    }


@dataclass(frozen=True)
class Consensus:
    membership: str
    reason: str
    verdict: str | None
    path_relevance: str | None
    cohort_keys: tuple[str, ...] = ()


def reviewer_consensus(judgments: Sequence[Mapping[str, Any]]) -> Consensus:
    """Resolve two independent reviews; disagreement is always pending."""
    if len(judgments) != 2:
        return Consensus("pending", "requires_two_independent_reviews", None, None)
    by_slot = {str(item["review_slot"]): item for item in judgments}
    if set(by_slot) != REVIEW_SLOTS:
        return Consensus("pending", "requires_distinct_A_and_B_reviews", None, None)
    first, second = by_slot["A"], by_slot["B"]
    if first["evidence_sha256"] != second["evidence_sha256"]:
        return Consensus("pending", "review_evidence_drift", None, None)
    if first["verdict"] != second["verdict"]:
        return Consensus("pending", "reviewer_verdict_disagreement", None, None)
    verdict = str(first["verdict"])
    if verdict == "uncertain":
        return Consensus("pending", "reviewers_uncertain", verdict, "uncertain")
    if verdict == "environment_semantic_bug":
        if first["path_relevance"] != second["path_relevance"]:
            return Consensus("pending", "reviewer_path_relevance_disagreement", verdict, None)
        relevance = str(first["path_relevance"])
    else:
        relevance = "not_applicable"
    if exclusion_recommended(verdict, relevance):
        if min(float(first["confidence"]), float(second["confidence"])) < MIN_EXCLUSION_CONFIDENCE:
            return Consensus("pending", "exclusion_confidence_below_0.90", verdict, relevance)
        cohort_keys = tuple(sorted(set(first["cohort_keys"]) & set(second["cohort_keys"])))
        if not cohort_keys:
            return Consensus("pending", "reviewers_share_no_exclusion_cohort_key", verdict, relevance)
        return Consensus("excluded", f"consensus:{verdict}:{relevance}", verdict, relevance, cohort_keys)
    if verdict in {"healthy_success", "confirmed_policy_failure"}:
        return Consensus("included", f"consensus:{verdict}", verdict, relevance)
    if verdict == "environment_semantic_bug" and relevance == "avoidable":
        return Consensus("included", "consensus:avoidable_environment_bug", verdict, relevance)
    return Consensus("pending", "unsupported_or_uncertain_consensus", verdict, relevance)
