"""Canonical prompt for independent Codex AWM semantic reviewers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .judgments import JUDGMENT_PROTOCOL_VERSION

REVIEWER_PROMPT_PROTOCOL_VERSION = 1

REVIEWER_INSTRUCTIONS = """You are one of two independent reviewers of an AgentWorldModel task.

Judge only the attached evidence packet. Do not infer task quality from the expert's success/failure label alone. Determine whether the task, required environment path, and code verifier form a reliable training/evaluation instance.

Key rules:
- A policy mistake is healthy data. Use confirmed_policy_failure when the environment and verifier are usable and the recorded failure is attributable to the policy.
- An environment bug excludes a task only when it affects a required task path/state/verifier. If the bug is avoidable and a healthy path can complete the task, mark environment_semantic_bug + avoidable; it remains includable.
- Use verifier_false_negative when a correct final state fails; verifier_false_positive when an incorrect final state passes.
- Use task_infeasible only when no valid sequence using exposed tools can satisfy the stated task and verifier.
- Use uncertain whenever the packet cannot support a high-confidence determination.
- Cite packet JSON pointers in evidence_refs. For exclusion, select one or more exact cohort_keys provided by the packet; do not invent keys.
- Do not inspect or refer to the other reviewer. Return JSON only, with exactly the requested fields.
"""


def judgment_template(task_id: str, slot: str, evidence_sha256: str) -> dict[str, Any]:
    return {
        "protocol_version": JUDGMENT_PROTOCOL_VERSION,
        "task_id": task_id,
        "review_slot": slot,
        "evidence_sha256": evidence_sha256,
        "verdict": "uncertain",
        "path_relevance": "uncertain",
        "confidence": 0.0,
        "evidence_refs": ["/fresh_replay/verify"],
        "cohort_keys": [],
        "rationale": "replace with evidence-grounded rationale",
        "required_path_explanation": "explain whether the observed issue is required, avoidable, or not applicable",
    }


def render_review_prompt(queue_item: Mapping[str, Any], evidence_path: Path) -> str:
    template = judgment_template(
        str(queue_item["task_id"]),
        str(queue_item["review_slot"]),
        str(queue_item["evidence_sha256"]),
    )
    return (
        f"{REVIEWER_INSTRUCTIONS}\n"
        f"Reviewer slot: {queue_item['review_slot']}\n"
        f"Evidence packet: {evidence_path.resolve()}\n"
        f"Evidence SHA-256: {queue_item['evidence_sha256']}\n"
        f"Judgment output: {Path(str(queue_item['judgment_path'])).resolve()}\n\n"
        "Return exactly this JSON shape (replace values):\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
