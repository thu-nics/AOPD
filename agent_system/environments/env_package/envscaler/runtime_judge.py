"""Code-augmented adjudication for EnvScaler tool execution exceptions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent_system.environments.env_package.awm.runtime.judge import (
    validate_runtime_judge_verdict,
)

ENVSCALER_RUNTIME_JUDGE_PROTOCOL_VERSION = 2
ENVSCALER_RUNTIME_JUDGE_SCOPE = "envscaler_tool_exception"
ENVSCALER_RUNTIME_JUDGE_INSTRUCTION = """You classify one failed EnvScaler agent tool execution using the supplied task, visible conversation, public tool schemas, failed action, local state, generated environment source, and Python exception traceback.

The failed action has already passed the public tool JSON schema. The local environment state was snapshotted before execution and restored after the exception.

error_class:
- policy_execution_error: the chosen action violates the current public tool/state contract. Examples include guessed or missing IDs, duplicate creation forbidden by current state, missing prerequisites, invalid state transitions, or values that are schema-valid but invalid for the current state.
- infrastructure_error: the action is a reasonable state-valid use of the public tool, but the generated environment implementation is defective. Examples include missing internal attributes, broken helper logic, inconsistent generated code, or an exception that valid public inputs should not cause.
- uncertain: the supplied evidence cannot distinguish the two reliably.

Critical rules:
- Do not classify from the Python exception type alone; inspect the public contract, current state, source, and traceback together.
- An irrelevant or unnecessary but schema-valid tool call is not automatically a policy_execution_error. Semantic relevance is handled by the teacher reward. Use policy_execution_error only when the action violates the tool/state contract.
- A different successful path does not prove that the failed action is invalid; equivalent valid public-tool paths are allowed.
- Base the cause on the supplied state, public contract, source, and actual traceback. Tool-description examples are not database observations.
  Do not infer nonexistent IDs, duplicate records, or omitted lookups without affirmative evidence. A possible explanation is not an established cause;
  return uncertain if the supplied evidence cannot distinguish policy error from an implementation defect.
- Do not judge whether the overall task is complete or whether the action matches the checklist.
- post_error_state must be "unchanged" because the runtime restores the exact pre-call local snapshot after every exception.

Return exactly this JSON shape and no other keys:
{"error_class":"policy_execution_error|infrastructure_error|uncertain","classification_confidence":95,"post_error_state":"unchanged","rationale":"short explanation"}
"""
ENVSCALER_RUNTIME_JUDGE_PROMPT_HASH = hashlib.sha256(ENVSCALER_RUNTIME_JUDGE_INSTRUCTION.encode()).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=repr)
    return repr(value)


def build_envscaler_runtime_judge_evidence(
    *,
    source_identity: Mapping[str, Any],
    task: Mapping[str, Any],
    environment: Mapping[str, Any],
    visible_chat: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    failed_action: Mapping[str, Any],
    exception_type: str,
    exception_message: str,
    traceback_text: str,
    state_before: Mapping[str, Any],
    state_at_exception: Mapping[str, Any],
) -> dict[str, Any]:
    """Build stable, JSON-safe source/state evidence for one failed call."""
    action_name = str(failed_action.get("name") or "")
    relevant_tools = [dict(tool) for tool in tools if str(tool.get("name") or tool.get("function", {}).get("name") or "") == action_name]
    history = [dict(message) for message in visible_chat]
    if len(history) > 10:
        history = history[:2] + history[-8:]
    checklist = [
        {
            "check_item": str(item.get("check_item") or ""),
            "check_func": str(item.get("check_func") or ""),
        }
        for item in task.get("checklist_with_func") or []
    ]
    return _json_safe(
        {
            "source_identity": dict(source_identity),
            "env_id": str(task.get("env_id") or ""),
            "task_id": str(task.get("task_id") or ""),
            "task": str(task.get("task") or ""),
            "visible_chat": history,
            "public_tool": relevant_tools,
            "failed_action": dict(failed_action),
            "exception": {
                "type": str(exception_type),
                "message": str(exception_message),
                "traceback": str(traceback_text)[-12000:],
            },
            "state_before": dict(state_before),
            "state_at_exception": dict(state_at_exception),
            "state_restored": True,
            "environment_source": str(environment.get("env_class_code") or ""),
            "checklist": checklist,
        }
    )


def envscaler_runtime_judge_fingerprint(
    *,
    model: str,
    decoding_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    payload = {
        "scope": ENVSCALER_RUNTIME_JUDGE_SCOPE,
        "protocol_version": ENVSCALER_RUNTIME_JUDGE_PROTOCOL_VERSION,
        "prompt_hash": ENVSCALER_RUNTIME_JUDGE_PROMPT_HASH,
        "model": str(model),
        "decoding_config": dict(decoding_config),
        "evidence": dict(evidence),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_envscaler_runtime_judge_verdict(value: Any) -> dict[str, Any]:
    verdict = validate_runtime_judge_verdict(value)
    if verdict["post_error_state"] != "unchanged":
        raise ValueError("EnvScaler runtime judge must return post_error_state='unchanged'")
    return verdict
