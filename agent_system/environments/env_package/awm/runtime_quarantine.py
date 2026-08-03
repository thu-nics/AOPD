"""Deterministic runtime quarantine for AWM environment defects.

The semantic training path deliberately does not qualify tasks with an expert
trajectory.  Consequently, some defects are first observed after a student has
already visited a state.  This module keeps the defect test conservative and
provider-free: only strong environment-error payloads are replay candidates,
and a task is quarantined only when the exact action prefix reproduces the same
stable signature after a fresh reset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import ray

RUNTIME_QUARANTINE_PROTOCOL_VERSION = 1

_STATUS_RE = re.compile(r"Status code:\s*([45]\d\d)", re.IGNORECASE)
_ROUTE_COLLISION_RE = re.compile(
    r'"loc"\s*:\s*\[\s*"path".*?"input"\s*:\s*"([^"\\]+)"',
    re.DOTALL,
)
_SPACE_RE = re.compile(r"\s+")
_VOLATILE_PATH_RE = re.compile(r"/tmp/[A-Za-z0-9_./-]+")


def task_id(scenario: str, task_idx: int) -> str:
    return f"{scenario}:{int(task_idx)}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_text(value: Any) -> str:
    text = str(value or "")
    text = _VOLATILE_PATH_RE.sub("<tmp-path>", text)
    return _SPACE_RE.sub(" ", text).strip()


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, str):
        return _stable_text(value)
    return value


def replay_observation_signature(payload: Mapping[str, Any]) -> str:
    """Hash a replay observation after removing known volatile text."""
    digest = hashlib.sha256(_canonical_json(_stable_value(payload)).encode()).hexdigest()
    return f"awm-replay-observation-v{RUNTIME_QUARANTINE_PROTOCOL_VERSION}:{digest}"


def _http_status(payload: Mapping[str, Any]) -> int | None:
    for key in ("status_code", "http_status", "http_status_code"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    match = _STATUS_RE.search(str(payload.get("error") or ""))
    return int(match.group(1)) if match else None


def _is_route_collision_422(payload: Mapping[str, Any]) -> bool:
    """Recognize FastAPI static-route shadowing, not ordinary bad arguments."""
    if _http_status(payload) != 422:
        return False
    error = str(payload.get("error") or "")
    match = _ROUTE_COLLISION_RE.search(error)
    if match is None:
        return False
    route_segment = match.group(1)
    return bool(re.search(r"[A-Za-z-]", route_segment))


def deterministic_error_signature(
    payload: Mapping[str, Any],
    *,
    phase: str,
    tool_name: str | None = None,
) -> str | None:
    """Return a stable signature only for strong deterministic-defect evidence."""
    reward_type = str(payload.get("reward_type") or "")
    status = _http_status(payload)
    if phase == "tool":
        if not ((status is not None and status >= 500) or _is_route_collision_422(payload)):
            return None
    elif phase == "verify":
        if reward_type not in {"server_error", "no_verifier"}:
            return None
    else:
        raise ValueError(f"unknown AWM runtime error phase: {phase!r}")

    signature_payload = {
        "phase": phase,
        "tool_name": tool_name,
        "reward_type": reward_type,
        "http_status": status,
        "error": _stable_text(payload.get("error")),
    }
    digest = hashlib.sha256(_canonical_json(signature_payload).encode()).hexdigest()
    return f"awm-runtime-v{RUNTIME_QUARANTINE_PROTOCOL_VERSION}:{digest}"


def infrastructure_error(payload: Mapping[str, Any], *, phase: str) -> bool:
    """Classify errors that must mask/retry rather than become policy outcomes."""
    reward_type = str(payload.get("reward_type") or "")
    if reward_type in {
        "timeout",
        "no_verifier",
        "judge_error",
        "runtime_exception",
    }:
        return True
    if phase == "verify":
        return reward_type == "server_error"
    if phase != "tool":
        raise ValueError(f"unknown AWM runtime error phase: {phase!r}")
    status = _http_status(payload)
    if status is None:
        return reward_type == "server_error"
    # Ordinary 4xx responses are model/domain errors. The one exception is
    # AWM's known FastAPI static-route shadowing signature.
    return status >= 500 or _is_route_collision_422(payload)


@ray.remote
class AWMRuntimeQuarantineRegistry:
    """Single-writer, run-local registry shared by all AWM training workers."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            torn_tail = False
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == len(lines) - 1:
                        torn_tail = True
                        break
                    raise RuntimeError(f"invalid AWM runtime-quarantine JSONL record {index + 1}") from exc
                if record.get("protocol_version") != RUNTIME_QUARANTINE_PROTOCOL_VERSION:
                    raise RuntimeError("AWM runtime-quarantine protocol mismatch")
                item_task_id = str(record["task_id"])
                previous = self.records.get(item_task_id)
                if previous is not None and previous.get("signature") != record.get("signature"):
                    raise RuntimeError(f"conflicting AWM runtime-quarantine records for {item_task_id}")
                self.records[item_task_id] = record
            if torn_tail:
                self.path.write_text(
                    "".join(_canonical_json(record) + "\n" for record in self.records.values()),
                    encoding="utf-8",
                )

    def is_quarantined(self, item_task_id: str) -> bool:
        return str(item_task_id) in self.records

    def record(self, record: Mapping[str, Any]) -> bool:
        output = dict(record)
        output["protocol_version"] = RUNTIME_QUARANTINE_PROTOCOL_VERSION
        item_task_id = str(output["task_id"])
        previous = self.records.get(item_task_id)
        if previous is not None:
            if previous.get("signature") != output.get("signature"):
                raise RuntimeError(f"conflicting AWM runtime-quarantine signature for {item_task_id}")
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(output) + "\n")
            handle.flush()
        self.records[item_task_id] = output
        return True

    def stats(self) -> dict[str, int]:
        return {"quarantined_tasks": len(self.records)}
