"""Minimal runtime infrastructure-error isolation for AWM semantic training.

Strong infrastructure errors end and mask only the current state group. They are
recorded for diagnostics and never become a persistent task blacklist.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import ray

RUNTIME_FAILURE_PROTOCOL_VERSION = 2

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
    elif phase in {"verify", "done"}:
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
    return f"awm-runtime-v{RUNTIME_FAILURE_PROTOCOL_VERSION}:{digest}"


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
    if phase in {"verify", "done"}:
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
class AWMRuntimeFailureRecorder:
    """Append-only, run-local diagnostics; records never filter future tasks."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_count = 0
        if self.path.is_file():
            valid_lines: list[str] = []
            lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
            repair = False
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                        repair = True
                        break
                    raise RuntimeError(f"invalid AWM runtime-failure JSONL record {index + 1}") from exc
                if record.get("protocol_version") != RUNTIME_FAILURE_PROTOCOL_VERSION:
                    raise RuntimeError("AWM runtime-failure protocol mismatch")
                if record.get("status") != "masked":
                    raise RuntimeError("invalid AWM runtime-failure status")
                valid_lines.append(_canonical_json(record))
                self.record_count += 1
            if lines and not lines[-1].endswith(("\n", "\r")):
                repair = True
            if repair:
                self.path.write_text(
                    "".join(item + "\n" for item in valid_lines),
                    encoding="utf-8",
                )

    def record(self, record: Mapping[str, Any]) -> None:
        output = dict(record)
        output["protocol_version"] = RUNTIME_FAILURE_PROTOCOL_VERSION
        if output.get("status") != "masked":
            raise ValueError("invalid AWM runtime-failure status")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(output) + "\n")
            handle.flush()
        self.record_count += 1

    def stats(self) -> dict[str, int]:
        return {"runtime_failure_records": self.record_count}
