"""Verifier-reliability screening for deterministic-pass EnvScaler tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from http.client import HTTPException
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from agent_system.environments.env_package.awm.runtime.actions import openai_tools
from agent_system.environments.static_feasibility import (
    STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS,
    STATIC_FEASIBILITY_LABELS,
    STATIC_FEASIBILITY_MEMBERSHIP_RULE,
    STATIC_FEASIBILITY_PROTOCOL_VERSION,
    static_feasibility_generation_settings,
    static_feasibility_instruction,
    static_feasibility_review_is_complete,
)

from .filtering import FILTER_PROTOCOL_VERSION, training_row
from .source import (
    DEFAULT_SOURCE_ROOT,
    build_environment_instance,
    checker_summary,
    evaluate_checkers,
    load_envscaler_source,
    sha256_file,
    state_dict,
    validate_tool_contract,
)

JUDGE_MAX_TOKENS = STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS
VALID_LABELS = STATIC_FEASIBILITY_LABELS

JUDGE_INSTRUCTION = static_feasibility_instruction(
    environment_description="You audit one EnvScaler task using static, code-augmented evidence.",
    evidence_notes=(
        "The supplied initial_state is authoritative. EnvScaler constructs the "
        "environment and then injects every init_config field with setattr; the "
        "adapter reproduces that behavior. The no-action checker result is diagnostic "
        "only: an unchanged initial state is normally incomplete, and an individual "
        "checker passing initially is a defect only when it makes a requested change "
        "unverifiable."
    ),
)


class DeepSeekScreeningClient:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key_env: str,
        timeout_seconds: float,
        max_retries: int,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.api_key = api_key
        self.model = str(model)
        self.url = str(api_base).rstrip("/") + "/chat/completions"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        if self.max_retries <= 0 or self.max_tokens <= 0:
            raise ValueError("screening max_retries and max_tokens must be positive")
        self._stats = {
            "requests": 0,
            "retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._lock = threading.Lock()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode())
                usage = result.get("usage") or {}
                with self._lock:
                    self._stats["requests"] += 1
                    self._stats["retries"] += attempt
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ):
                        self._stats[key] += int(usage.get(key, 0) or 0)
                return result
            except (
                OSError,
                TimeoutError,
                HTTPException,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                if isinstance(exc, HTTPError):
                    try:
                        body = exc.read().decode(errors="replace")
                    except Exception:
                        body = ""
                    last_error = RuntimeError(f"{exc}; response_body={body[:2048]!r}")
                else:
                    last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(16.0, 2.0**attempt) + random.random() * 0.2)
        raise RuntimeError(f"DeepSeek request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def message(response: Mapping[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek response has no choices")
        return dict(choices[0].get("message") or {})

    @staticmethod
    def _parse_judge_content(content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("screening judge returned empty content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as direct_error:
            # Be tolerant of an otherwise valid object wrapped in prose or a
            # Markdown fence. raw_decode avoids pairing the first opening brace
            # with an unrelated later closing brace.
            decoder = json.JSONDecoder()
            value = None
            for offset, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content[offset:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    value = candidate
                    break
            if value is None:
                raise direct_error
        if not isinstance(value, dict):
            raise TypeError("screening judge response must be an object")
        return value

    @staticmethod
    def _validate_judge_value(value: Mapping[str, Any]) -> dict[str, Any]:
        label = value.get("label")
        confidence = value.get("confidence")
        if label not in VALID_LABELS:
            raise ValueError(f"invalid screening judge label: {label!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("screening judge confidence must be integer 0..100")
        rationale = value.get("rationale")
        evidence_items = value.get("evidence")
        if not isinstance(rationale, str) or not isinstance(evidence_items, list):
            raise ValueError("screening judge rationale/evidence malformed")
        return {
            "label": label,
            "confidence": confidence,
            "rationale": rationale,
            "evidence": [str(item) for item in evidence_items],
        }

    def judge(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_INSTRUCTION},
                {
                    "role": "user",
                    # Preserve insertion order so tasks from one environment
                    # share the largest possible provider-cache prefix.
                    "content": json.dumps(evidence, ensure_ascii=False),
                },
            ],
            **static_feasibility_generation_settings(max_tokens=self.max_tokens),
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        cumulative_usage: dict[str, int] = {}
        structured_errors: list[str] = []
        for attempt in range(self.max_retries):
            try:
                response = self.post(payload)
            except Exception as exc:
                error = RuntimeError(f"DeepSeek judge request failed: {exc}")
                # Preserve paid structured responses from earlier attempts even
                # if a later retry fails before returning another response.
                error.usage = cumulative_usage  # type: ignore[attr-defined]
                error.structured_errors = structured_errors  # type: ignore[attr-defined]
                raise error from exc
            for key, raw_value in (response.get("usage") or {}).items():
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                cumulative_usage[key] = cumulative_usage.get(key, 0) + int(raw_value)
            try:
                content = str(self.message(response).get("content") or "")
                parsed = self._validate_judge_value(self._parse_judge_content(content))
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                structured_errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 < self.max_retries:
                    time.sleep(min(4.0, 2.0**attempt) + random.random() * 0.2)
                    continue
                error = RuntimeError(f"DeepSeek returned no valid structured judge response after {self.max_retries} attempts: {structured_errors[-1]}")
                error.usage = cumulative_usage  # type: ignore[attr-defined]
                error.structured_errors = structured_errors  # type: ignore[attr-defined]
                raise error from exc
            return {
                "protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
                **parsed,
                "usage": cumulative_usage,
                "structured_response_attempts": attempt + 1,
                "structured_response_errors": structured_errors,
            }
        raise AssertionError("positive max_retries must execute at least one judge attempt")


def build_code_evidence(
    task_index: int,
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    deterministic_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reproducible static evidence without running an expert policy."""
    source = load_envscaler_source(source_root)
    task = deepcopy(source.tasks[int(task_index)])
    environment = deepcopy(source.environments[str(task["env_id"])])
    runtime = build_environment_instance(environment, task)
    tools = validate_tool_contract(environment, runtime)
    initial_state = state_dict(runtime)
    initial_results = evaluate_checkers(task, initial_state, initial_state)
    initial_summary = checker_summary(initial_results)
    return {
        "runtime_initialization_protocol": {
            "authoritative_state": "task.initial_state",
            "native_helper": "init_env_instance",
            "steps": [
                "deep-copy init_config",
                "construct class with init_config when supported",
                "setattr(instance, key, deepcopy(value)) for every init_config item",
                "snapshot the resulting instance as task.initial_state",
            ],
            "warning": "Do not infer empty runtime state from the class __init__ body.",
        },
        # Keep environment-static evidence first for API prefix caching.
        "environment": {
            "env_id": str(task["env_id"]),
            "introduction": environment.get("environment_introduction"),
            "constraints_rules": environment.get("constraints_rules") or [],
            "class_name": task.get("env_class_name"),
            "class_code": environment.get("env_class_code"),
            "native_tools": openai_tools(tools),
        },
        "task": {
            "task_index": int(task_index),
            "task_id": str(task["task_id"]),
            "request": task.get("task"),
            "init_config": task.get("init_config") or {},
            "initial_state": initial_state,
            "checkers": task.get("checklist_with_func") or [],
        },
        "no_action_checker_evidence": {
            "summary": initial_summary,
            "results": [
                {
                    "index": result["index"],
                    "check_item": result["check_item"],
                    "success": result["success"],
                    "result": result["result"],
                    "error": result["error"],
                }
                for result in initial_results
            ],
        },
        "deterministic_audit": dict(deterministic_record or {}),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    task = evidence["task"]
    environment = evidence["environment"]
    checker_evidence = evidence["no_action_checker_evidence"]
    encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "task_id": str(task["task_id"]),
        "env_id": str(environment["env_id"]),
        "tool_names": [str(item["function"]["name"]) for item in environment["native_tools"]],
        "checker_count": int(checker_evidence["summary"]["checker_count"]),
        "initial_checker_passed": int(checker_evidence["summary"]["checker_passed"]),
    }


def screen_one(
    task_index: int,
    *,
    source_root: str | Path,
    client: DeepSeekScreeningClient,
    deterministic_record: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        evidence = build_code_evidence(
            task_index,
            source_root=source_root,
            deterministic_record=deterministic_record,
        )
        evidence_summary = _evidence_summary(evidence)
    except Exception as exc:
        return {
            "task_index": int(task_index),
            "static_feasibility_protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
            "accepted": False,
            "status": "quarantine",
            "status_reason": "static_evidence_failure",
            "screening_error": f"{type(exc).__name__}: {exc}",
            "evidence_summary": None,
            "judge": None,
        }
    try:
        judge = client.judge(evidence)
    except Exception as exc:
        return {
            "task_index": int(task_index),
            "static_feasibility_protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
            "accepted": None,
            "status": "pending",
            "status_reason": "judge_infrastructure_exhausted",
            "screening_error": f"{type(exc).__name__}: {exc}",
            "judge_attempt_usage": dict(getattr(exc, "usage", {}) or {}),
            "judge_attempt_errors": list(getattr(exc, "structured_errors", []) or []),
            "evidence_summary": evidence_summary,
            "judge": None,
        }
    accepted = judge["label"] == "healthy"
    return {
        "task_index": int(task_index),
        "static_feasibility_protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "accepted": accepted,
        "status": "healthy" if accepted else "quarantine",
        "status_reason": "healthy" if accepted else str(judge["label"]),
        "screening_error": None,
        "evidence_summary": evidence_summary,
        "judge": judge,
    }


def _write_audit(path: Path, records: Mapping[int, Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n" for index in sorted(records)),
        encoding="utf-8",
    )
    temporary.replace(path)


def _aggregate_api_usage(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, int]:
    totals = {
        "successful_responses": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in records.values():
        reviews = [
            *(record.get("health_review_history") or []),
            record.get("health_review") or {},
        ]
        for review in reviews:
            judge = review.get("judge") or {}
            judge_usage = judge.get("usage") or {}
            attempt_usage = review.get("judge_attempt_usage") or {}
            usage = judge_usage or attempt_usage
            if not usage:
                continue
            if judge_usage:
                response_count = int(judge.get("structured_response_attempts", 1) or 1)
            else:
                # Each structured-response error follows an HTTP response whose
                # content could not be parsed. Network attempts without a
                # response carry no usage and are not counted here.
                response_count = len(review.get("judge_attempt_errors") or []) or 1
            totals["successful_responses"] += response_count
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                totals[field] += int(usage.get(field, 0) or 0)
    return totals


def _is_completed_review(record: Mapping[str, Any]) -> bool:
    """Preserve only reviews produced by the sole current protocol."""
    return static_feasibility_review_is_complete(record.get("health_review"))


def _load_validated_deterministic_records(
    audit_path: Path,
    manifest: Mapping[str, Any],
    tasks: tuple[dict[str, Any], ...],
) -> dict[int, dict[str, Any]]:
    """Load the deterministic audit and verify its manifest/task identity."""
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(audit_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        index = int(item["task_index"])
        if index in records:
            raise RuntimeError(f"EnvScaler deterministic audit has duplicate task_index {index} at line {line_number}")
        records[index] = item

    expected_indices = set(range(len(tasks)))
    actual_indices = set(records)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)[:10]
        unexpected = sorted(actual_indices - expected_indices)[:10]
        raise RuntimeError(f"EnvScaler deterministic audit task-index coverage mismatch: missing={missing}, unexpected={unexpected}")

    pass_indices = []
    quarantine = []
    for index, task in enumerate(tasks):
        record = records[index]
        if record.get("protocol_version") != FILTER_PROTOCOL_VERSION:
            raise RuntimeError(f"EnvScaler deterministic audit protocol mismatch at task {index}")
        if record.get("task_id") != task.get("task_id") or record.get("env_id") != task.get("env_id"):
            raise RuntimeError(f"EnvScaler deterministic audit source identity mismatch at task {index}")
        reasons = record.get("reasons")
        if not isinstance(reasons, list):
            raise RuntimeError(f"EnvScaler deterministic audit reasons must be a list at task {index}")
        expected_status = "quarantine" if reasons else "pass"
        if record.get("status") != expected_status:
            raise RuntimeError(f"EnvScaler deterministic audit status/reasons mismatch at task {index}")
        if expected_status == "pass":
            pass_indices.append(index)
        else:
            quarantine.append(
                {
                    "task_index": index,
                    "task_id": str(task["task_id"]),
                    "reasons": reasons,
                }
            )

    expected_manifest_fields = {
        "total_tasks": len(tasks),
        "pass_tasks": len(pass_indices),
        "quarantine_tasks": len(quarantine),
        "pass_task_indices": pass_indices,
        "quarantine": quarantine,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"EnvScaler deterministic manifest/audit mismatch for {field}")
    return records


def _load_existing_screening_records(audit_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not audit_path.is_file():
        return records
    for line_number, line in enumerate(audit_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("health_review"), Mapping):
            raise RuntimeError(f"EnvScaler screening audit line {line_number} has no health_review")
        index = int(record["task_index"])
        if index in records:
            raise RuntimeError(f"EnvScaler screening audit has duplicate task_index {index} at line {line_number}")
        records[index] = record
    return records


def _validate_resume_record(
    index: int,
    record: Mapping[str, Any],
    deterministic_record: Mapping[str, Any],
) -> None:
    for field, expected in deterministic_record.items():
        if record.get(field) != expected:
            raise RuntimeError(f"EnvScaler screening resume deterministic field {field!r} mismatch at task {index}")
    review = record["health_review"]
    if int(review.get("task_index", -1)) != index:
        raise RuntimeError(f"EnvScaler screening resume review identity mismatch at task {index}")

    accepted = review.get("accepted")
    status = str(review.get("status") or "")
    reason = str(review.get("status_reason") or "")
    if accepted is True:
        if status != "healthy" or reason != "healthy":
            raise RuntimeError(f"EnvScaler screening resume healthy verdict mismatch at task {index}")
    elif accepted is False:
        if status != "quarantine" or reason == "healthy":
            raise RuntimeError(f"EnvScaler screening resume quarantine verdict mismatch at task {index}")
    elif accepted is None:
        if status != "pending" or reason != "judge_infrastructure_exhausted":
            raise RuntimeError(f"EnvScaler screening resume pending verdict mismatch at task {index}")
    else:
        raise RuntimeError(f"EnvScaler screening resume accepted flag is invalid at task {index}")

    judge = review.get("judge")
    if isinstance(judge, Mapping):
        DeepSeekScreeningClient._validate_judge_value(judge)
        label = str(judge.get("label") or "")
        expected_reason = "healthy" if label == "healthy" else label
        if label not in VALID_LABELS or reason != expected_reason:
            raise RuntimeError(f"EnvScaler screening resume judge label mismatch at task {index}")
    elif reason not in {
        "static_evidence_failure",
        "judge_infrastructure_exhausted",
    }:
        raise RuntimeError(f"EnvScaler screening resume record has no judge at task {index}")


def _group_task_indices_by_environment(
    indices: list[int],
    tasks: tuple[dict[str, Any], ...],
) -> list[list[int]]:
    """Group tasks by environment while preserving source order within each group."""
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(str(tasks[index]["env_id"]), []).append(index)
    return [groups[env_id] for env_id in sorted(groups)]


def run_screening(
    *,
    source_root: str | Path,
    deterministic_dir: Path,
    output_dir: Path,
    model: str,
    api_base: str,
    api_key_env: str,
    concurrency: int,
    timeout_seconds: float,
    max_retries: int,
    max_tokens: int,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    source = load_envscaler_source(source_root)
    deterministic_manifest_path = deterministic_dir / "deterministic_manifest.json"
    deterministic_audit_path = deterministic_dir / "task_audit.jsonl"
    deterministic_manifest = json.loads(deterministic_manifest_path.read_text())
    if deterministic_manifest.get("protocol_version") != FILTER_PROTOCOL_VERSION:
        raise RuntimeError("EnvScaler deterministic filter protocol mismatch")
    if deterministic_manifest.get("source") != source.identity:
        raise RuntimeError("EnvScaler deterministic filter source identity mismatch")
    if int(deterministic_manifest.get("total_tasks", -1)) != len(source.tasks):
        raise RuntimeError("EnvScaler deterministic filter task-count mismatch")
    deterministic_records = _load_validated_deterministic_records(
        deterministic_audit_path,
        deterministic_manifest,
        source.tasks,
    )
    eligible = [int(value) for value in deterministic_manifest.get("pass_task_indices") or []]
    all_deterministic_eligible = list(eligible)
    if limit is not None:
        eligible = eligible[: int(limit)]
    if not eligible:
        raise RuntimeError("EnvScaler screening has no eligible tasks")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": "envscaler_static_feasibility_screening",
        "protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "source": source.identity,
        "deterministic_manifest": str(deterministic_manifest_path),
        "deterministic_manifest_sha256": sha256_file(deterministic_manifest_path),
        "deterministic_audit_sha256": sha256_file(deterministic_audit_path),
        "model": model,
        "api_base": api_base,
        "api_key_env": api_key_env,
        **static_feasibility_generation_settings(max_tokens=max_tokens),
        "timeout_seconds": float(timeout_seconds),
        "max_retries": int(max_retries),
        "membership_rule": STATIC_FEASIBILITY_MEMBERSHIP_RULE,
        "expert_outcome_membership_gate": False,
        "eligible_task_indices": eligible,
    }
    # Validate API credentials before writing a resumable output identity.
    client = DeepSeekScreeningClient(
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )
    config_path = output_dir / "config.json"
    audit_path = output_dir / "task_audit.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if resume:
        if not config_path.is_file():
            raise RuntimeError("--resume requires an existing config.json")
        existing_config = json.loads(config_path.read_text())
        if existing_config != config:
            raise RuntimeError("EnvScaler screening resume config mismatch")
        existing = _load_existing_screening_records(audit_path)
    elif config_path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite EnvScaler screening output {output_dir}")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    unexpected_existing = set(existing) - set(eligible)
    if unexpected_existing:
        raise RuntimeError("EnvScaler screening audit contains tasks outside the configured eligible set")
    for index, record in existing.items():
        _validate_resume_record(index, record, deterministic_records[index])

    completed = {index: record for index, record in existing.items() if _is_completed_review(record)}
    refreshed_existing = [index for index in existing if index not in completed]
    unseen = [index for index in eligible if index not in existing]
    merged = dict(existing)
    pending = [index for index in eligible if index not in completed]
    print(
        json.dumps(
            {
                "resume_reused": len(completed),
                "resume_refresh_existing": len(refreshed_existing),
                "resume_unseen": len(unseen),
                "pending": len(pending),
                "total": len(eligible),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    groups = _group_task_indices_by_environment(pending, source.tasks)

    def screen_environment(group: list[int]) -> list[tuple[int, dict[str, Any]]]:
        return [
            (
                index,
                screen_one(
                    index,
                    source_root=source_root,
                    client=client,
                    deterministic_record=deterministic_records[index],
                ),
            )
            for index in group
        ]

    processed = 0
    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = {pool.submit(screen_environment, group): str(source.tasks[group[0]]["env_id"]) for group in groups}
        for future in as_completed(futures):
            for index, health_review in future.result():
                processed += 1
                previous_record = merged.get(index) or {}
                history = list(previous_record.get("health_review_history") or [])
                previous_review = previous_record.get("health_review")
                if isinstance(previous_review, Mapping):
                    history.append(dict(previous_review))
                updated_record = {
                    **deterministic_records[index],
                    "health_review": health_review,
                }
                if history:
                    updated_record["health_review_history"] = history
                merged[index] = updated_record
                _write_audit(audit_path, merged)
                print(
                    json.dumps(
                        {
                            "completed": len(completed) + processed,
                            "total": len(eligible),
                            "task_index": index,
                            "status": health_review["status"],
                            "reason": health_review["status_reason"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    incomplete_indices = [index for index in eligible if not _is_completed_review(merged[index])]
    if incomplete_indices:
        raise RuntimeError(f"EnvScaler static feasibility has pending infrastructure failures; resume to retry before building the healthy pool ({len(incomplete_indices)} tasks)")

    accepted_indices = [index for index in eligible if merged[index]["health_review"].get("accepted") is True]
    rows = [training_row(index, source.tasks[index]) for index in accepted_indices]
    pool_path = output_dir / "envscaler_training_pool.parquet"
    columns = list(training_row(eligible[0], source.tasks[eligible[0]]))
    pd.DataFrame(rows, columns=columns).to_parquet(pool_path, index=False)
    reason_counts: dict[str, int] = {}
    for index in eligible:
        reason = str(merged[index]["health_review"]["status_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    manifest = {
        "kind": "envscaler_healthy_task_pool",
        "protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "source": source.identity,
        "filter_logic": STATIC_FEASIBILITY_MEMBERSHIP_RULE,
        "expert_outcome_membership_gate": False,
        "funnel": {
            "source_tasks": len(source.tasks),
            "deterministic_pass": len(all_deterministic_eligible),
            "deterministic_quarantine": len(source.tasks) - len(all_deterministic_eligible),
            "judge_eligible": len(eligible),
            "judge_healthy": len(accepted_indices),
            "judge_quarantine": len(eligible) - len(accepted_indices),
            "final_healthy": len(accepted_indices),
            "selection_limited": limit is not None,
        },
        "counts": {
            "eligible": len(eligible),
            "healthy": len(accepted_indices),
            "quarantine": len(eligible) - len(accepted_indices),
            "reasons": dict(sorted(reason_counts.items())),
            "healthy_environments": len({str(source.tasks[index]["env_id"]) for index in accepted_indices}),
        },
        "accepted_task_indices": accepted_indices,
        "accepted_task_ids": [str(source.tasks[index]["task_id"]) for index in accepted_indices],
        # Recomputed from durable judge records so resume does not
        # under-report work completed by a previous process.
        "api_usage": _aggregate_api_usage(merged),
        "screening_concurrency": int(concurrency),
        "artifacts": {
            "config": {
                "path": "config.json",
                "sha256": sha256_file(config_path),
            },
            "task_audit": {
                "path": "task_audit.jsonl",
                "sha256": sha256_file(audit_path),
            },
            "training_pool": {
                "path": "envscaler_training_pool.parquet",
                "sha256": sha256_file(pool_path),
            },
        },
    }
    (output_dir / "health_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--deterministic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.timeout_seconds <= 0 or args.max_retries <= 0 or args.max_tokens <= 0:
        parser.error("--timeout-seconds, --max-retries, and --max-tokens must be positive")
    manifest = run_screening(
        source_root=args.source_root,
        deterministic_dir=args.deterministic_dir,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
