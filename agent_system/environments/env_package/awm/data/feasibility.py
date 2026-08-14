"""Static, code-augmented feasibility screening for deterministic-pass AWM tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from jsonschema import Draft202012Validator

from agent_system.environments.static_feasibility import (
    STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS,
    STATIC_FEASIBILITY_LABELS,
    STATIC_FEASIBILITY_MEMBERSHIP_RULE,
    STATIC_FEASIBILITY_PROTOCOL_VERSION,
    static_feasibility_generation_settings,
    static_feasibility_instruction,
    static_feasibility_review_is_complete,
)

from ..runtime.actions import openai_tools, tool_schema_audit
from ..runtime.logical_time import (
    LOGICAL_TIME_PROTOCOL_VERSION,
    freeze_temporal_source,
    freeze_temporal_structure,
    load_logical_time_policy,
    require_server_protocol,
)
from ..runtime.rollout import observation_dict, sha256_file
from .audit_utils import _load_candidate_rows, _normalize_scenario
from .deterministic_health import (
    MANIFEST_FILENAME as DETERMINISTIC_MANIFEST_FILENAME,
)
from .deterministic_health import TASK_FILENAME as DETERMINISTIC_TASK_FILENAME
from .deterministic_health import verify as verify_deterministic_audit
from .integrity import _load_jsonl
from .prepare import DATASET_NAME, DATASET_REVISION, EXPECTED_SOURCE_SHA256
from .selection import SELECTION_PROTOCOL_VERSION

HEALTH_POOL_PROTOCOL_VERSION = 3
HEALTH_POOL_KIND = "awm_healthy_task_pool"
HEALTH_POOL_FILENAME = "awm_training_pool.parquet"
HEALTH_MANIFEST_FILENAME = "health_manifest.json"
TASK_AUDIT_FILENAME = "task_audit.jsonl"
CONFIG_FILENAME = "config.json"
JUDGE_MAX_TOKENS = STATIC_FEASIBILITY_DEFAULT_MAX_TOKENS
VALID_LABELS = STATIC_FEASIBILITY_LABELS
_AUXILIARY_OUTPUT_FILENAMES = frozenset({"run.log", "server.log"})

JUDGE_INSTRUCTION = static_feasibility_instruction(
    environment_description="You audit one AgentWorldModel (AWM) task using static, code-augmented evidence.",
    evidence_notes=(
        "The supplied logical time, frozen environment source, frozen database schema "
        "and seed, native tool schemas, and SQL verifier use the same pinned runtime "
        "protocol as training. Treat them as authoritative. The no-action verifier "
        "execution is diagnostic only: an unchanged initial state is normally "
        "incomplete and is not itself evidence that a task is broken."
    ),
)


class DeepSeekFeasibilityClient:
    """Small, retrying DeepSeek JSON client with durable usage accounting."""

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
        if int(max_retries) <= 0 or int(max_tokens) <= 0:
            raise ValueError("judge max_retries and max_tokens must be positive")
        self.api_key = api_key
        self.model = str(model)
        self.url = str(api_base).rstrip("/") + "/chat/completions"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
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

    def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
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
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
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
    def _message(response: Mapping[str, Any]) -> Mapping[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek response has no choices")
        return choices[0].get("message") or {}

    @staticmethod
    def _parse_content(content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("static feasibility judge returned empty content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as direct_error:
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
            raise TypeError("static feasibility judge response must be an object")
        return value

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> dict[str, Any]:
        label = value.get("label")
        confidence = value.get("confidence")
        if label not in VALID_LABELS:
            raise ValueError(f"invalid static feasibility label: {label!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("static feasibility confidence must be integer 0..100")
        rationale = value.get("rationale")
        evidence = value.get("evidence")
        if not isinstance(rationale, str) or not isinstance(evidence, list):
            raise ValueError("static feasibility rationale/evidence malformed")
        return {
            "label": label,
            "confidence": confidence,
            "rationale": rationale,
            "evidence": [str(item) for item in evidence],
        }

    def judge(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_INSTRUCTION},
                {
                    "role": "user",
                    # Environment-static fields intentionally come first in
                    # build_static_evidence for provider prefix caching.
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
                error = RuntimeError(f"DeepSeek feasibility request failed: {exc}")
                error.usage = cumulative_usage  # type: ignore[attr-defined]
                error.structured_errors = structured_errors  # type: ignore[attr-defined]
                raise error from exc
            for key, raw_value in (response.get("usage") or {}).items():
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                cumulative_usage[key] = cumulative_usage.get(key, 0) + int(raw_value)
            try:
                content = str(self._message(response).get("content") or "")
                parsed = self._validate(self._parse_content(content))
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                structured_errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 < self.max_retries:
                    time.sleep(min(4.0, 2.0**attempt) + random.random() * 0.2)
                    continue
                error = RuntimeError(f"DeepSeek returned no valid structured feasibility response after {self.max_retries} attempts: {structured_errors[-1]}")
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
        raise AssertionError("positive max_retries must execute at least one attempt")


class StaticEvidenceStore:
    """Offset-backed access to pinned AWM evidence and frozen SQL execution."""

    def __init__(
        self,
        *,
        data_dir: Path,
        tools_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
    ):
        self.data_dir = Path(data_dir).resolve()
        self.tools_by_scenario = {_normalize_scenario(key): [dict(item) for item in value] for key, value in tools_by_scenario.items()}
        self._paths = {
            "environment": self.data_dir / "gen_envs.jsonl",
            "schema": self.data_dir / "gen_db.jsonl",
            "sample": self.data_dir / "gen_sample.jsonl",
            "verifier": self.data_dir / "gen_verifier.jsonl",
        }
        self._scenario_offsets = {name: self._offsets(path, with_task=False) for name, path in self._paths.items() if name != "verifier"}
        self._verifier_offsets = self._offsets(self._paths["verifier"], with_task=True)
        self._logical_time = load_logical_time_policy(self.data_dir)

    @staticmethod
    def _offsets(path: Path, *, with_task: bool) -> dict[Any, list[int]]:
        output: dict[Any, list[int]] = {}
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                scenario = _normalize_scenario(record["scenario"])
                key: Any = (scenario, int(record["task_idx"])) if with_task else scenario
                output.setdefault(key, []).append(offset)
        return output

    @staticmethod
    def _read(path: Path, offset: int) -> dict[str, Any]:
        with path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def _scenario_record(self, kind: str, scenario: str) -> dict[str, Any]:
        key = _normalize_scenario(scenario)
        try:
            offsets = self._scenario_offsets[kind][key]
        except KeyError as exc:
            raise RuntimeError(f"missing AWM {kind} evidence for {scenario!r}") from exc
        if len(offsets) != 1:
            raise RuntimeError(f"conflicting AWM {kind} evidence for {scenario!r}")
        return self._read(self._paths[kind], offsets[0])

    def _verifier_record(self, scenario: str, task_idx: int) -> dict[str, Any]:
        key = (_normalize_scenario(scenario), int(task_idx))
        try:
            offsets = self._verifier_offsets[key]
        except KeyError as exc:
            raise RuntimeError(f"missing AWM SQL verifier for {key!r}") from exc
        if len(offsets) != 1:
            raise RuntimeError(f"conflicting AWM SQL verifier for {key!r}")
        return self._read(self._paths["verifier"], offsets[0])

    @staticmethod
    def _parsed_verifier_response(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, Mapping):
            # The generator response duplicates the complete verifier code.
            # Retain its criteria/reasoning without paying for the same source
            # twice in every task-specific request.
            return {key: item for key, item in parsed.items() if key != "code"}
        return parsed

    @staticmethod
    def _no_action_verifier(
        *,
        schema: Mapping[str, Any],
        sample: Any,
        code: str,
    ) -> dict[str, Any]:
        from agent_world_model_env.server.db_manager import create_database
        from agent_world_model_env.server.verifier import execute_sql_verifier

        with tempfile.TemporaryDirectory(prefix="awm-static-feasibility-") as directory:
            initial_path = Path(directory) / "initial.db"
            final_path = Path(directory) / "final.db"
            create_database(str(initial_path), dict(schema), sample)
            shutil.copy2(initial_path, final_path)
            return execute_sql_verifier(
                code,
                "verify_task",
                str(initial_path),
                str(final_path),
            )

    def build_static_evidence(
        self,
        row: Mapping[str, Any],
        *,
        deterministic_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        scenario = str(row["scenario"])
        key = _normalize_scenario(scenario)
        logical_time = self._logical_time.for_scenario(key)
        environment_record = self._scenario_record("environment", scenario)
        schema_record = self._scenario_record("schema", scenario)
        sample_record = self._scenario_record("sample", scenario)
        verifier_record = self._verifier_record(scenario, int(row["task_idx"]))
        schema = freeze_temporal_structure(schema_record.get("db_schema") or {}, logical_time)
        sample = freeze_temporal_structure(sample_record.get("sample_data") or {}, logical_time)
        environment_code = freeze_temporal_source(str(environment_record.get("full_code") or ""), logical_time)
        verification = dict(verifier_record.get("verification") or {})
        verifier_code = freeze_temporal_source(str(verification.get("code") or ""), logical_time)
        tools = self.tools_by_scenario.get(key)
        if not tools:
            raise RuntimeError(f"missing runtime native tools for scenario {scenario!r}")
        no_action = self._no_action_verifier(
            schema=schema,
            sample=sample,
            code=verifier_code,
        )
        return {
            "runtime_protocol": {
                "dataset_revision": DATASET_REVISION,
                "logical_time_protocol_version": LOGICAL_TIME_PROTOCOL_VERSION,
                **self._logical_time.scenario_record(scenario),
                "temporal_freezing": ("environment source, SQL verifier, schema, and seed data are frozen with the same protocol used by the AWM runtime"),
            },
            # Static environment data comes before task-specific fields to
            # maximize DeepSeek prefix-cache reuse across the ten tasks.
            "environment": {
                "scenario": scenario,
                "full_code": environment_code,
                "native_tools": openai_tools(tools),
                "database_schema": schema,
                "initial_database_seed": sample,
            },
            "task": {
                "task_id": str(row["task_id"]),
                "task_idx": int(row["task_idx"]),
                "request": str(row["task"]),
            },
            "sql_verifier": {
                "code": verifier_code,
                "generator_response": self._parsed_verifier_response(verification.get("raw_response")),
                "no_action_execution": no_action,
            },
            "deterministic_audit": dict(deterministic_record),
        }


async def _fetch_tools_for_scenario(
    row: Mapping[str, Any],
    *,
    awm_base_url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict[str, Any]]]:
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    async with semaphore:
        async with AWMEnv(base_url=awm_base_url) as env:
            reset = await env.reset(
                scenario=str(row["scenario"]),
                task_idx=int(row["task_idx"]),
                seed=0,
            )
            payload = observation_dict(reset)
            if payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"AWM reset failed while collecting tools: {payload}")
            if str(payload.get("task") or "") != str(row["task"]):
                raise RuntimeError("AWM runtime task identity mismatch while collecting tools")
            raw_tools = await env.list_tools(use_cache=False)
            audit = tool_schema_audit(raw_tools)
            for tool in audit["canonical_tools"]:
                Draft202012Validator.check_schema(tool["inputSchema"])
            expected_hash = str(row.get("tool_schema_hash") or "")
            if expected_hash and audit["canonical_tool_schema_hash"] != expected_hash:
                raise RuntimeError("AWM runtime canonical tool schema mismatch")
            try:
                await env.step(CallToolAction(tool_name="done", arguments={}))
            except Exception:
                pass
    return _normalize_scenario(row["scenario"]), audit["canonical_tools"]


async def fetch_tools_by_scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    awm_base_url: str,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    first_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        first_rows.setdefault(_normalize_scenario(row["scenario"]), row)
    semaphore = asyncio.Semaphore(int(concurrency))
    results = await asyncio.gather(
        *(
            _fetch_tools_for_scenario(
                row,
                awm_base_url=awm_base_url,
                semaphore=semaphore,
            )
            for row in first_rows.values()
        )
    )
    return dict(results)


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode()
    environment = evidence["environment"]
    task = evidence["task"]
    no_action = evidence["sql_verifier"]["no_action_execution"]
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "task_id": str(task["task_id"]),
        "scenario": str(environment["scenario"]),
        "tool_names": [str(item["function"]["name"]) for item in environment["native_tools"]],
        "no_action_execution_status": str(no_action.get("execution_status") or ""),
    }


def screen_one(
    row: Mapping[str, Any],
    *,
    store: StaticEvidenceStore,
    client: DeepSeekFeasibilityClient,
    deterministic_record: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    try:
        evidence = store.build_static_evidence(
            row,
            deterministic_record=deterministic_record,
        )
        evidence_summary = _evidence_summary(evidence)
    except Exception as exc:
        return {
            "task_id": task_id,
            "scenario": str(row["scenario"]),
            "task_idx": int(row["task_idx"]),
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
            "task_id": task_id,
            "scenario": str(row["scenario"]),
            "task_idx": int(row["task_idx"]),
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
        "task_id": task_id,
        "scenario": str(row["scenario"]),
        "task_idx": int(row["task_idx"]),
        "static_feasibility_protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "accepted": accepted,
        "status": "healthy" if accepted else "quarantine",
        "status_reason": "healthy" if accepted else str(judge["label"]),
        "screening_error": None,
        "evidence_summary": evidence_summary,
        "judge": judge,
    }


def _write_audit(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    order: Sequence[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(records[task_id], ensure_ascii=False, sort_keys=True) + "\n" for task_id in order if task_id in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _aggregate_api_usage(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    totals = {
        "successful_responses": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in records.values():
        reviews = [
            *(record.get("screening_history") or []),
            record,
        ]
        for review in reviews:
            judge = review.get("judge") or {}
            usage = judge.get("usage") or review.get("judge_attempt_usage") or {}
            if not usage:
                continue
            if judge:
                responses = int(judge.get("structured_response_attempts", 1) or 1)
            else:
                responses = len(review.get("judge_attempt_errors") or []) or 1
            totals["successful_responses"] += responses
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[field] += int(usage.get(field, 0) or 0)
    return totals


def _validate_review(
    record: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    if str(record.get("task_id") or "") != str(row["task_id"]):
        raise RuntimeError("AWM static feasibility task identity mismatch")
    if str(record.get("scenario") or "") != str(row["scenario"]):
        raise RuntimeError("AWM static feasibility scenario identity mismatch")
    if int(record.get("task_idx", -1)) != int(row["task_idx"]):
        raise RuntimeError("AWM static feasibility task index mismatch")
    accepted = record.get("accepted")
    status = record.get("status")
    reason = str(record.get("status_reason") or "")
    if accepted is True:
        if status != "healthy" or reason != "healthy":
            raise RuntimeError("AWM static feasibility healthy verdict mismatch")
    elif accepted is False:
        if status != "quarantine" or reason == "healthy":
            raise RuntimeError("AWM static feasibility quarantine verdict mismatch")
    elif accepted is None:
        if status != "pending" or reason != "judge_infrastructure_exhausted":
            raise RuntimeError("AWM static feasibility pending verdict mismatch")
    else:
        raise RuntimeError("AWM static feasibility accepted flag is invalid")
    judge = record.get("judge")
    if isinstance(judge, Mapping):
        DeepSeekFeasibilityClient._validate(judge)
        label = str(judge.get("label") or "")
        expected_reason = "healthy" if label == "healthy" else label
        if label not in VALID_LABELS or reason != expected_reason:
            raise RuntimeError("AWM static feasibility judge label mismatch")
    elif reason not in {
        "static_evidence_failure",
        "judge_infrastructure_exhausted",
    }:
        raise RuntimeError("AWM static feasibility missing a required judge verdict")


def _is_completed_review(record: Mapping[str, Any]) -> bool:
    """Preserve only reviews produced by the sole current protocol."""
    return static_feasibility_review_is_complete(record)


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row["training_row"])
    extra = dict(output["extra_info"])
    extra.update(
        {
            "awm_health_pool_protocol_version": HEALTH_POOL_PROTOCOL_VERSION,
            "awm_static_feasibility_protocol_version": (STATIC_FEASIBILITY_PROTOCOL_VERSION),
            "awm_static_feasibility_status": "healthy",
            "awm_training_pool_status": "active",
        }
    )
    output["extra_info"] = extra
    return output


def verify_healthy_pool(data: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != HEALTH_POOL_PROTOCOL_VERSION:
        raise RuntimeError("AWM static-feasibility pool protocol mismatch")
    if manifest.get("kind") != HEALTH_POOL_KIND:
        raise RuntimeError("AWM training manifest is not a healthy task pool")
    if manifest.get("static_feasibility_protocol_version") != STATIC_FEASIBILITY_PROTOCOL_VERSION:
        raise RuntimeError("AWM static-feasibility protocol mismatch")
    if data.name != HEALTH_POOL_FILENAME:
        raise RuntimeError(f"AWM healthy pool data must be named {HEALTH_POOL_FILENAME}")
    root = manifest_path.parent
    paths = {
        "config_sha256": root / CONFIG_FILENAME,
        "task_audit_sha256": root / TASK_AUDIT_FILENAME,
        "training_pool_data_sha256": data,
    }
    for field, path in paths.items():
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM static-feasibility artifact hash mismatch: {path}")
    relative_dir = manifest.get("deterministic_audit_relative_dir")
    if not isinstance(relative_dir, str) or not relative_dir:
        raise RuntimeError("AWM static-feasibility deterministic audit is missing")
    deterministic_dir = (root / relative_dir).resolve()
    verify_deterministic_audit(deterministic_dir)
    deterministic_manifest = deterministic_dir / DETERMINISTIC_MANIFEST_FILENAME
    if sha256_file(deterministic_manifest) != manifest.get("deterministic_manifest_sha256"):
        raise RuntimeError("AWM deterministic manifest hash mismatch")
    deterministic_records = _load_jsonl(deterministic_dir / DETERMINISTIC_TASK_FILENAME)
    if sha256_file(deterministic_dir / DETERMINISTIC_TASK_FILENAME) != manifest.get("deterministic_audit_sha256"):
        raise RuntimeError("AWM deterministic task-audit hash mismatch")
    candidate_ids = [str(value) for value in manifest.get("candidate_task_ids") or []]
    deterministic_ids = [str(record["task_id"]) for record in deterministic_records]
    if candidate_ids != deterministic_ids:
        raise RuntimeError("AWM candidates differ from the deterministic audit")
    deterministic_pass = [record for record in deterministic_records if record.get("status") == "healthy"]
    eligible_ids = [str(value) for value in manifest.get("judge_eligible_task_ids") or []]
    deterministic_pass_ids = [str(record["task_id"]) for record in deterministic_pass]
    if eligible_ids != deterministic_pass_ids[: len(eligible_ids)]:
        raise RuntimeError("AWM static-feasibility eligible tasks are not a deterministic-pass prefix")
    rows_by_id = {
        str(record["task_id"]): {
            "task_id": str(record["task_id"]),
            "scenario": str(record["scenario"]),
            "task_idx": int(record["task_idx"]),
        }
        for record in deterministic_pass
    }
    reviews = _load_jsonl(root / TASK_AUDIT_FILENAME)
    if [str(record["task_id"]) for record in reviews] != eligible_ids:
        raise RuntimeError("AWM static-feasibility audit order mismatch")
    for record in reviews:
        _validate_review(record, row=rows_by_id[str(record["task_id"])])
        if not _is_completed_review(record):
            raise RuntimeError("AWM static-feasibility pool contains an incomplete or stale review")
    healthy_ids = [str(record["task_id"]) for record in reviews if record.get("accepted") is True]
    if manifest.get("training_pool_task_ids") != healthy_ids:
        raise RuntimeError("AWM static-feasibility pool task IDs differ from manifest")
    frame = pd.read_parquet(data)
    extras = [dict(value) for value in frame["extra_info"].tolist()]
    if [str(value["task_id"]) for value in extras] != healthy_ids:
        raise RuntimeError("AWM static-feasibility Parquet task IDs differ")
    if any(value.get("awm_health_pool_protocol_version") != HEALTH_POOL_PROTOCOL_VERSION or value.get("awm_static_feasibility_protocol_version") != STATIC_FEASIBILITY_PROTOCOL_VERSION or value.get("awm_static_feasibility_status") != "healthy" for value in extras):
        raise RuntimeError("AWM static-feasibility row metadata mismatch")
    if manifest.get("expert_outcome_membership_gate") is not False:
        raise RuntimeError("AWM expert outcomes must not gate static feasibility")
    expected_counts = {
        "context_eligible": len(deterministic_records),
        "deterministic_pass": len(deterministic_pass),
        "deterministic_quarantine": len(deterministic_records) - len(deterministic_pass),
        "judge_eligible": len(eligible_ids),
        "healthy": len(healthy_ids),
        "quarantine": len(eligible_ids) - len(healthy_ids),
        "healthy_environments": len({str(record["scenario"]) for record in reviews if record.get("accepted") is True}),
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("AWM static-feasibility manifest counts mismatch")
    reason_counts: dict[str, int] = {}
    for record in reviews:
        reason = str(record["status_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if manifest.get("reason_counts") != dict(sorted(reason_counts.items())):
        raise RuntimeError("AWM static-feasibility reason counts mismatch")
    return {"tasks": len(healthy_ids), "data": str(data), "kind": HEALTH_POOL_KIND}


def run_screening(args: argparse.Namespace) -> dict[str, Any]:
    rows, selection = _load_candidate_rows(args.data, args.candidate_manifest)
    source_hashes = {name: sha256_file(args.awm_data_dir / name) for name in EXPECTED_SOURCE_SHA256}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("AWM static feasibility requires the pinned source hashes")
    deterministic_manifest_path = args.deterministic_dir / DETERMINISTIC_MANIFEST_FILENAME
    deterministic_audit_path = args.deterministic_dir / DETERMINISTIC_TASK_FILENAME
    verify_deterministic_audit(args.deterministic_dir)
    deterministic_records = _load_jsonl(deterministic_audit_path)
    if [str(record["task_id"]) for record in deterministic_records] != [str(row["task_id"]) for row in rows]:
        raise RuntimeError("AWM deterministic audit and context candidates differ")
    deterministic_by_id = {str(record["task_id"]): record for record in deterministic_records}
    eligible_rows = [row for row in rows if deterministic_by_id[str(row["task_id"])].get("status") == "healthy"]
    all_eligible_count = len(eligible_rows)
    if args.limit is not None:
        eligible_rows = eligible_rows[: int(args.limit)]
    if not eligible_rows:
        raise RuntimeError("AWM static feasibility has no deterministic-pass tasks")
    eligible_ids = [str(row["task_id"]) for row in eligible_rows]
    require_server_protocol(args.awm_base_url, args.awm_data_dir)
    client = DeepSeekFeasibilityClient(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )
    identity = {
        "kind": "awm_static_feasibility_screening",
        "protocol_version": HEALTH_POOL_PROTOCOL_VERSION,
        "static_feasibility_protocol_version": STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": source_hashes,
        "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "selection_counts": selection["selected_counts"],
        "candidate_task_ids": [str(row["task_id"]) for row in rows],
        "deterministic_audit_relative_dir": os.path.relpath(args.deterministic_dir.resolve(), args.output_dir.resolve()),
        "deterministic_manifest_sha256": sha256_file(deterministic_manifest_path),
        "deterministic_audit_sha256": sha256_file(deterministic_audit_path),
        "model": args.model,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        **static_feasibility_generation_settings(max_tokens=args.max_tokens),
        "timeout_seconds": float(args.timeout_seconds),
        "max_retries": int(args.max_retries),
        "membership_rule": STATIC_FEASIBILITY_MEMBERSHIP_RULE,
        "expert_outcome_membership_gate": False,
        "prior_code_augmented_screening_dependency": False,
        "judge_eligible_task_ids": eligible_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / CONFIG_FILENAME
    audit_path = args.output_dir / TASK_AUDIT_FILENAME
    existing: dict[str, dict[str, Any]] = {}
    if args.resume:
        if not config_path.is_file():
            raise RuntimeError("--resume requires an existing static feasibility config")
        if json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("AWM static feasibility resume configuration mismatch")
        existing = {str(record["task_id"]): record for record in _load_jsonl(audit_path, repair_torn_tail=True)}
    else:
        unexpected = {path.name for path in args.output_dir.iterdir() if path.name not in _AUXILIARY_OUTPUT_FILENAMES}
        if unexpected:
            raise FileExistsError(f"refusing to overwrite AWM static feasibility output: {sorted(unexpected)}")
        config_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if set(existing) - set(eligible_ids):
        raise RuntimeError("AWM static feasibility resume has unexpected task IDs")
    rows_by_id = {str(row["task_id"]): row for row in eligible_rows}
    for task_id, record in existing.items():
        _validate_review(record, row=rows_by_id[task_id])
    completed_ids = {task_id for task_id, record in existing.items() if _is_completed_review(record)}
    refreshed_existing = set(existing) - completed_ids
    pending_rows = [row for row in eligible_rows if str(row["task_id"]) not in completed_ids]
    print(
        json.dumps(
            {
                "resume_reused": len(completed_ids),
                "resume_refresh_existing": len(refreshed_existing),
                "resume_unseen": len(pending_rows) - len(refreshed_existing),
                "pending": len(pending_rows),
                "total": len(eligible_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if pending_rows:
        tools = asyncio.run(
            fetch_tools_by_scenario(
                pending_rows,
                awm_base_url=args.awm_base_url,
                concurrency=args.runtime_concurrency,
            )
        )
        store = StaticEvidenceStore(
            data_dir=args.awm_data_dir,
            tools_by_scenario=tools,
        )
        # Run environments in parallel but tasks from one environment in
        # sequence. This lets the first task establish DeepSeek's large shared
        # environment prefix before the remaining task-specific requests.
        scenario_rows: dict[str, list[Mapping[str, Any]]] = {}
        for row in pending_rows:
            scenario_rows.setdefault(_normalize_scenario(row["scenario"]), []).append(row)
        for group in scenario_rows.values():
            group.sort(key=lambda row: int(row["task_idx"]))

        def screen_scenario(group: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            return [
                screen_one(
                    row,
                    store=store,
                    client=client,
                    deterministic_record=deterministic_by_id[str(row["task_id"])],
                )
                for row in group
            ]

        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as pool:
            futures = {pool.submit(screen_scenario, group): scenario for scenario, group in sorted(scenario_rows.items())}
            completed = len(eligible_rows) - len(pending_rows)
            for future in as_completed(futures):
                for record in future.result():
                    task_id = str(record["task_id"])
                    previous = existing.get(task_id)
                    history = list((previous or {}).get("screening_history") or [])
                    if previous is not None:
                        prior = {key: value for key, value in previous.items() if key != "screening_history"}
                        history.append(prior)
                    if history:
                        record = dict(record)
                        record["screening_history"] = history
                    existing[task_id] = record
                    completed += 1
                    _write_audit(audit_path, existing, eligible_ids)
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(eligible_rows),
                                "task_id": task_id,
                                "status": record["status"],
                                "reason": record["status_reason"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    if set(existing) != set(eligible_ids):
        raise RuntimeError("AWM static feasibility audit is incomplete")
    incomplete_ids = [task_id for task_id in eligible_ids if not _is_completed_review(existing[task_id])]
    if incomplete_ids:
        raise RuntimeError(f"AWM static feasibility has pending infrastructure failures; resume to retry before building the healthy pool ({len(incomplete_ids)} tasks)")
    ordered_reviews = [existing[task_id] for task_id in eligible_ids]
    _write_audit(audit_path, existing, eligible_ids)
    healthy_ids = [str(record["task_id"]) for record in ordered_reviews if record.get("accepted") is True]
    healthy_set = set(healthy_ids)
    pool_path = args.output_dir / HEALTH_POOL_FILENAME
    training_rows = [_training_row(row) for row in eligible_rows if row["task_id"] in healthy_set]
    pd.DataFrame(training_rows, columns=list(_training_row(eligible_rows[0]))).to_parquet(pool_path, index=False)
    counts = {
        "context_eligible": len(rows),
        "deterministic_pass": all_eligible_count,
        "deterministic_quarantine": len(rows) - all_eligible_count,
        "judge_eligible": len(eligible_rows),
        "healthy": len(healthy_ids),
        "quarantine": len(eligible_rows) - len(healthy_ids),
        "healthy_environments": len({str(record["scenario"]) for record in ordered_reviews if record.get("accepted") is True}),
    }
    reasons: dict[str, int] = {}
    for record in ordered_reviews:
        reason = str(record["status_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    manifest = {
        **identity,
        "kind": HEALTH_POOL_KIND,
        "filter_logic": identity["membership_rule"],
        "counts": counts,
        "reason_counts": dict(sorted(reasons.items())),
        "selection_limited": args.limit is not None,
        "training_pool_filename": HEALTH_POOL_FILENAME,
        "training_pool_task_ids": healthy_ids,
        "task_audit_sha256": sha256_file(audit_path),
        "config_sha256": sha256_file(config_path),
        "training_pool_data_sha256": sha256_file(pool_path),
        "api_usage": _aggregate_api_usage(existing),
        "screening_concurrency": int(args.concurrency),
    }
    manifest_path = args.output_dir / HEALTH_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_healthy_pool(pool_path, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--awm-data-dir", type=Path, required=True)
    parser.add_argument("--deterministic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--runtime-concurrency", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        summary = verify_healthy_pool(
            args.output_dir / HEALTH_POOL_FILENAME,
            args.output_dir / HEALTH_MANIFEST_FILENAME,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    for name in ("concurrency", "runtime_concurrency", "max_retries", "max_tokens"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    manifest = run_screening(args)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
