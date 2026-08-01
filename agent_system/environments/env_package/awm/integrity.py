"""Integrity audit and conservative quarantine for AWM training tasks."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .actions import tool_schema_audit
from .data import DATASET_NAME, DATASET_REVISION, EXPECTED_SOURCE_SHA256
from .native_rollout import observation_dict, sha256_file
from .selection import SELECTION_PROTOCOL_VERSION, stable_rank

INTEGRITY_PROTOCOL_VERSION = 3
JUDGE_PROTOCOL_VERSION = 3
CURRENT_VERIFIER_PROTOCOL = "sql"
KNOWN_CALIBRATION_TASKS = (
    "application_registration_management_1:1",
    "tournament_management_1:4",
    "enterprise_software_1:6",
    "forms_and_survey_platform_1:5",
)
_VERIFIER_FILES = {
    "code": "gen_verifier.pure_code.jsonl",
    "sql": "gen_verifier.jsonl",
}
_REQUIRED_ENTRYPOINTS = {
    "code": "verify_task_completion",
    "sql": "verify_task",
}
_INFRA_REWARD_TYPES = frozenset({"judge_error", "server_error", "no_verifier"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}") from exc
    return records


def _normalize_scenario(value: str) -> str:
    result = re.sub(r"[^a-z0-9_]", "_", str(value).lower())
    return re.sub(r"_+", "_", result).strip("_")


def _load_multimap(path: Path, key) -> dict[Any, list[dict[str, Any]]]:
    output: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in _load_jsonl(path):
        output[key(record)].append(record)
    return dict(output)


def _unique_record(
    records: Sequence[Mapping[str, Any]],
    *,
    missing_code: str,
    conflict_code: str,
    findings: list[dict[str, Any]],
    identity=None,
) -> Mapping[str, Any] | None:
    if not records:
        findings.append({"severity": "quarantine", "code": missing_code})
        return None
    fingerprints = {_sha256_json(identity(record) if identity is not None else record) for record in records}
    if len(fingerprints) > 1:
        findings.append(
            {
                "severity": "quarantine",
                "code": conflict_code,
                "records": len(records),
                "record_hashes": sorted(fingerprints),
            }
        )
        return None
    if len(records) > 1:
        findings.append(
            {
                "severity": "info",
                "code": "identical_duplicate_source_records",
                "records": len(records),
            }
        )
    return records[0]


def _verifier_source_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    verification = record.get("verification")
    code = verification.get("code") if isinstance(verification, Mapping) else None
    return {
        "scenario": _normalize_scenario(str(record.get("scenario") or "")),
        "task_idx": record.get("task_idx"),
        "task": record.get("task"),
        "code": code,
    }


def _verifier_check(
    entry: Mapping[str, Any] | None,
    *,
    mode: str,
    expected_task: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if entry is None:
        return None
    if str(entry.get("task") or "") != expected_task:
        findings.append({"severity": "quarantine", "code": f"{mode}_verifier_task_mismatch"})
    verification = entry.get("verification")
    code = verification.get("code") if isinstance(verification, Mapping) else None
    if not isinstance(code, str) or not code.strip():
        findings.append({"severity": "quarantine", "code": f"{mode}_verifier_missing_code"})
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code, filename=f"awm_{mode}_verifier.py")
            compile(tree, f"awm_{mode}_verifier.py", "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        findings.append(
            {
                "severity": "quarantine",
                "code": f"{mode}_verifier_compile_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {"code": code, "sha256": hashlib.sha256(code.encode()).hexdigest()}
    functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    expected = _REQUIRED_ENTRYPOINTS[mode]
    if expected not in functions:
        findings.append(
            {
                "severity": "quarantine",
                "code": f"{mode}_verifier_missing_entrypoint",
                "expected": expected,
            }
        )
    return {
        "code": code,
        "sha256": hashlib.sha256(code.encode()).hexdigest(),
        "entrypoint": expected,
    }


def _semantic_warning_codes(task: str, sample_data: Any) -> list[str]:
    task_lower = task.casefold()
    mutation_verbs = ("create", "add", "upload", "generate", "register", "schedule")
    verb_positions = [task_lower.find(verb) for verb in mutation_verbs if task_lower.find(verb) >= 0]
    if not verb_positions:
        return []
    mutation_clause = task_lower[min(verb_positions) :]
    quoted = {value.strip().casefold() for value in re.findall(r"['\"]([^'\"]{3,})['\"]", task) if value.strip()}
    tables = sample_data.get("tables") if isinstance(sample_data, Mapping) else []
    for table in tables or []:
        table_name = str(table.get("table_name") or "").casefold()
        table_phrase = table_name.replace("_", " ")
        if table_name not in mutation_clause and table_phrase not in mutation_clause:
            continue
        initial_rows = " ".join(str(item) for item in table.get("insert_statements") or []).casefold()
        if any(value in initial_rows for value in quoted):
            return ["requested_literal_present_in_initial_target_table"]
    return []


def refine_semantic_warnings(
    record: Mapping[str, Any],
    *,
    sample_data: Any | None = None,
) -> dict[str, Any]:
    """Use actual tool capabilities instead of treating every mutation verb as suspicious."""
    output = dict(record)
    runtime = dict(output.get("runtime") or {})
    errors = [str(error) for error in runtime.get("errors") or []]
    if runtime.get("status") == "infrastructure_exhausted" and errors and all(error.startswith("SchemaError:") for error in errors):
        runtime["status"] = "deterministic_failure"
        runtime["reason"] = "invalid_canonical_tool_schema"
        output["runtime"] = runtime
    obsolete = {
        "mutation_reachability_requires_review",
        "no_obvious_mutation_capability",
        "requested_literal_present_in_initial_state",
    }
    findings = [dict(item) for item in output.get("findings") or [] if not (item.get("severity") == "warning" and item.get("code") in obsolete)]
    warning_codes = [str(code) for code in output.get("semantic_warning_codes") or [] if code not in obsolete]
    if sample_data is not None:
        for code in _semantic_warning_codes(str(output.get("task") or ""), sample_data):
            warning_codes.append(code)
            if not any(item.get("severity") == "warning" and item.get("code") == code for item in findings):
                findings.append({"severity": "warning", "code": code})
    task = str(output.get("task") or "").casefold()
    mutation_verbs = (
        "create",
        "update",
        "change",
        "replace",
        "remove",
        "delete",
        "add",
        "upload",
        "generate",
        "send",
        "trigger",
        "submit",
        "schedule",
        "cancel",
        "complete",
        "register",
        "purchase",
        "order",
        "return",
    )
    mutation_tool_tokens = (
        "create",
        "update",
        "set",
        "change",
        "edit",
        "replace",
        "remove",
        "delete",
        "add",
        "upload",
        "generate",
        "send",
        "trigger",
        "submit",
        "schedule",
        "cancel",
        "complete",
        "register",
        "purchase",
        "order",
        "return",
        "assign",
        "link",
        "post",
        "approve",
        "reject",
        "initiate",
    )
    capabilities = (output.get("runtime") or {}).get("tool_capabilities") or []
    tool_names = [str(tool.get("name") or "").casefold() for tool in capabilities]
    if any(verb in task for verb in mutation_verbs) and tool_names and not any(token in name for token in mutation_tool_tokens for name in tool_names):
        warning_codes.append("no_obvious_mutation_capability")
        findings.append({"severity": "warning", "code": "no_obvious_mutation_capability"})
    output["findings"] = findings
    output["semantic_warning_codes"] = sorted(set(warning_codes))
    return output


def static_source_audit(
    row: Mapping[str, Any],
    *,
    task_records: Sequence[Mapping[str, Any]],
    code_records: Sequence[Mapping[str, Any]],
    sql_records: Sequence[Mapping[str, Any]],
    sample_data: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit immutable upstream records without executing model or environment code."""
    findings: list[dict[str, Any]] = []
    task_entry = _unique_record(
        task_records,
        missing_code="missing_task_source",
        conflict_code="conflicting_task_source_records",
        findings=findings,
    )
    expected_task = str(row["task"])
    if task_entry is not None:
        tasks = list(task_entry.get("tasks") or [])
        task_idx = int(row["task_idx"])
        if not 0 <= task_idx < len(tasks):
            findings.append({"severity": "quarantine", "code": "task_index_out_of_range"})
        elif str(tasks[task_idx]) != expected_task:
            findings.append({"severity": "quarantine", "code": "candidate_task_text_mismatch"})

    code_entry = _unique_record(
        code_records,
        missing_code="missing_code_verifier",
        conflict_code="conflicting_code_verifiers",
        findings=findings,
        identity=_verifier_source_identity,
    )
    sql_entry = _unique_record(
        sql_records,
        missing_code="missing_sql_verifier",
        conflict_code="conflicting_sql_verifiers",
        findings=findings,
        identity=_verifier_source_identity,
    )
    verifier_sources = {
        "code": _verifier_check(code_entry, mode="code", expected_task=expected_task, findings=findings),
        "sql": _verifier_check(sql_entry, mode="sql", expected_task=expected_task, findings=findings),
    }
    warning_codes = _semantic_warning_codes(expected_task, sample_data)
    findings.extend({"severity": "warning", "code": code} for code in warning_codes)
    return (
        {
            "findings": findings,
            "semantic_warning_codes": warning_codes,
            "verifier_sha256": {mode: source["sha256"] if source is not None else None for mode, source in verifier_sources.items()},
        },
        verifier_sources,
    )


def _compact_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for tool in tools:
        schema = dict(tool.get("inputSchema") or {})
        properties = dict(schema.get("properties") or {})
        output.append(
            {
                "name": tool.get("name"),
                "description": str(tool.get("description") or "")[:1000],
                "required": list(schema.get("required") or []),
                "parameters": {str(name): {key: value for key, value in dict(spec).items() if key in {"type", "anyOf", "oneOf", "enum", "description", "format"}} for name, spec in properties.items() if isinstance(spec, Mapping)},
            }
        )
    return output


async def _runtime_audit_once(
    row: Mapping[str, Any],
    *,
    awm_base_url: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    async with semaphore:
        async with AWMEnv(base_url=awm_base_url) as env:
            reset = await env.reset(
                scenario=str(row["scenario"]),
                task_idx=int(row["task_idx"]),
                seed=0,
            )
            reset_payload = observation_dict(reset)
            if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"reset failed: {reset_payload}")
            runtime_task = str(reset_payload.get("task") or "")
            raw_tools = await env.list_tools(use_cache=False)
            schema = tool_schema_audit(raw_tools)
            for tool in schema["canonical_tools"]:
                Draft202012Validator.check_schema(tool["inputSchema"])
            verify = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": "code", "final_answer": None},
                )
            )
            verify_payload = observation_dict(verify)
            await env.step(CallToolAction(tool_name="done", arguments={}))

    expected_canonical = str(row.get("tool_schema_hash") or "")
    expected_raw = str(row.get("raw_tool_schema_hash") or "")
    canonical_hash = str(schema["canonical_tool_schema_hash"])
    raw_hash = str(schema["raw_tool_schema_hash"])
    repaired_names = {repair["tool_name"] for repair in schema["schema_repairs"]}
    raw_by_name = {tool["name"]: tool["inputSchema"] for tool in schema["raw_tools"]}
    canonical_by_name = {tool["name"]: tool["inputSchema"] for tool in schema["canonical_tools"]}
    return {
        "runtime_task_matches": runtime_task == str(row["task"]),
        "runtime_task": runtime_task,
        "reset_reward_type": reset_payload.get("reward_type"),
        "raw_tool_schema_hash": raw_hash,
        "canonical_tool_schema_hash": canonical_hash,
        "raw_tool_schema_matches": not expected_raw or raw_hash == expected_raw,
        "canonical_tool_schema_matches": not expected_canonical or canonical_hash == expected_canonical,
        "schema_repairs": schema["schema_repairs"],
        "repaired_schema_pairs": {name: {"raw": raw_by_name[name], "canonical": canonical_by_name[name]} for name in sorted(repaired_names)},
        "tool_capabilities": _compact_tools(schema["canonical_tools"]),
        "no_op_reward_type": verify_payload.get("reward_type"),
        "no_op_verify_result": str(verify_payload.get("verify_result") or "")[:4000],
    }


async def runtime_audit(
    row: Mapping[str, Any],
    *,
    awm_base_url: str,
    semaphore: asyncio.Semaphore,
    attempts: int = 3,
) -> dict[str, Any]:
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            result = await _runtime_audit_once(row, awm_base_url=awm_base_url, semaphore=semaphore)
            reward_type = str(result.get("no_op_reward_type") or "")
            if reward_type in _INFRA_REWARD_TYPES:
                errors.append(f"no-op verifier infrastructure reward_type={reward_type}")
                continue
            return {"status": "ok", "attempt": attempt, **result}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "status": "infrastructure_exhausted",
        "attempt": attempts,
        "errors": errors,
    }


def classify_static_record(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons = [str(item["code"]) for item in record.get("findings") or [] if item.get("severity") == "quarantine"]
    runtime = record.get("runtime") or {}
    if runtime.get("status") == "infrastructure_exhausted":
        return "infrastructure_pending", ["runtime_infrastructure_exhausted"]
    if runtime.get("status") == "deterministic_failure":
        return "quarantine", [str(runtime.get("reason") or "runtime_deterministic_failure")]
    if not runtime.get("runtime_task_matches", False):
        reasons.append("runtime_task_mismatch")
    if not runtime.get("raw_tool_schema_matches", False):
        reasons.append("raw_tool_schema_mismatch")
    if not runtime.get("canonical_tool_schema_matches", False):
        reasons.append("canonical_tool_schema_mismatch")
    if runtime.get("no_op_reward_type") == "complete":
        reasons.append("no_op_code_verifier_complete")
    if reasons:
        return "quarantine", sorted(set(reasons))
    return "pass", []


def refresh_cached_static_record(
    record: Mapping[str, Any],
    *,
    sample_data: Any,
) -> dict[str, Any]:
    output = refine_semantic_warnings(record, sample_data=sample_data)
    status, reasons = classify_static_record(output)
    output["status"] = status
    output["status_reasons"] = reasons
    return output


def _truncate(tokenizer, value: Any, max_tokens: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def _relevant_initial_state(task: str, sample_data: Any, limit: int = 48) -> list[str]:
    keywords = {word for word in re.findall(r"[a-z0-9_]+", task.casefold()) if len(word) >= 4 and word not in {"with", "from", "that", "this", "each", "using", "current"}}
    statements = []
    tables = sample_data.get("tables") if isinstance(sample_data, Mapping) else []
    for table in tables or []:
        table_name = str(table.get("table_name") or "")
        for statement in table.get("insert_statements") or []:
            lowered = str(statement).casefold()
            score = sum(keyword in lowered for keyword in keywords)
            statements.append((score, table_name, str(statement)))
    statements.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [f"[{table}] {statement}" for _, table, statement in statements[:limit]]


def build_judge_messages(
    row: Mapping[str, Any],
    static_record: Mapping[str, Any],
    verifier_sources: Mapping[str, Mapping[str, Any] | None],
    *,
    sample_data: Any,
    db_schema: Any,
    tokenizer,
    judge_index: int,
) -> list[dict[str, str]]:
    runtime = static_record.get("runtime") or {}
    bundle = {
        "task_id": row["task_id"],
        "task": row["task"],
        "semantic_warnings": static_record.get("semantic_warning_codes") or [],
        "initial_state_relevant_rows": _truncate(
            tokenizer,
            _relevant_initial_state(str(row["task"]), sample_data),
            3500,
        ),
        "database_schema": _truncate(tokenizer, db_schema, 2500),
        "available_tool_capabilities": _truncate(
            tokenizer,
            runtime.get("tool_capabilities") or [],
            3500,
        ),
        "raw_to_canonical_schema_repairs": _truncate(
            tokenizer,
            {
                "repairs": runtime.get("schema_repairs") or [],
                "schema_pairs": runtime.get("repaired_schema_pairs") or {},
            },
            1500,
        ),
        "code_verifier": _truncate(
            tokenizer,
            (verifier_sources.get("code") or {}).get("code", ""),
            4000,
        ),
        "sql_code_augmented_verifier": _truncate(
            tokenizer,
            (verifier_sources.get("sql") or {}).get("code", ""),
            4000,
        ),
        "independent_review_index": int(judge_index),
    }
    system = (
        "You are auditing an AgentWorldModel task for dataset integrity. "
        "Inspect the task, initial database evidence, actual tool capabilities, "
        "raw-to-canonical schema repairs, and BOTH executable verifier programs. "
        "Decide whether a competent policy can reach a state accepted by the current "
        "SQL/code-augmented verifier from this initial state using only the available tools. "
        "Do not score model behavior. Distinguish an impossible or contradictory task from "
        "a merely difficult task. A repaired nullable interface schema is not itself an "
        "infeasible task. Return one JSON object only with: verdict "
        "(feasible|infeasible|uncertain), defect_kind (short stable snake_case string), "
        "confidence (number 0..1), affected_protocols (array drawn from code, sql), and "
        "evidence (concise array of concrete facts)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(bundle, ensure_ascii=False, sort_keys=True)},
    ]


def _parse_judge_json(content: str) -> dict[str, Any]:
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge output must be an object")
    verdict = value.get("verdict")
    if verdict not in {"feasible", "infeasible", "uncertain"}:
        raise ValueError("judge verdict is invalid")
    defect_kind = value.get("defect_kind")
    if not isinstance(defect_kind, str) or not defect_kind.strip():
        raise ValueError("judge defect_kind is missing")
    confidence = float(value.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("judge confidence is outside [0, 1]")
    affected = value.get("affected_protocols")
    if not isinstance(affected, list) or any(item not in {"code", "sql"} for item in affected):
        raise ValueError("judge affected_protocols is invalid")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise ValueError("judge evidence must be a string list")
    return {
        "verdict": verdict,
        "defect_kind": defect_kind.strip().casefold(),
        "confidence": confidence,
        "affected_protocols": sorted(set(affected)),
        "evidence": evidence[:8],
    }


class DeepSeekIntegrityJudge:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key_env: str,
        timeout_seconds: float,
        max_tokens: int,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.provider_identities: set[tuple[str, Any]] = set()
        self.lock = asyncio.Lock()

    async def judge(self, messages: list[dict[str, str]], *, attempts: int) -> dict[str, Any]:
        errors = []
        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    extra_body={
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": "max",
                    },
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue

            message = response.choices[0].message
            usage = response.usage.model_dump() if response.usage is not None else {}
            identity = (str(response.model or ""), response.system_fingerprint)
            if identity[0] != self.model:
                errors.append(f"provider model mismatch: returned {identity[0]!r}, expected {self.model!r}")
                continue
            async with self.lock:
                if self.provider_identities and identity not in self.provider_identities:
                    raise RuntimeError(f"DeepSeek provider identity changed: {self.provider_identities!r} -> {identity!r}")
                self.provider_identities.add(identity)
                self.usage["requests"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.usage[key] += int(usage.get(key, 0) or 0)
            try:
                verdict = _parse_judge_json(message.content or "")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}; finish_reason={response.choices[0].finish_reason!r}; content_chars={len(message.content or '')}; reasoning_chars={len(getattr(message, 'reasoning_content', '') or '')}; completion_tokens={int(usage.get('completion_tokens', 0) or 0)}")
                continue
            return {
                "status": "ok",
                "attempt": attempt,
                "verdict": verdict,
                "finish_reason": response.choices[0].finish_reason,
                "model": identity[0],
                "system_fingerprint": identity[1],
                "usage": usage,
            }
        return {"status": "infrastructure_exhausted", "attempt": attempts, "errors": errors}


def judge_consensus(judgments: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    if len(judgments) != 2 or any(item.get("status") != "ok" for item in judgments):
        return "infrastructure_pending", ["judge_infrastructure_exhausted"]
    values = [dict(item["verdict"]) for item in judgments]
    if all(item["verdict"] == "infeasible" for item in values) and all(float(item["confidence"]) >= 0.9 for item in values) and len({str(item["defect_kind"]) for item in values}) == 1 and all(CURRENT_VERIFIER_PROTOCOL in item["affected_protocols"] for item in values):
        return "quarantine", [f"judge:{values[0]['defect_kind']}"]
    if all(item["verdict"] == "feasible" and float(item["confidence"]) >= 0.8 for item in values):
        return "pass", []
    return "needs_review", ["judge_disagreement_or_uncertainty"]


def select_judge_task_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
    clean_controls: int = 16,
) -> list[str]:
    eligible = [record for record in records if record.get("status") == "pass"]
    by_id = {str(record["task_id"]): record for record in eligible}
    selected = [task_id for task_id in KNOWN_CALIBRATION_TASKS if task_id in by_id]
    selected_set = set(selected)
    warnings = sorted(
        (record for record in eligible if record.get("semantic_warning_codes") and record["task_id"] not in selected_set),
        key=lambda record: stable_rank(str(record["task_id"])),
    )
    warning_budget = max(0, maximum - clean_controls - len(selected))
    for record in warnings[:warning_budget]:
        selected.append(str(record["task_id"]))
        selected_set.add(str(record["task_id"]))

    clean = [record for record in eligible if not record.get("semantic_warning_codes") and record["task_id"] not in selected_set]
    if clean and len(selected) < maximum:
        values = np.asarray([int(record["native_prompt_tokens"]) for record in clean], dtype=np.float64)
        boundaries = np.quantile(values, [0.25, 0.5, 0.75])
        quartiles: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for record in clean:
            quartile = min(int(np.searchsorted(boundaries, int(record["native_prompt_tokens"]), side="right")), 3)
            quartiles[quartile].append(record)
        per_quartile = max(1, clean_controls // 4)
        for quartile in range(4):
            ranked = sorted(quartiles[quartile], key=lambda record: stable_rank(str(record["task_id"])))
            for record in ranked[:per_quartile]:
                if len(selected) >= maximum:
                    break
                selected.append(str(record["task_id"]))
                selected_set.add(str(record["task_id"]))
    return selected[:maximum]


def _usage_totals(value: Mapping[str, Any] | None) -> dict[str, int]:
    value = value or {}
    return {key: int(value.get(key, 0) or 0) for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")}


def _add_usage(*values: Mapping[str, Any] | None) -> dict[str, int]:
    totals = _usage_totals(None)
    for value in values:
        for key, amount in _usage_totals(value).items():
            totals[key] += amount
    return totals


def _load_candidate_rows(data_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM integrity input selection protocol mismatch")
    if sha256_file(data_path) != manifest.get("candidate_data_sha256"):
        raise RuntimeError("AWM integrity input candidate parquet hash mismatch")
    frame = pd.read_parquet(data_path)
    rows = []
    for _, raw in frame.iterrows():
        extra = dict(raw["extra_info"])
        env_kwargs = dict(raw["env_kwargs"])
        rows.append(
            {
                "task_id": str(extra["task_id"]),
                "scenario": str(env_kwargs["scenario"]),
                "task_idx": int(env_kwargs["task_idx"]),
                "task": str(extra["task"]),
                "native_prompt_tokens": int(extra["native_prompt_tokens"]),
                "tool_schema_hash": str(extra["tool_schema_hash"]),
                "raw_tool_schema_hash": str(extra["raw_tool_schema_hash"]),
                "tool_schema_repair_count": int(extra["tool_schema_repair_count"]),
                "training_row": raw.to_dict(),
            }
        )
    if [row["task_id"] for row in rows] != manifest.get("task_ids"):
        raise RuntimeError("AWM integrity candidate IDs differ from selection manifest")
    return rows, manifest


def _source_indexes(data_dir: Path) -> dict[str, Any]:
    tasks = _load_multimap(
        data_dir / "gen_tasks.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    samples = _load_multimap(
        data_dir / "gen_sample.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    schemas = _load_multimap(
        data_dir / "gen_db.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    verifiers = {
        mode: _load_multimap(
            data_dir / filename,
            lambda record: (_normalize_scenario(record["scenario"]), int(record["task_idx"])),
        )
        for mode, filename in _VERIFIER_FILES.items()
    }
    return {"tasks": tasks, "samples": samples, "schemas": schemas, "verifiers": verifiers}


def _one_scenario_payload(index: Mapping[str, Sequence[Mapping[str, Any]]], scenario: str, field: str) -> Any:
    records = list(index.get(_normalize_scenario(scenario)) or [])
    if len(records) != 1:
        return {}
    return records[0].get(field) or {}


def _training_row(row: Mapping[str, Any], status: str) -> dict[str, Any]:
    output = dict(row["training_row"])
    extra = dict(output["extra_info"])
    extra.update(
        {
            "awm_integrity_protocol_version": INTEGRITY_PROTOCOL_VERSION,
            "awm_integrity_status": status,
        }
    )
    output["extra_info"] = extra
    return output


async def audit_integrity(args) -> None:
    rows, selection_manifest = _load_candidate_rows(args.data, args.candidate_manifest)
    source_files = ["gen_tasks.jsonl", "gen_sample.jsonl", "gen_db.jsonl", *_VERIFIER_FILES.values()]
    identity = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_file_sha256": {name: sha256_file(args.awm_data_dir / name) for name in source_files},
        "selection_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "candidate_task_ids": [row["task_id"] for row in rows],
        "awm_base_url": args.awm_base_url,
        "runtime_attempts": int(args.runtime_attempts),
        "judge": {
            "enabled": not args.skip_judge,
            "model": args.model,
            "api_base": args.api_base,
            "maximum_tasks": int(args.max_judge_tasks),
            "independent_calls": 2,
            "minimum_quarantine_confidence": 0.9,
            "reasoning_effort": "max",
            "response_format": "json_object",
            "max_output_tokens": int(args.judge_max_tokens),
            "input_budget_qwen_tokens": 24000,
            "semantic_warning_policy": ("quoted requested literal already present in the explicitly named target table; or mutation request with no obvious mutation-capable tool"),
            "infrastructure_attempts": int(args.judge_attempts),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    prior_manifest: dict[str, Any] = {}
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != identity:
            raise RuntimeError("AWM integrity audit resume configuration mismatch")
        prior_manifest_path = args.output_dir / "integrity_manifest.json"
        if prior_manifest_path.is_file():
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    indexes = _source_indexes(args.awm_data_dir)
    static_path = args.output_dir / "static_audit.jsonl"
    static_by_id = {
        str(record["task_id"]): refresh_cached_static_record(
            record,
            sample_data=_one_scenario_payload(
                indexes["samples"],
                str(record["scenario"]),
                "sample_data",
            ),
        )
        for record in _load_jsonl(static_path)
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def audit_row(row):
        task_id = str(row["task_id"])
        if task_id in static_by_id:
            return
        scenario_key = _normalize_scenario(row["scenario"])
        key = (scenario_key, int(row["task_idx"]))
        sample_data = _one_scenario_payload(indexes["samples"], row["scenario"], "sample_data")
        static, _ = static_source_audit(
            row,
            task_records=indexes["tasks"].get(scenario_key) or [],
            code_records=indexes["verifiers"]["code"].get(key) or [],
            sql_records=indexes["verifiers"]["sql"].get(key) or [],
            sample_data=sample_data,
        )
        runtime = await runtime_audit(
            row,
            awm_base_url=args.awm_base_url,
            semaphore=semaphore,
            attempts=args.runtime_attempts,
        )
        record = {
            "task_id": task_id,
            "scenario": row["scenario"],
            "task_idx": int(row["task_idx"]),
            "task": row["task"],
            "native_prompt_tokens": int(row["native_prompt_tokens"]),
            "tool_schema_repair_count": int(row["tool_schema_repair_count"]),
            **static,
            "runtime": runtime,
        }
        record = refine_semantic_warnings(record)
        status, reasons = classify_static_record(record)
        record["status"] = status
        record["status_reasons"] = reasons
        async with write_lock:
            if task_id not in static_by_id:
                static_by_id[task_id] = record
                _append_jsonl(static_path, record)
                if len(static_by_id) % 50 == 0:
                    print(f"integrity_static {len(static_by_id)}/{len(rows)}", flush=True)

    await asyncio.gather(*(audit_row(row) for row in rows))
    if set(static_by_id) != {row["task_id"] for row in rows}:
        raise RuntimeError("AWM integrity static audit is incomplete")
    ordered_static = [static_by_id[row["task_id"]] for row in rows]
    with static_path.open("w", encoding="utf-8") as handle:
        for record in ordered_static:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    judge_ids = [] if args.skip_judge else select_judge_task_ids(ordered_static, maximum=args.max_judge_tasks)
    judge_path = args.output_dir / "judge_audit.jsonl"
    judge_records = _load_jsonl(judge_path)
    judge_by_key = {(str(record["task_id"]), int(record["judge_index"])): record for record in judge_records}
    judge_client = None
    if judge_ids:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        judge_client = DeepSeekIntegrityJudge(
            model=args.model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.judge_max_tokens,
        )
        judge_slots = asyncio.Semaphore(args.judge_concurrency)
        rows_by_id = {row["task_id"]: row for row in rows}

        async def judge_once(task_id: str, judge_index: int):
            key = (task_id, judge_index)
            if key in judge_by_key:
                return
            row = rows_by_id[task_id]
            scenario_key = _normalize_scenario(row["scenario"])
            verifier_key = (scenario_key, int(row["task_idx"]))
            _, verifier_sources = static_source_audit(
                row,
                task_records=indexes["tasks"].get(scenario_key) or [],
                code_records=indexes["verifiers"]["code"].get(verifier_key) or [],
                sql_records=indexes["verifiers"]["sql"].get(verifier_key) or [],
                sample_data=_one_scenario_payload(indexes["samples"], row["scenario"], "sample_data"),
            )
            messages = build_judge_messages(
                row,
                static_by_id[task_id],
                verifier_sources,
                sample_data=_one_scenario_payload(indexes["samples"], row["scenario"], "sample_data"),
                db_schema=_one_scenario_payload(indexes["schemas"], row["scenario"], "db_schema"),
                tokenizer=tokenizer,
                judge_index=judge_index,
            )
            rendered_tokens = len(tokenizer.encode(_canonical_json(messages), add_special_tokens=False))
            if rendered_tokens > 24000:
                raise RuntimeError(f"integrity judge prompt exceeds 24K tokens: {task_id}={rendered_tokens}")
            async with judge_slots:
                result = await judge_client.judge(messages, attempts=args.judge_attempts)
            record = {
                "task_id": task_id,
                "judge_index": judge_index,
                "judge_prompt_qwen_tokens": rendered_tokens,
                **result,
            }
            async with write_lock:
                if key not in judge_by_key:
                    judge_by_key[key] = record
                    _append_jsonl(judge_path, record)
                    print(f"integrity_judge {len(judge_by_key)}/{len(judge_ids) * 2}", flush=True)

        try:
            await asyncio.gather(*(judge_once(task_id, index) for task_id in judge_ids for index in range(2)))
        finally:
            await judge_client.client.close()

    final_records = []
    for record in ordered_static:
        final = dict(record)
        if record["task_id"] in judge_ids:
            judgments = [judge_by_key.get((record["task_id"], index), {}) for index in range(2)]
            status, reasons = judge_consensus(judgments)
            final["judgments"] = judgments
            final["status"] = status
            final["status_reasons"] = reasons
        elif record["status"] == "pass" and record.get("semantic_warning_codes"):
            final["status"] = "needs_review"
            final["status_reasons"] = ["semantic_warning_not_judged"]
        final_records.append(final)

    audit_path = args.output_dir / "integrity_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    judge_records = [judge_by_key[key] for key in sorted(judge_by_key)]
    with judge_path.open("w", encoding="utf-8") as handle:
        for record in judge_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    status_by_id = {record["task_id"]: record["status"] for record in final_records}
    passing_rows = [row for row in rows if status_by_id[row["task_id"]] == "pass"]
    filtered_path = args.output_dir / "awm_integrity_filtered.parquet"
    pd.DataFrame([_training_row(row, "pass") for row in passing_rows]).to_parquet(filtered_path, index=False)
    for name, status in (
        ("quarantine_task_ids.json", "quarantine"),
        ("needs_review_task_ids.json", "needs_review"),
        ("infrastructure_pending_task_ids.json", "infrastructure_pending"),
    ):
        task_ids = [record["task_id"] for record in final_records if record["status"] == status]
        (args.output_dir / name).write_text(json.dumps(task_ids, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = defaultdict(int)
    for record in final_records:
        counts[str(record["status"])] += 1
    live_judge_usage = judge_client.usage if judge_client is not None else _usage_totals(None)
    prior_cumulative_usage = prior_manifest.get("cumulative_judge_usage")
    if prior_cumulative_usage is None and prior_manifest:
        prior_cumulative_usage = prior_manifest.get("live_judge_usage")
    cumulative_judge_usage = _add_usage(prior_cumulative_usage, live_judge_usage)
    provider_identities = {(str(item.get("model") or ""), item.get("system_fingerprint")) for item in prior_manifest.get("judge_provider_identities") or []}
    provider_identities.update((str(record.get("model") or ""), record.get("system_fingerprint")) for record in judge_records if record.get("status") == "ok")
    if judge_client is not None:
        provider_identities.update(judge_client.provider_identities)
    provider_identities.discard(("", None))
    manifest = {
        **identity,
        "kind": "awm_task_integrity_filter",
        "selection_counts": selection_manifest["selected_counts"],
        "counts": dict(sorted(counts.items())),
        "judge_task_ids": judge_ids,
        "judge_provider_identities": sorted(
            [{"model": model, "system_fingerprint": fingerprint} for model, fingerprint in provider_identities],
            key=_canonical_json,
        ),
        "live_judge_usage": live_judge_usage,
        "cumulative_judge_usage": cumulative_judge_usage,
        "filtered_task_ids": [row["task_id"] for row in passing_rows],
        "static_audit_sha256": sha256_file(static_path),
        "judge_audit_sha256": sha256_file(judge_path),
        "integrity_audit_sha256": sha256_file(audit_path),
        "filtered_data_sha256": sha256_file(filtered_path),
        "quarantine_task_ids_sha256": sha256_file(args.output_dir / "quarantine_task_ids.json"),
        "needs_review_task_ids_sha256": sha256_file(args.output_dir / "needs_review_task_ids.json"),
        "infrastructure_pending_task_ids_sha256": sha256_file(args.output_dir / "infrastructure_pending_task_ids.json"),
    }
    (args.output_dir / "integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"counts": manifest["counts"], "judge_tasks": len(judge_ids)}, indent=2, sort_keys=True))


def verify_integrity(output_dir: Path) -> None:
    manifest_path = output_dir / "integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != INTEGRITY_PROTOCOL_VERSION:
        raise RuntimeError("AWM integrity manifest protocol mismatch")
    paths = {
        "static_audit_sha256": output_dir / "static_audit.jsonl",
        "judge_audit_sha256": output_dir / "judge_audit.jsonl",
        "integrity_audit_sha256": output_dir / "integrity_audit.jsonl",
        "filtered_data_sha256": output_dir / "awm_integrity_filtered.parquet",
        "quarantine_task_ids_sha256": output_dir / "quarantine_task_ids.json",
        "needs_review_task_ids_sha256": output_dir / "needs_review_task_ids.json",
        "infrastructure_pending_task_ids_sha256": output_dir / "infrastructure_pending_task_ids.json",
    }
    for field, path in paths.items():
        if sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM integrity artifact hash mismatch: {path}")
    records = _load_jsonl(output_dir / "integrity_audit.jsonl")
    if len(records) != len(manifest["candidate_task_ids"]):
        raise RuntimeError("AWM integrity audit record count mismatch")
    if [record["task_id"] for record in records] != manifest["candidate_task_ids"]:
        raise RuntimeError("AWM integrity audit task order mismatch")
    computed_counts: dict[str, int] = defaultdict(int)
    for record in records:
        computed_counts[str(record["status"])] += 1
    if dict(sorted(computed_counts.items())) != manifest["counts"]:
        raise RuntimeError("AWM integrity manifest status counts mismatch")
    for name, status in (
        ("quarantine_task_ids.json", "quarantine"),
        ("needs_review_task_ids.json", "needs_review"),
        ("infrastructure_pending_task_ids.json", "infrastructure_pending"),
    ):
        task_ids = json.loads((output_dir / name).read_text(encoding="utf-8"))
        expected_ids = [record["task_id"] for record in records if record["status"] == status]
        if task_ids != expected_ids:
            raise RuntimeError(f"AWM integrity {status} task IDs differ from the audit records")
    pass_ids = [record["task_id"] for record in records if record["status"] == "pass"]
    frame = pd.read_parquet(output_dir / "awm_integrity_filtered.parquet")
    filtered_ids = [str(extra["task_id"]) for extra in frame["extra_info"].tolist()]
    if filtered_ids != manifest["filtered_task_ids"]:
        raise RuntimeError("AWM integrity filtered parquet IDs mismatch")
    if filtered_ids != pass_ids:
        raise RuntimeError("AWM integrity filtered parquet does not exactly contain pass tasks")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--awm-data-dir", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--runtime-attempts", type=int, default=3)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--max-judge-tasks", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_integrity(args.output_dir)
        return
    for name in ("data", "candidate_manifest", "awm_data_dir", "tokenizer"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    limits = (
        args.concurrency,
        args.runtime_attempts,
        args.judge_concurrency,
        args.judge_attempts,
        args.judge_max_tokens,
        args.max_judge_tasks,
        args.timeout_seconds,
    )
    if any(value <= 0 for value in limits):
        parser.error("concurrency, attempts, task, and timeout limits must be positive")
    asyncio.run(audit_integrity(args))


if __name__ == "__main__":
    main()
