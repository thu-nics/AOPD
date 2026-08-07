"""Integrity audit and conservative quarantine for AWM training tasks."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from jsonschema import Draft202012Validator

from ..runtime.actions import tool_schema_audit
from ..runtime.rollout import observation_dict, sha256_file
from .prepare import DATASET_NAME, DATASET_REVISION, EXPECTED_SOURCE_SHA256
from .selection import SELECTION_PROTOCOL_VERSION

INTEGRITY_PROTOCOL_VERSION = 6
PREFILTER_PROTOCOL_VERSION = 3
TRAINING_POOL_PROTOCOL_VERSION = 3
TRAINING_POOL_FILENAME = "awm_training_pool.parquet"
CURRENT_VERIFIER_PROTOCOL = "code"
_VERIFIER_FILES = {
    "code": "gen_verifier.pure_code.jsonl",
    "sql": "gen_verifier.jsonl",
}
_REQUIRED_ENTRYPOINTS = {
    "code": "verify_task_completion",
    "sql": "verify_task",
}
_INFRA_REWARD_TYPES = frozenset({"judge_error", "server_error", "no_verifier", "timeout", "runtime_exception"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(
    path: Path,
    *,
    repair_torn_tail: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    repair = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if repair_torn_tail and index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                repair = True
                break
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}") from exc
    if repair_torn_tail and lines and not lines[-1].endswith(("\n", "\r")):
        repair = True
    if repair:
        _write_jsonl(path, records)
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
    finding_severity: str = "quarantine",
) -> Mapping[str, Any] | None:
    if not records:
        findings.append({"severity": finding_severity, "code": missing_code})
        return None
    fingerprints = {_sha256_json(identity(record) if identity is not None else record) for record in records}
    if len(fingerprints) > 1:
        findings.append(
            {
                "severity": finding_severity,
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
    finding_severity: str = "quarantine",
) -> dict[str, Any] | None:
    if entry is None:
        return None
    if str(entry.get("task") or "") != expected_task:
        findings.append(
            {
                "severity": finding_severity,
                "code": f"{mode}_verifier_task_mismatch",
            }
        )
    verification = entry.get("verification")
    code = verification.get("code") if isinstance(verification, Mapping) else None
    if not isinstance(code, str) or not code.strip():
        findings.append(
            {
                "severity": finding_severity,
                "code": f"{mode}_verifier_missing_code",
            }
        )
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code, filename=f"awm_{mode}_verifier.py")
            compile(tree, f"awm_{mode}_verifier.py", "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        findings.append(
            {
                "severity": finding_severity,
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
                "severity": finding_severity,
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
        finding_severity="warning",
    )
    verifier_sources = {
        "code": _verifier_check(code_entry, mode="code", expected_task=expected_task, findings=findings),
        "sql": _verifier_check(
            sql_entry,
            mode="sql",
            expected_task=expected_task,
            findings=findings,
            finding_severity="warning",
        ),
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
    mismatches = []
    for attempt in range(1, attempts + 1):
        try:
            result = await _runtime_audit_once(row, awm_base_url=awm_base_url, semaphore=semaphore)
            mismatch_reasons = []
            if not result.get("runtime_task_matches", False):
                mismatch_reasons.append("runtime_task_mismatch")
            if not result.get("raw_tool_schema_matches", False):
                mismatch_reasons.append("raw_tool_schema_mismatch")
            if not result.get("canonical_tool_schema_matches", False):
                mismatch_reasons.append("canonical_tool_schema_mismatch")
            if mismatch_reasons:
                mismatches.append(result)
                errors.append(f"runtime identity mismatch: {','.join(mismatch_reasons)}")
                continue
            reward_type = str(result.get("no_op_reward_type") or "")
            if reward_type in _INFRA_REWARD_TYPES:
                errors.append(f"no-op verifier infrastructure reward_type={reward_type}")
                continue
            return {
                "status": "ok",
                "attempt": attempt,
                "retry_errors": errors,
                **result,
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if len(mismatches) == attempts:
        return {
            "status": "ok",
            "attempt": attempts,
            "retry_errors": errors,
            **mismatches[-1],
        }
    return {
        "status": "infrastructure_exhausted",
        "attempt": attempts,
        "errors": errors,
    }


def classify_static_record(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons = [str(item["code"]) for item in record.get("findings") or [] if item.get("severity") == "quarantine"]
    runtime = record.get("runtime") or {}
    if runtime.get("status") == "infrastructure_exhausted":
        return "quarantine", ["runtime_infrastructure_exhausted"]
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
    reasons.extend(f"semantic_warning:{code}" for code in record.get("semantic_warning_codes") or [])
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
        if extra.get("selection_protocol_version") != SELECTION_PROTOCOL_VERSION:
            raise RuntimeError("AWM integrity candidate row selection protocol mismatch")
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


def _prefilter_training_row(row: Mapping[str, Any], status: str) -> dict[str, Any]:
    output = _training_row(row, status)
    extra = dict(output["extra_info"])
    extra.update(
        {
            "awm_prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
            "awm_prefilter_status": "candidate",
            "awm_training_pool_protocol_version": TRAINING_POOL_PROTOCOL_VERSION,
            "awm_training_pool_status": "active",
        }
    )
    output["extra_info"] = extra
    return output


def _write_prefilter_artifacts(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize the cheap deterministic partition without rerunning audits."""
    status_by_id = {str(record["task_id"]): str(record["status"]) for record in records}
    row_ids = [str(row["task_id"]) for row in rows]
    if row_ids != [str(record["task_id"]) for record in records]:
        raise RuntimeError("AWM prefilter rows and integrity records differ")
    rejected_ids = [task_id for task_id in row_ids if status_by_id[task_id] == "quarantine"]
    candidate_rows = [row for row in rows if status_by_id[str(row["task_id"])] == "pass"]
    candidate_ids = [str(row["task_id"]) for row in candidate_rows]

    candidate_path = output_dir / TRAINING_POOL_FILENAME
    pd.DataFrame([_prefilter_training_row(row, status_by_id[str(row["task_id"])]) for row in candidate_rows]).to_parquet(candidate_path, index=False)
    return {
        "prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
        "training_pool_protocol_version": TRAINING_POOL_PROTOCOL_VERSION,
        "training_pool_policy": "deterministic pass only; expert success is required by the final pool",
        "training_pool_task_ids": candidate_ids,
        "training_pool_data_sha256": sha256_file(candidate_path),
        "training_pool_filename": TRAINING_POOL_FILENAME,
        "prefilter_policy": "strict binary pass or quarantine",
        "prefilter_candidate_task_ids": candidate_ids,
        "rejected_prefilter_task_ids": rejected_ids,
        "prefilter_data_sha256": sha256_file(candidate_path),
    }


def _strict_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse legacy uncertainty states into the permanent quarantine."""
    output = dict(record)
    status = str(output.get("status") or "")
    if status == "pass":
        return output
    if status == "quarantine":
        return output
    if status not in {"needs_review", "infrastructure_pending"}:
        raise RuntimeError(f"unsupported AWM integrity status {status!r}")
    reasons = [str(value) for value in output.get("status_reasons") or []]
    reasons.append(f"legacy_{status}_permanent_quarantine")
    output["status"] = "quarantine"
    output["status_reasons"] = sorted(set(reasons))
    return output


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_integrity_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    output_dir: Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if [str(row["task_id"]) for row in rows] != [str(record["task_id"]) for record in records]:
        raise RuntimeError("AWM integrity rows and records differ")
    invalid = sorted({str(record.get("status")) for record in records} - {"pass", "quarantine"})
    if invalid:
        raise RuntimeError(f"AWM integrity output contains non-binary statuses: {invalid}")

    audit_path = output_dir / "integrity_audit.jsonl"
    _write_jsonl(audit_path, records)
    quarantine_ids = [str(record["task_id"]) for record in records if record["status"] == "quarantine"]
    quarantine_path = output_dir / "quarantine_task_ids.json"
    quarantine_path.write_text(json.dumps(quarantine_ids, indent=2) + "\n", encoding="utf-8")
    pool_fields = _write_prefilter_artifacts(rows, records, output_dir)
    counts = {status: sum(str(record["status"]) == status for record in records) for status in ("pass", "quarantine")}
    manifest = {
        **identity,
        "kind": "awm_task_integrity_filter",
        "selection_counts": selection_manifest["selected_counts"],
        "counts": counts,
        "filtered_task_ids": [str(record["task_id"]) for record in records if record["status"] == "pass"],
        "integrity_audit_sha256": sha256_file(audit_path),
        "quarantine_task_ids_sha256": sha256_file(quarantine_path),
        **pool_fields,
    }
    if provenance is not None:
        manifest["migration_provenance"] = dict(provenance)
    (output_dir / "integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _verify_legacy_integrity(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly verify the protocol-v5 evidence used for the zero-API migration."""
    manifest_path = source_dir / "integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != 5 or manifest.get("kind") != "awm_task_integrity_filter":
        raise RuntimeError("AWM migration source must be a protocol-v5 integrity artifact")
    paths = {
        "static_audit_sha256": source_dir / "static_audit.jsonl",
        "integrity_audit_sha256": source_dir / "integrity_audit.jsonl",
        "quarantine_task_ids_sha256": source_dir / "quarantine_task_ids.json",
        "needs_review_task_ids_sha256": source_dir / "needs_review_task_ids.json",
        "infrastructure_pending_task_ids_sha256": source_dir / "infrastructure_pending_task_ids.json",
        "training_pool_data_sha256": source_dir / str(manifest.get("training_pool_filename")),
    }
    for field, path in paths.items():
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM migration source hash mismatch: {path}")
    records = _load_jsonl(source_dir / "integrity_audit.jsonl")
    candidate_ids = [str(value) for value in manifest.get("candidate_task_ids") or []]
    if [str(record.get("task_id")) for record in records] != candidate_ids:
        raise RuntimeError("AWM migration source task order mismatch")
    allowed = {"pass", "quarantine", "needs_review", "infrastructure_pending"}
    if {str(record.get("status")) for record in records} - allowed:
        raise RuntimeError("AWM migration source contains unknown statuses")
    return manifest, records


def migrate_integrity(
    *,
    source_dir: Path,
    data_path: Path,
    candidate_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Reuse verified protocol-v5 evidence while permanently quarantining uncertainty."""
    source_manifest, source_records = _verify_legacy_integrity(source_dir)
    rows, selection_manifest = _load_candidate_rows(data_path, candidate_manifest_path)
    candidate_ids = [str(row["task_id"]) for row in rows]
    if candidate_ids != [str(record["task_id"]) for record in source_records]:
        raise RuntimeError("AWM migration candidate IDs differ from the legacy audit")
    if source_manifest.get("candidate_data_sha256") != sha256_file(data_path):
        raise RuntimeError("AWM migration candidate parquet mismatch")
    if source_manifest.get("selection_manifest_sha256") != sha256_file(candidate_manifest_path):
        raise RuntimeError("AWM migration selection manifest mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_records = [_strict_record(record) for record in source_records]
    identity = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "dataset": source_manifest["dataset"],
        "dataset_revision": source_manifest["dataset_revision"],
        "source_sha256": source_manifest["source_sha256"],
        "source_file_sha256": source_manifest["source_file_sha256"],
        "verifier_protocol": CURRENT_VERIFIER_PROTOCOL,
        "selection_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_data_sha256": sha256_file(data_path),
        "candidate_task_ids": candidate_ids,
        "awm_base_url": source_manifest["awm_base_url"],
        "runtime_attempts": int(source_manifest["runtime_attempts"]),
        "deterministic_policy": "binary pass/quarantine; warnings and infrastructure exhaustion are permanent quarantine",
    }
    (output_dir / "config.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "kind": "protocol_v5_binary_status_migration",
        "source_integrity_manifest": str(source_dir / "integrity_manifest.json"),
        "source_integrity_manifest_sha256": sha256_file(source_dir / "integrity_manifest.json"),
        "source_protocol_version": 5,
        "api_calls": 0,
        "status_mapping": {
            "pass": "pass",
            "quarantine": "quarantine",
            "needs_review": "quarantine",
            "infrastructure_pending": "quarantine",
        },
    }
    manifest = _write_integrity_artifacts(
        rows=rows,
        records=final_records,
        selection_manifest=selection_manifest,
        identity=identity,
        output_dir=output_dir,
        provenance=provenance,
    )
    verify_integrity(output_dir)
    return manifest


async def audit_integrity(args) -> None:
    rows, selection_manifest = _load_candidate_rows(args.data, args.candidate_manifest)
    source_files = ["gen_tasks.jsonl", "gen_sample.jsonl", "gen_db.jsonl", *_VERIFIER_FILES.values()]
    identity = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_file_sha256": {name: sha256_file(args.awm_data_dir / name) for name in source_files},
        "verifier_protocol": CURRENT_VERIFIER_PROTOCOL,
        "selection_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "candidate_task_ids": [row["task_id"] for row in rows],
        "awm_base_url": args.awm_base_url,
        "runtime_attempts": int(args.runtime_attempts),
        "deterministic_policy": "binary pass/quarantine; warnings and infrastructure exhaustion are permanent quarantine",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("AWM integrity audit resume configuration mismatch")
        if (args.output_dir / "integrity_manifest.json").is_file():
            verify_integrity(args.output_dir)
            return
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    indexes = _source_indexes(args.awm_data_dir)
    static_path = args.output_dir / "static_audit.jsonl"
    static_by_id = {
        str(record["task_id"]): refresh_cached_static_record(
            record,
            sample_data=_one_scenario_payload(indexes["samples"], str(record["scenario"]), "sample_data"),
        )
        for record in _load_jsonl(static_path, repair_torn_tail=True)
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def audit_row(row: Mapping[str, Any]) -> None:
        task_id = str(row["task_id"])
        if task_id in static_by_id:
            return
        scenario_key = _normalize_scenario(str(row["scenario"]))
        key = (scenario_key, int(row["task_idx"]))
        sample_data = _one_scenario_payload(indexes["samples"], str(row["scenario"]), "sample_data")
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
        record = refine_semantic_warnings(
            {
                "task_id": task_id,
                "scenario": row["scenario"],
                "task_idx": int(row["task_idx"]),
                "task": row["task"],
                "native_prompt_tokens": int(row["native_prompt_tokens"]),
                "tool_schema_repair_count": int(row["tool_schema_repair_count"]),
                **static,
                "runtime": runtime,
            },
            sample_data=sample_data,
        )
        record["status"], record["status_reasons"] = classify_static_record(record)
        async with write_lock:
            if task_id not in static_by_id:
                static_by_id[task_id] = record
                _append_jsonl(static_path, record)
                if len(static_by_id) % 50 == 0:
                    print(f"integrity_static {len(static_by_id)}/{len(rows)}", flush=True)

    await asyncio.gather(*(audit_row(row) for row in rows))
    ordered = [static_by_id[row["task_id"]] for row in rows]
    if len(ordered) != len(static_by_id):
        raise RuntimeError("AWM integrity static audit contains unexpected tasks")
    _write_jsonl(static_path, ordered)
    manifest = _write_integrity_artifacts(
        rows=rows,
        records=ordered,
        selection_manifest=selection_manifest,
        identity=identity,
        output_dir=args.output_dir,
    )
    static_path.unlink()
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


def verify_integrity(output_dir: Path) -> None:
    manifest_path = output_dir / "integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != INTEGRITY_PROTOCOL_VERSION:
        raise RuntimeError("AWM integrity manifest protocol mismatch")
    if manifest.get("kind") != "awm_task_integrity_filter":
        raise RuntimeError("AWM integrity manifest kind mismatch")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in config.items()):
        raise RuntimeError("AWM integrity manifest/config identity mismatch")
    if manifest.get("prefilter_protocol_version") != PREFILTER_PROTOCOL_VERSION:
        raise RuntimeError("AWM prefilter protocol mismatch")
    if manifest.get("training_pool_protocol_version") != TRAINING_POOL_PROTOCOL_VERSION:
        raise RuntimeError("AWM training-pool protocol mismatch")
    if manifest.get("training_pool_filename") != TRAINING_POOL_FILENAME:
        raise RuntimeError("AWM training-pool filename mismatch")
    paths = {
        "integrity_audit_sha256": output_dir / "integrity_audit.jsonl",
        "quarantine_task_ids_sha256": output_dir / "quarantine_task_ids.json",
        "training_pool_data_sha256": output_dir / TRAINING_POOL_FILENAME,
    }
    for field, path in paths.items():
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM integrity artifact hash mismatch: {path}")

    records = _load_jsonl(output_dir / "integrity_audit.jsonl")
    candidate_ids = [str(value) for value in manifest.get("candidate_task_ids") or []]
    if [str(record.get("task_id")) for record in records] != candidate_ids:
        raise RuntimeError("AWM integrity audit task order mismatch")
    statuses = [str(record.get("status")) for record in records]
    if set(statuses) - {"pass", "quarantine"}:
        raise RuntimeError("AWM integrity audit contains non-binary statuses")
    counts = {status: statuses.count(status) for status in ("pass", "quarantine")}
    if manifest.get("counts") != counts:
        raise RuntimeError("AWM integrity manifest status counts mismatch")
    pass_ids = [task_id for task_id, status in zip(candidate_ids, statuses, strict=True) if status == "pass"]
    quarantine_ids = [task_id for task_id, status in zip(candidate_ids, statuses, strict=True) if status == "quarantine"]
    if json.loads((output_dir / "quarantine_task_ids.json").read_text(encoding="utf-8")) != quarantine_ids:
        raise RuntimeError("AWM quarantine IDs differ from audit records")
    if manifest.get("filtered_task_ids") != pass_ids or manifest.get("training_pool_task_ids") != pass_ids:
        raise RuntimeError("AWM deterministic pass IDs differ from manifest")
    if manifest.get("rejected_prefilter_task_ids") != quarantine_ids:
        raise RuntimeError("AWM deterministic quarantine IDs differ from manifest")
    frame = pd.read_parquet(output_dir / TRAINING_POOL_FILENAME)
    extras = [dict(extra) for extra in frame["extra_info"].tolist()]
    pool_ids = [str(extra["task_id"]) for extra in extras]
    if pool_ids != pass_ids:
        raise RuntimeError("AWM deterministic pool does not exactly contain pass tasks")
    if any(extra.get("awm_integrity_status") != "pass" or extra.get("awm_training_pool_protocol_version") != TRAINING_POOL_PROTOCOL_VERSION for extra in extras):
        raise RuntimeError("AWM deterministic pool row metadata mismatch")
    if len(pass_ids) + len(quarantine_ids) != len(candidate_ids):
        raise RuntimeError("AWM deterministic partition is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--awm-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--runtime-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--migrate-from", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        verify_integrity(args.output_dir)
        return
    if args.migrate_from is not None:
        if args.data is None or args.candidate_manifest is None:
            parser.error("--data and --candidate-manifest are required for migration")
        manifest = migrate_integrity(
            source_dir=args.migrate_from,
            data_path=args.data,
            candidate_manifest_path=args.candidate_manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return
    for name in ("data", "candidate_manifest", "awm_data_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.concurrency <= 0 or args.runtime_attempts <= 0:
        parser.error("concurrency and runtime attempts must be positive")
    asyncio.run(audit_integrity(args))


if __name__ == "__main__":
    main()
