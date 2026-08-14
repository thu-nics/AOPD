"""Shared local utilities for deterministic AWM data auditing."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..runtime.rollout import sha256_file
from .integrity import _load_jsonl
from .selection import SELECTION_PROTOCOL_VERSION

SCENARIO_HEALTH_FILENAME = "scenario_health.jsonl"


def _normalize_scenario(value: str) -> str:
    result = re.sub(r"[^a-z0-9_]", "_", str(value).lower())
    return re.sub(r"_+", "_", result).strip("_")


def _load_multimap(path: Path, key) -> dict[Any, list[dict[str, Any]]]:
    output: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in _load_jsonl(path):
        output[key(record)].append(record)
    return dict(output)


def _unique_source(
    records: Sequence[Mapping[str, Any]],
    *,
    missing_reason: str,
    conflict_reason: str,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    if not records:
        return None, [missing_reason]
    if len(records) != 1:
        return None, [conflict_reason]
    return records[0], []


def _strict_database_check(
    schema_record: Mapping[str, Any] | None,
    sample_record: Mapping[str, Any] | None,
) -> list[str]:
    if schema_record is None or sample_record is None:
        return []
    schema = schema_record.get("db_schema") or {}
    sample = sample_record.get("sample_data") or {}
    errors = []
    with tempfile.TemporaryDirectory(prefix="awm-health-") as directory:
        connection = sqlite3.connect(Path(directory) / "initial.db")
        try:
            cursor = connection.cursor()
            for table in schema.get("tables") or []:
                ddl = str(table.get("ddl") or "").strip()
                if ddl:
                    try:
                        cursor.execute(ddl)
                    except sqlite3.Error as exc:
                        errors.append(f"ddl:{table.get('name') or 'unknown'}:{type(exc).__name__}:{exc}")
                for statement in table.get("indexes") or []:
                    statement = str(statement).strip()
                    if not statement:
                        continue
                    try:
                        cursor.execute(statement)
                    except sqlite3.Error as exc:
                        errors.append(f"index:{table.get('name') or 'unknown'}:{type(exc).__name__}:{exc}")
            tables = sample.get("tables") if isinstance(sample, Mapping) else sample
            for item in tables or []:
                if isinstance(item, Mapping):
                    table_name = str(item.get("table_name") or "unknown")
                    statements = item.get("insert_statements") or []
                else:
                    table_name = "unknown"
                    statements = [item]
                for statement in statements:
                    statement = str(statement).strip()
                    if not statement:
                        continue
                    try:
                        cursor.execute(statement)
                    except sqlite3.Error as exc:
                        errors.append(f"seed_insert:{table_name}:{type(exc).__name__}:{exc}")
            if errors:
                connection.rollback()
            else:
                connection.commit()
        finally:
            connection.close()
    return errors


def audit_scenarios(data_dir: Path, scenarios: Sequence[str]) -> list[dict[str, Any]]:
    tasks = _load_multimap(
        data_dir / "gen_tasks.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    schemas = _load_multimap(
        data_dir / "gen_db.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    samples = _load_multimap(
        data_dir / "gen_sample.jsonl",
        lambda record: _normalize_scenario(record["scenario"]),
    )
    records = []
    for scenario in scenarios:
        key = _normalize_scenario(scenario)
        task_record, task_errors = _unique_source(
            tasks.get(key) or [],
            missing_reason="missing_task_source",
            conflict_reason="conflicting_task_source",
        )
        schema_record, schema_errors = _unique_source(
            schemas.get(key) or [],
            missing_reason="missing_schema_source",
            conflict_reason="conflicting_schema_source",
        )
        sample_record, sample_errors = _unique_source(
            samples.get(key) or [],
            missing_reason="missing_sample_source",
            conflict_reason="conflicting_sample_source",
        )
        reasons = [*task_errors, *schema_errors, *sample_errors]
        if task_record is not None and len(task_record.get("tasks") or []) != 10:
            reasons.append("unexpected_task_count")
        database_errors = _strict_database_check(schema_record, sample_record)
        if database_errors:
            reasons.append("strict_database_build_failed")
        records.append(
            {
                "scenario": scenario,
                "status": "healthy" if not reasons else "quarantine",
                "status_reasons": sorted(set(reasons)),
                "database_errors": database_errors,
            }
        )
    return records


def audit_sql_verifier(
    row: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], str | None]:
    entry, reasons = _unique_source(
        records,
        missing_reason="missing_sql_verifier",
        conflict_reason="conflicting_sql_verifier",
    )
    if entry is None:
        return reasons, None
    if str(entry.get("task") or "") != str(row["task"]):
        reasons.append("sql_verifier_task_mismatch")
    verification = entry.get("verification")
    code = verification.get("code") if isinstance(verification, Mapping) else None
    if not isinstance(code, str) or not code.strip():
        reasons.append("sql_verifier_missing_code")
        return sorted(set(reasons)), None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code, filename="awm_sql_verifier.py")
            compile(tree, "awm_sql_verifier.py", "exec")
    except (SyntaxError, TypeError, ValueError):
        reasons.append("sql_verifier_compile_error")
        return sorted(set(reasons)), hashlib.sha256(code.encode()).hexdigest()
    functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "verify_task" not in functions:
        reasons.append("sql_verifier_missing_entrypoint")
    return sorted(set(reasons)), hashlib.sha256(code.encode()).hexdigest()


def _load_candidate_rows(
    data_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM healthy-pool context-selection protocol mismatch")
    if sha256_file(data_path) != manifest.get("candidate_data_sha256"):
        raise RuntimeError("AWM healthy-pool candidate parquet hash mismatch")
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
                "tool_schema_hash": str(extra["tool_schema_hash"]),
                "training_row": raw.to_dict(),
            }
        )
    if [row["task_id"] for row in rows] != manifest.get("task_ids"):
        raise RuntimeError("AWM healthy-pool candidate task order mismatch")
    return rows, manifest
