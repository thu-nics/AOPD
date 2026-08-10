"""Build and verify the AWM verifier-reliable healthy task pool."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from jsonschema import Draft202012Validator

from ..runtime.actions import tool_schema_audit
from ..runtime.rollout import observation_dict, sha256_file
from ..runtime.terminal_judge import (
    DEFAULT_TERMINAL_JUDGE_API_BASE,
    DEFAULT_TERMINAL_JUDGE_MODEL,
    TERMINAL_JUDGE_PROTOCOL_VERSION,
)
from .integrity import _load_jsonl, _write_jsonl
from .prepare import (
    DATASET_NAME,
    DATASET_REVISION,
    EXPECTED_SOURCE_SHA256,
)
from .selection import SELECTION_PROTOCOL_VERSION

HEALTH_POOL_PROTOCOL_VERSION = 1
HEALTH_POOL_KIND = "awm_healthy_task_pool"
HEALTH_POOL_FILENAME = "awm_training_pool.parquet"
HEALTH_MANIFEST_FILENAME = "health_manifest.json"
SCENARIO_HEALTH_FILENAME = "scenario_health.jsonl"
TASK_HEALTH_FILENAME = "task_health.jsonl"

_VALID_NOOP_LABELS = frozenset({"complete", "incomplete", "server_error", "agent_error"})
_HEALTHY_NOOP_LABELS = frozenset({"incomplete", "agent_error"})
_INFRA_NOOP_LABELS = frozenset({"judge_error", "no_verifier", "timeout", "runtime_exception"})


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


def _compact_verify_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    verify_result = payload.get("verify_result")
    judge = verify_result.get("llm_judge") if isinstance(verify_result, Mapping) else None
    compact = {
        "reward_type": payload.get("reward_type"),
        "error": payload.get("error"),
        "judge": judge if isinstance(judge, Mapping) else None,
    }
    if isinstance(verify_result, Mapping):
        compact["execution_status"] = verify_result.get("execution_status")
        compact["result"] = verify_result.get("result")
        compact["llm_judge_error"] = verify_result.get("llm_judge_error")
    return compact


async def _noop_once(
    row: Mapping[str, Any],
    *,
    base_url: str,
    api_base: str,
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    async with semaphore:
        async with AWMEnv(base_url=base_url) as env:
            reset = await env.reset(
                scenario=str(row["scenario"]),
                task_idx=int(row["task_idx"]),
                seed=0,
                llm_base_url=api_base,
                llm_api_key=api_key,
                llm_model=model,
            )
            reset_payload = observation_dict(reset)
            if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"reset failed: {reset_payload}")
            if str(reset_payload.get("task") or "") != str(row["task"]):
                raise RuntimeError("runtime task identity mismatch")
            raw_tools = await env.list_tools(use_cache=False)
            schema = tool_schema_audit(raw_tools)
            for tool in schema["canonical_tools"]:
                Draft202012Validator.check_schema(tool["inputSchema"])
            expected_hash = str(row.get("tool_schema_hash") or "")
            if expected_hash and schema["canonical_tool_schema_hash"] != expected_hash:
                raise RuntimeError("runtime canonical tool schema mismatch")
            result = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": "sql", "final_answer": None},
                )
            )
            payload = observation_dict(result)
            try:
                await env.step(CallToolAction(tool_name="done", arguments={}))
            except Exception:
                pass
    return _compact_verify_payload(payload)


async def audit_noop(
    row: Mapping[str, Any],
    *,
    base_url: str,
    api_base: str,
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
    attempts: int,
) -> dict[str, Any]:
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            payload = await _noop_once(
                row,
                base_url=base_url,
                api_base=api_base,
                api_key=api_key,
                model=model,
                semaphore=semaphore,
            )
            label = str(payload.get("reward_type") or "")
            if label in _VALID_NOOP_LABELS:
                status = "healthy" if label in _HEALTHY_NOOP_LABELS else "quarantine"
                reason = None if status == "healthy" else f"noop_{label}"
                return {
                    "attempt": attempt,
                    "label": label,
                    "status": status,
                    "status_reason": reason,
                    "payload": payload,
                    "retry_errors": errors,
                }
            if label not in _INFRA_NOOP_LABELS:
                errors.append(f"unexpected reward_type={label!r}")
            else:
                errors.append(f"infrastructure reward_type={label}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "attempt": attempts,
        "label": "judge_error",
        "status": "quarantine",
        "status_reason": "noop_judge_exhausted",
        "payload": None,
        "retry_errors": errors,
    }


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


def load_expert_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid expert JSONL record {line_number} in {path}") from exc
            task_id = str(record.get("task_id") or "")
            if not task_id or task_id in output:
                raise RuntimeError("expert one-off metadata must contain unique non-empty task IDs")
            result = record.get("result")
            result = result if isinstance(result, Mapping) else {}
            trajectory = result.get("trajectory")
            first_step = trajectory[0] if isinstance(trajectory, list) and trajectory and isinstance(trajectory[0], Mapping) else {}
            output[task_id] = {
                "available": True,
                "status": str(record.get("status") or "unknown"),
                "legacy_status": str(record.get("legacy_status") or ""),
                "success": bool(result.get("success", False)),
                "reward_type": str(result.get("reward_type") or ""),
                "model": str(first_step.get("model") or ""),
            }
    return output


def _training_row(row: Mapping[str, Any], expert_metadata: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row["training_row"])
    extra = dict(output["extra_info"])
    extra.update(
        {
            "awm_health_pool_protocol_version": HEALTH_POOL_PROTOCOL_VERSION,
            "awm_health_status": "healthy",
            "awm_training_pool_status": "active",
            "awm_expert_one_off": dict(expert_metadata),
        }
    )
    output["extra_info"] = extra
    return output


def verify_healthy_pool(data: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != HEALTH_POOL_PROTOCOL_VERSION:
        raise RuntimeError("AWM healthy-pool protocol mismatch")
    if manifest.get("kind") != HEALTH_POOL_KIND:
        raise RuntimeError("AWM training manifest is not a healthy task pool")
    if data.name != HEALTH_POOL_FILENAME:
        raise RuntimeError(f"AWM healthy pool data must be named {HEALTH_POOL_FILENAME}")
    root = manifest_path.parent
    paths = {
        "scenario_health_sha256": root / SCENARIO_HEALTH_FILENAME,
        "task_health_sha256": root / TASK_HEALTH_FILENAME,
        "training_pool_data_sha256": data,
    }
    for field, path in paths.items():
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM healthy-pool artifact hash mismatch: {path}")
    task_records = _load_jsonl(root / TASK_HEALTH_FILENAME)
    scenario_records = _load_jsonl(root / SCENARIO_HEALTH_FILENAME)
    for record in scenario_records:
        status = record.get("status")
        reasons = list(record.get("status_reasons") or [])
        database_errors = list(record.get("database_errors") or [])
        if status not in {"healthy", "quarantine"}:
            raise RuntimeError("AWM scenario-health audit is not binary")
        if (status == "healthy") != (not reasons and not database_errors):
            raise RuntimeError("AWM scenario-health status disagrees with its evidence")
    candidate_ids = [str(value) for value in manifest.get("candidate_task_ids") or []]
    if [str(record["task_id"]) for record in task_records] != candidate_ids:
        raise RuntimeError("AWM task-health audit order mismatch")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("AWM healthy-pool candidates contain duplicate task IDs")
    for record in task_records:
        status = record.get("status")
        reasons = list(record.get("status_reasons") or [])
        if status not in {"healthy", "quarantine"}:
            raise RuntimeError("AWM task-health audit is not binary")
        if (status == "healthy") != (not reasons):
            raise RuntimeError("AWM task-health status disagrees with its evidence")
        if status == "healthy":
            noop = record.get("noop")
            if not isinstance(noop, Mapping) or noop.get("status") != "healthy":
                raise RuntimeError("AWM healthy task lacks a healthy no-action audit")
            if noop.get("label") not in _HEALTHY_NOOP_LABELS:
                raise RuntimeError("AWM healthy task has an invalid no-action label")
            if not record.get("sql_verifier_sha256"):
                raise RuntimeError("AWM healthy task lacks SQL-verifier identity")
    healthy_ids = [str(record["task_id"]) for record in task_records if record.get("status") == "healthy"]
    if manifest.get("training_pool_task_ids") != healthy_ids:
        raise RuntimeError("AWM healthy task IDs differ from manifest")
    frame = pd.read_parquet(data)
    extras = [dict(value) for value in frame["extra_info"].tolist()]
    if [str(value["task_id"]) for value in extras] != healthy_ids:
        raise RuntimeError("AWM healthy-pool parquet task IDs differ")
    if any(value.get("awm_health_pool_protocol_version") != HEALTH_POOL_PROTOCOL_VERSION or value.get("awm_health_status") != "healthy" for value in extras):
        raise RuntimeError("AWM healthy-pool row metadata mismatch")
    if any(not isinstance(value.get("awm_expert_one_off"), Mapping) for value in extras):
        raise RuntimeError("AWM healthy-pool expert metadata must be structured")
    if manifest.get("expert_outcome_membership_gate") is not False:
        raise RuntimeError("AWM expert outcomes must not gate healthy-pool membership")
    expected_counts = {
        "context_eligible": len(candidate_ids),
        "healthy": len(healthy_ids),
        "quarantine": len(candidate_ids) - len(healthy_ids),
        "healthy_environments": len({str(record["scenario"]) for record in task_records if record.get("status") == "healthy"}),
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("AWM healthy-pool manifest counts mismatch")
    return {"tasks": len(healthy_ids), "data": str(data), "kind": HEALTH_POOL_KIND}


async def build_healthy_pool(args) -> dict[str, Any]:
    rows, selection = _load_candidate_rows(args.data, args.candidate_manifest)
    source_hashes = {name: sha256_file(args.awm_data_dir / name) for name in EXPECTED_SOURCE_SHA256}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("AWM healthy pool requires the pinned source file hashes")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing required environment variable {args.api_key_env}")
    expert_metadata = load_expert_metadata(args.expert_trials)
    default_expert_metadata = {
        "available": False,
        "status": "not_screened",
        "legacy_status": "",
        "success": False,
        "reward_type": "",
        "model": "",
    }
    identity = {
        "protocol_version": HEALTH_POOL_PROTOCOL_VERSION,
        "kind": HEALTH_POOL_KIND,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": source_hashes,
        "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "selection_counts": selection["selected_counts"],
        "candidate_task_ids": [row["task_id"] for row in rows],
        "terminal_judge_protocol_version": TERMINAL_JUDGE_PROTOCOL_VERSION,
        "terminal_judge_model": args.model,
        "terminal_judge_api_base": args.api_base,
        "terminal_judge_reasoning_effort": "max",
        "terminal_judge_max_tokens": 8192,
        "noop_attempts": args.attempts,
        "policy": ("context eligible AND strict scenario database build AND valid SQL verifier AND no-op SQL+LLM outcome in {incomplete,agent_error}"),
        "expert_outcome_membership_gate": False,
        "expert_trials_sha256": (sha256_file(args.expert_trials) if args.expert_trials else None),
        "expert_metadata_tasks": len(expert_metadata),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("AWM healthy-pool resume configuration mismatch")
        manifest_path = args.output_dir / HEALTH_MANIFEST_FILENAME
        if manifest_path.is_file():
            verify_healthy_pool(args.output_dir / HEALTH_POOL_FILENAME, manifest_path)
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    scenario_order = list(dict.fromkeys(row["scenario"] for row in rows))
    scenario_records = audit_scenarios(args.awm_data_dir, scenario_order)
    scenario_path = args.output_dir / SCENARIO_HEALTH_FILENAME
    _write_jsonl(scenario_path, scenario_records)
    scenario_by_name = {record["scenario"]: record for record in scenario_records}

    verifier_index = _load_multimap(
        args.awm_data_dir / "gen_verifier.jsonl",
        lambda record: (
            _normalize_scenario(record["scenario"]),
            int(record["task_idx"]),
        ),
    )
    task_path = args.output_dir / TASK_HEALTH_FILENAME
    existing = {str(record["task_id"]): record for record in _load_jsonl(task_path, repair_torn_tail=True)}
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def audit_row(row: Mapping[str, Any]) -> None:
        task_id = str(row["task_id"])
        if task_id in existing:
            return
        reasons = []
        scenario_health = scenario_by_name[row["scenario"]]
        if scenario_health["status"] != "healthy":
            reasons.append("scenario_quarantine")
        key = (_normalize_scenario(row["scenario"]), int(row["task_idx"]))
        verifier_reasons, verifier_sha256 = audit_sql_verifier(
            row,
            verifier_index.get(key) or [],
        )
        reasons.extend(verifier_reasons)
        noop = None
        if not reasons:
            noop = await audit_noop(
                row,
                base_url=args.awm_base_url,
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                semaphore=semaphore,
                attempts=args.attempts,
            )
            if noop["status"] != "healthy":
                reasons.append(str(noop["status_reason"]))
        record = {
            "task_id": task_id,
            "scenario": row["scenario"],
            "task_idx": int(row["task_idx"]),
            "status": "healthy" if not reasons else "quarantine",
            "status_reasons": sorted(set(reasons)),
            "sql_verifier_sha256": verifier_sha256,
            "noop": noop,
            "expert_one_off": expert_metadata.get(task_id, default_expert_metadata),
        }
        async with write_lock:
            if task_id not in existing:
                existing[task_id] = record
                with task_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                if len(existing) % 50 == 0:
                    print(f"healthy_pool {len(existing)}/{len(rows)}", flush=True)

    await asyncio.gather(*(audit_row(row) for row in rows))
    if set(existing) != {row["task_id"] for row in rows}:
        raise RuntimeError("AWM healthy-pool audit contains unexpected task IDs")
    ordered = [existing[row["task_id"]] for row in rows]
    _write_jsonl(task_path, ordered)
    healthy_ids = [record["task_id"] for record in ordered if record["status"] == "healthy"]
    pool_path = args.output_dir / HEALTH_POOL_FILENAME
    pd.DataFrame(
        [
            _training_row(
                row,
                expert_metadata.get(row["task_id"], default_expert_metadata),
            )
            for row, record in zip(rows, ordered, strict=True)
            if record["status"] == "healthy"
        ]
    ).to_parquet(pool_path, index=False)
    counts = {
        "context_eligible": len(rows),
        "healthy": len(healthy_ids),
        "quarantine": len(rows) - len(healthy_ids),
        "healthy_environments": len({record["scenario"] for record in ordered if record["status"] == "healthy"}),
    }
    manifest = {
        **identity,
        "counts": counts,
        "training_pool_filename": HEALTH_POOL_FILENAME,
        "training_pool_task_ids": healthy_ids,
        "scenario_health_sha256": sha256_file(scenario_path),
        "task_health_sha256": sha256_file(task_path),
        "training_pool_data_sha256": sha256_file(pool_path),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_TERMINAL_JUDGE_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_TERMINAL_JUDGE_API_BASE)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--expert-trials", type=Path)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_healthy_pool(
            args.output_dir / HEALTH_POOL_FILENAME,
            args.output_dir / HEALTH_MANIFEST_FILENAME,
        )
        return
    if args.expert_trials is not None and not args.expert_trials.is_file():
        parser.error(f"expert trials do not exist: {args.expert_trials}")
    if args.concurrency <= 0 or args.attempts <= 0:
        parser.error("concurrency and attempts must be positive")
    manifest = asyncio.run(build_healthy_pool(args))
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
