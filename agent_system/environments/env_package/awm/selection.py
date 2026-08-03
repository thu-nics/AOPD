#!/usr/bin/env python3
"""Audit native Qwen3 AWM prompts and select a deterministic environment pool."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from transformers import AutoTokenizer

from .actions import tool_schema_audit
from .data import (
    DATASET_NAME,
    DATASET_REVISION,
    EXPECTED_SOURCE_SHA256,
)
from .native_rollout import (
    fixed_native_prompt_token_count,
    model_artifact_identity,
    observation_dict,
    sha256_file,
)

SELECTION_PROTOCOL_VERSION = 5
SELECTION_MODE_ALL_ELIGIBLE = "all_context_eligible"
SELECTION_MODE_ONE_PER_ENVIRONMENT = "one_per_eligible_environment"
SELECTION_MODE_FIXED_TARGET = "fixed_target_round_robin"
CANDIDATE_FILENAME = "awm_context_candidates.parquet"
EXPECTED_NATIVE_PROMPT_AUDIT_COUNTS = {
    "tasks": 10000,
    "eligible_tasks": 9380,
    "eligible_environments": 938,
    "all_tasks_eligible_environments": 938,
}


def stable_rank(value: str) -> tuple[str, str]:
    return hashlib.sha256(value.encode()).hexdigest(), value


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
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
                "training_row": raw.to_dict(),
            }
        )
    if len(rows) != 10000 or len({row["task_id"] for row in rows}) != 10000:
        raise RuntimeError("AWM selection requires the complete unique 10,000-task parquet")
    return rows


def validate_base_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_NAME:
        raise RuntimeError("AWM base manifest dataset mismatch")
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("AWM base manifest revision mismatch")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("AWM base manifest source hashes mismatch")
    all_ids = list((manifest.get("split_task_ids") or {}).get("all") or [])
    if len(all_ids) != 10000:
        raise RuntimeError("AWM base manifest must contain the full all split")
    return manifest


def audit_counts(records: list[Mapping[str, Any]], cutoff: int) -> dict[str, int]:
    eligible = [record for record in records if int(record["native_prompt_tokens"]) <= cutoff]
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(str(record["scenario"]), []).append(record)
    return {
        "tasks": len(records),
        "eligible_tasks": len(eligible),
        "eligible_environments": sum(any(int(item["native_prompt_tokens"]) <= cutoff for item in items) for items in by_scenario.values()),
        "all_tasks_eligible_environments": sum(all(int(item["native_prompt_tokens"]) <= cutoff for item in items) for items in by_scenario.values()),
    }


def eligible_by_scenario(
    audit_records: list[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    cutoff: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for record in audit_records:
        if int(record["native_prompt_tokens"]) > cutoff:
            continue
        task_id = str(record["task_id"])
        row = dict(rows_by_id[task_id])
        row["native_prompt_tokens"] = int(record["native_prompt_tokens"])
        row["tool_schema_hash"] = str(record["tool_schema_hash"])
        row["raw_tool_schema_hash"] = str(record["raw_tool_schema_hash"])
        row["tool_schema_repair_count"] = int(record["tool_schema_repair_count"])
        row["native_prompt_reset_reward_type"] = record.get("reset_reward_type")
        output.setdefault(str(record["scenario"]), []).append(row)
    for scenario in output:
        output[scenario].sort(key=lambda row: stable_rank(str(row["task_id"])))
    return output


def selection_rounds(
    eligible: Mapping[str, list[dict[str, Any]]],
) -> list[tuple[str, int, dict[str, Any]]]:
    """Yield environment-balanced candidates in deterministic round-robin order."""
    scenarios = sorted(eligible, key=stable_rank)
    maximum = max((len(eligible[scenario]) for scenario in scenarios), default=0)
    return [(scenario, rank, eligible[scenario][rank]) for rank in range(maximum) for scenario in scenarios if rank < len(eligible[scenario])]


def one_per_environment(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first deterministically ordered selected task for each environment."""
    seen = set()
    output = []
    for record in records:
        scenario = str(record["scenario"])
        if scenario in seen:
            continue
        seen.add(scenario)
        output.append(dict(record))
    return output


def _selection_description(mode: str) -> str:
    if mode == SELECTION_MODE_ALL_ELIGIBLE:
        return "all tasks whose fixed Qwen3 native-tool prompt is at most the cutoff"
    if mode == SELECTION_MODE_ONE_PER_ENVIRONMENT:
        return "sha256 environment order; sha256 task order; exactly one viable task per eligible environment"
    if mode == SELECTION_MODE_FIXED_TARGET:
        return "sha256 environment order; sha256 task order; one viable task per eligible environment, then deterministic environment round-robin"
    raise ValueError(f"unknown AWM selection mode: {mode}")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl_by(path: Path, key) -> dict[Any, dict[str, Any]]:
    output = {}
    if not path.is_file():
        return output
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                output[key(record)] = record
    return output


async def _audit_scenario(
    scenario: str,
    scenario_rows: list[dict[str, Any]],
    *,
    tokenizer,
    awm_base_url: str,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    from agent_world_model_env import AWMEnv

    async with semaphore:
        async with AWMEnv(base_url=awm_base_url) as env:
            reset = await env.reset(scenario=scenario, task_idx=0, seed=0)
            reset_payload = observation_dict(reset)
            if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"AWM native prompt audit reset failed for {scenario}: {reset_payload}")
            schema_audit = tool_schema_audit(await env.list_tools(use_cache=False))
    tools = schema_audit["canonical_tools"]
    schema_hash = schema_audit["canonical_tool_schema_hash"]
    return [
        {
            "task_id": row["task_id"],
            "scenario": scenario,
            "task_idx": int(row["task_idx"]),
            "native_prompt_tokens": fixed_native_prompt_token_count(tokenizer, row["task"], tools),
            "tool_schema_hash": schema_hash,
            "raw_tool_schema_hash": schema_audit["raw_tool_schema_hash"],
            "tool_schema_repair_count": len(schema_audit["schema_repairs"]),
            "schema_repairs": schema_audit["schema_repairs"],
            "reset_reward_type": reset_payload.get("reward_type"),
            "reset_warning": (reset_payload if reset_payload.get("reward_type") == "reset_warning" else None),
        }
        for row in scenario_rows
    ]


async def _preflight_task_once(
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
                return {
                    "task_id": row["task_id"],
                    "viable": False,
                    "reason": "reset_failed",
                    "reset_payload": reset_payload,
                }
            reset_task = str(reset_payload.get("task") or "")
            if reset_task != str(row["task"]):
                return {
                    "task_id": row["task_id"],
                    "viable": False,
                    "reason": "reset_task_mismatch",
                    "expected_task": row["task"],
                    "actual_task": reset_task,
                }
            schema_audit = tool_schema_audit(await env.list_tools(use_cache=False))
            actual_hash = schema_audit["canonical_tool_schema_hash"]
            actual_raw_hash = schema_audit["raw_tool_schema_hash"]
            if actual_hash != row["tool_schema_hash"] or actual_raw_hash != row["raw_tool_schema_hash"]:
                return {
                    "task_id": row["task_id"],
                    "viable": False,
                    "reason": "tool_schema_changed",
                    "expected_tool_schema_hash": row["tool_schema_hash"],
                    "actual_tool_schema_hash": actual_hash,
                    "expected_raw_tool_schema_hash": row["raw_tool_schema_hash"],
                    "actual_raw_tool_schema_hash": actual_raw_hash,
                }
            verify = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": "code", "final_answer": None},
                )
            )
            verify_payload = observation_dict(verify)
            await env.step(CallToolAction(tool_name="done", arguments={}))
    no_op_complete = verify_payload.get("reward_type") == "complete"
    return {
        "task_id": row["task_id"],
        "viable": not no_op_complete,
        "reason": "no_op_complete" if no_op_complete else "ok",
        "reset_reward_type": reset_payload.get("reward_type"),
        "verify_reward_type": verify_payload.get("reward_type"),
        "verify_result": verify_payload.get("verify_result"),
    }


async def _preflight_task(
    row: Mapping[str, Any],
    *,
    awm_base_url: str,
    semaphore: asyncio.Semaphore,
    attempts: int = 3,
) -> dict[str, Any]:
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            result = await _preflight_task_once(
                row,
                awm_base_url=awm_base_url,
                semaphore=semaphore,
            )
            result["attempt"] = attempt
            return result
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "task_id": row["task_id"],
        "viable": False,
        "reason": "infrastructure_exhausted",
        "attempt": attempts,
        "errors": errors,
    }


def _training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row["training_row"])
    extra = dict(output["extra_info"])
    extra.update(
        {
            "native_prompt_tokens": int(row["native_prompt_tokens"]),
            "tool_schema_hash": str(row["tool_schema_hash"]),
            "raw_tool_schema_hash": str(row["raw_tool_schema_hash"]),
            "tool_schema_repair_count": int(row["tool_schema_repair_count"]),
            "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        }
    )
    output["extra_info"] = extra
    return output


async def build_selection(args) -> None:
    base_manifest = validate_base_manifest(args.manifest)
    rows = load_training_rows(args.data)
    rows_by_id = {row["task_id"]: row for row in rows}
    base_ids = list(base_manifest["split_task_ids"]["all"])
    if set(base_ids) != set(rows_by_id):
        raise RuntimeError("AWM all parquet and base manifest task IDs differ")

    identity = {
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "base_manifest_sha256": sha256_file(args.manifest),
        "base_data_sha256": sha256_file(args.data),
        "tokenizer": model_artifact_identity(args.tokenizer),
        "awm_base_url": args.awm_base_url,
        "native_prompt_cutoff": int(args.cutoff),
        "selection_mode": args.selection_mode,
        "requested_target_tasks": (int(args.target) if args.selection_mode == SELECTION_MODE_FIXED_TARGET else None),
        "tool_schema_policy": "canonicalize_redundant_nullable_sibling_type",
        "selection": _selection_description(args.selection_mode),
        "preflight": {
            "enabled": args.selection_mode != SELECTION_MODE_ALL_ELIGIBLE,
            "delegated_to_deterministic_filter": args.selection_mode == SELECTION_MODE_ALL_ELIGIBLE,
            "reset_and_fetch_native_tool_schemas": args.selection_mode != SELECTION_MODE_ALL_ELIGIBLE,
            "untouched_code_verifier_must_not_complete": args.selection_mode != SELECTION_MODE_ALL_ELIGIBLE,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != identity:
            raise RuntimeError("AWM native prompt audit resume configuration mismatch")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    audit_path = args.output_dir / "native_prompt_audit.jsonl"
    existing_audit = _load_jsonl_by(audit_path, lambda record: str(record["task_id"]))
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    pending_scenarios = [scenario for scenario in sorted(by_scenario, key=stable_rank) if not all(row["task_id"] in existing_audit for row in by_scenario[scenario])]
    write_lock = asyncio.Lock()

    async def audit_and_write(scenario):
        records = await _audit_scenario(
            scenario,
            by_scenario[scenario],
            tokenizer=tokenizer,
            awm_base_url=args.awm_base_url,
            semaphore=semaphore,
        )
        async with write_lock:
            for record in records:
                _append_jsonl(audit_path, record)
                existing_audit[record["task_id"]] = record
            print(f"native_prompt_audit {len(existing_audit)}/10000", flush=True)

    await asyncio.gather(*(audit_and_write(scenario) for scenario in pending_scenarios))
    if set(existing_audit) != set(rows_by_id):
        raise RuntimeError("AWM native prompt audit did not produce exactly one record per task")
    audit_records = [existing_audit[task_id] for task_id in base_ids]
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = audit_counts(audit_records, args.cutoff)
    schema_by_scenario = {str(record["scenario"]): record for record in audit_records}
    schema_counts = {
        "environments": len(schema_by_scenario),
        "environments_with_repairs": sum(int(record["tool_schema_repair_count"]) > 0 for record in schema_by_scenario.values()),
        "repairs": sum(int(record["tool_schema_repair_count"]) for record in schema_by_scenario.values()),
    }
    audit_summary = {
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "audit_counts": counts,
        "schema_counts": schema_counts,
        "native_prompt_audit_sha256": sha256_file(audit_path),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.audit_only:
        print(json.dumps(audit_summary, indent=2, sort_keys=True))
        return
    if args.cutoff == 16000 and counts != EXPECTED_NATIVE_PROMPT_AUDIT_COUNTS:
        raise RuntimeError(f"AWM Qwen3 native prompt audit changed: expected {EXPECTED_NATIVE_PROMPT_AUDIT_COUNTS}, got {counts}")
    eligible = eligible_by_scenario(audit_records, rows_by_id, args.cutoff)
    rounds = selection_rounds(eligible)
    if args.selection_mode == SELECTION_MODE_FIXED_TARGET and len(rounds) < args.target:
        raise RuntimeError(f"only {len(rounds)} tasks satisfy the native prompt cutoff")

    preflight_path = args.output_dir / "preflight.jsonl"
    preflight = _load_jsonl_by(preflight_path, lambda record: str(record["task_id"]))

    async def ensure_preflight(row):
        task_id = str(row["task_id"])
        if task_id not in preflight:
            result = await _preflight_task(
                row,
                awm_base_url=args.awm_base_url,
                semaphore=semaphore,
            )
            async with write_lock:
                if task_id not in preflight:
                    preflight[task_id] = result
                    _append_jsonl(preflight_path, result)
        return preflight[task_id]

    async def first_viable(scenario):
        for row in eligible[scenario]:
            if (await ensure_preflight(row)).get("viable"):
                return row
        return None

    scenario_order = sorted(eligible, key=stable_rank)
    if args.selection_mode == SELECTION_MODE_ALL_ELIGIBLE:
        selected = [row for _, _, row in rounds]
        target_tasks = len(rounds)
    else:
        first_rows = await asyncio.gather(*(first_viable(scenario) for scenario in scenario_order))
        selected = [row for row in first_rows if row is not None]
        if args.selection_mode == SELECTION_MODE_ONE_PER_ENVIRONMENT:
            missing_scenarios = [scenario for scenario, row in zip(scenario_order, first_rows, strict=True) if row is None]
            if missing_scenarios:
                raise RuntimeError("one-task-per-environment preflight found no viable task for: " + ", ".join(missing_scenarios))
            target_tasks = len(eligible)
        else:
            target_tasks = int(args.target)
    selected_ids = {row["task_id"] for row in selected}
    scenario_counts = {row["scenario"]: 1 for row in selected}

    if args.selection_mode == SELECTION_MODE_FIXED_TARGET:
        # Fill the remainder with a second task per environment before any
        # third task, preserving deterministic environment balance.
        for _, _, row in rounds:
            if len(selected) >= target_tasks:
                break
            if row["task_id"] in selected_ids:
                continue
            if scenario_counts.get(row["scenario"], 0) >= 2:
                continue
            result = await ensure_preflight(row)
            if result.get("viable"):
                selected.append(row)
                selected_ids.add(row["task_id"])
                scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
    if len(selected) != target_tasks:
        raise RuntimeError(f"preflight left {len(selected)} viable tasks, expected {target_tasks}")

    scenario_counts: dict[str, int] = {}
    for row in selected:
        scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
    expected_max_tasks = 10 if args.selection_mode == SELECTION_MODE_ALL_ELIGIBLE else (1 if args.selection_mode == SELECTION_MODE_ONE_PER_ENVIRONMENT else 2)
    if max(scenario_counts.values()) > expected_max_tasks:
        raise RuntimeError("AWM candidate selection assigned too many tasks to an environment")
    with preflight_path.open("w", encoding="utf-8") as handle:
        for record in sorted(preflight.values(), key=lambda item: str(item["task_id"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    selected_records = [
        {
            "task_id": row["task_id"],
            "scenario": row["scenario"],
            "task_idx": row["task_idx"],
            "native_prompt_tokens": row["native_prompt_tokens"],
            "tool_schema_hash": row["tool_schema_hash"],
            "raw_tool_schema_hash": row["raw_tool_schema_hash"],
            "tool_schema_repair_count": row["tool_schema_repair_count"],
            "preflight": preflight.get(row["task_id"]),
        }
        for row in selected
    ]
    candidate_path = args.output_dir / CANDIDATE_FILENAME
    pd.DataFrame([_training_row(row) for row in selected]).to_parquet(candidate_path, index=False)
    manifest = {
        **identity,
        "kind": "awm_context_candidate_selection",
        "target_tasks": len(selected),
        "audit_counts": counts,
        "schema_counts": schema_counts,
        "selected_counts": {
            "tasks": len(selected),
            "environments": len(scenario_counts),
            "max_tasks_per_environment": max(scenario_counts.values()),
        },
        "native_prompt_audit_sha256": sha256_file(audit_path),
        "audit_summary_sha256": sha256_file(args.output_dir / "audit_summary.json"),
        "preflight_sha256": sha256_file(preflight_path),
        "candidate_data_sha256": sha256_file(candidate_path),
        "task_ids": [row["task_id"] for row in selected],
        "records": selected_records,
    }
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selected_counts"], indent=2, sort_keys=True))


def _selection_candidate_path(output_dir: Path) -> Path:
    current = output_dir / CANDIDATE_FILENAME
    if current.is_file():
        return current
    legacy = output_dir / "awm_expert_candidates_1k.parquet"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"missing AWM candidate parquet under {output_dir}")


def rebase_one_per_environment(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Subset a verified legacy round-robin selection without environment calls."""
    source_manifest_path = source_dir / "candidate_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("protocol_version") not in {3, 4, SELECTION_PROTOCOL_VERSION}:
        raise RuntimeError("unsupported AWM selection source protocol")
    source_candidate_path = _selection_candidate_path(source_dir)
    source_paths = {
        "native_prompt_audit_sha256": source_dir / "native_prompt_audit.jsonl",
        "audit_summary_sha256": source_dir / "audit_summary.json",
        "preflight_sha256": source_dir / "preflight.jsonl",
        "candidate_data_sha256": source_candidate_path,
    }
    for field, path in source_paths.items():
        if sha256_file(path) != source_manifest.get(field):
            raise RuntimeError(f"AWM source selection artifact hash mismatch: {path}")

    source_frame = pd.read_parquet(source_candidate_path)
    source_ids = [str(extra["task_id"]) for extra in source_frame["extra_info"]]
    if source_ids != source_manifest.get("task_ids"):
        raise RuntimeError("AWM source selection parquet IDs differ from its manifest")
    selected_records = one_per_environment(list(source_manifest.get("records") or []))
    selected_ids = [str(record["task_id"]) for record in selected_records]
    expected_environments = int(source_manifest["audit_counts"]["eligible_environments"])
    if len(selected_ids) != expected_environments:
        raise RuntimeError("AWM one-per-environment rebase does not cover every eligible environment")
    if any(not (record.get("preflight") or {}).get("viable") for record in selected_records):
        raise RuntimeError("AWM one-per-environment rebase retained a failed preflight")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("native_prompt_audit.jsonl", "preflight.jsonl"):
        shutil.copy2(source_dir / filename, output_dir / filename)

    audit_summary = json.loads((source_dir / "audit_summary.json").read_text(encoding="utf-8"))
    audit_summary["protocol_version"] = SELECTION_PROTOCOL_VERSION
    (output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    frame_by_id = {str(row["extra_info"]["task_id"]): row for row in source_frame.to_dict(orient="records")}
    selected_rows = []
    for task_id in selected_ids:
        row = dict(frame_by_id[task_id])
        extra = dict(row["extra_info"])
        extra["selection_protocol_version"] = SELECTION_PROTOCOL_VERSION
        row["extra_info"] = extra
        selected_rows.append(row)
    selected_frame = pd.DataFrame(selected_rows)
    candidate_path = output_dir / CANDIDATE_FILENAME
    selected_frame.to_parquet(candidate_path, index=False)

    identity_fields = (
        "dataset",
        "dataset_revision",
        "source_sha256",
        "base_manifest_sha256",
        "base_data_sha256",
        "tokenizer",
        "awm_base_url",
        "native_prompt_cutoff",
        "tool_schema_policy",
        "preflight",
    )
    identity = {field: source_manifest[field] for field in identity_fields}
    identity.update(
        {
            "protocol_version": SELECTION_PROTOCOL_VERSION,
            "selection_mode": SELECTION_MODE_ONE_PER_ENVIRONMENT,
            "requested_target_tasks": None,
            "selection": _selection_description(SELECTION_MODE_ONE_PER_ENVIRONMENT),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_scenarios = {str(record["scenario"]) for record in selected_records}
    removed_ids = [task_id for task_id in source_ids if task_id not in set(selected_ids)]
    manifest = {
        **identity,
        "kind": "awm_expert_candidate_selection",
        "target_tasks": len(selected_ids),
        "audit_counts": source_manifest["audit_counts"],
        "schema_counts": source_manifest["schema_counts"],
        "selected_counts": {
            "tasks": len(selected_ids),
            "environments": len(selected_scenarios),
            "max_tasks_per_environment": 1,
        },
        "native_prompt_audit_sha256": sha256_file(output_dir / "native_prompt_audit.jsonl"),
        "audit_summary_sha256": sha256_file(output_dir / "audit_summary.json"),
        "preflight_sha256": sha256_file(output_dir / "preflight.jsonl"),
        "candidate_data_sha256": sha256_file(candidate_path),
        "task_ids": selected_ids,
        "records": selected_records,
        "migration_provenance": {
            "kind": "one_per_eligible_environment_rebase",
            "source_protocol_version": source_manifest["protocol_version"],
            "source_candidate_manifest_sha256": sha256_file(source_manifest_path),
            "source_tasks": len(source_ids),
            "target_tasks": len(selected_ids),
            "removed_task_ids": removed_ids,
            "api_calls": 0,
        },
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_selection(output_dir)
    return manifest["migration_provenance"]


def verify_selection(output_dir: Path) -> None:
    manifest_path = output_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM candidate manifest protocol mismatch")
    audit_path = output_dir / "native_prompt_audit.jsonl"
    preflight_path = output_dir / "preflight.jsonl"
    candidate_path = output_dir / CANDIDATE_FILENAME
    audit_summary_path = output_dir / "audit_summary.json"
    expected_hashes = {
        audit_path: manifest["native_prompt_audit_sha256"],
        audit_summary_path: manifest["audit_summary_sha256"],
        preflight_path: manifest["preflight_sha256"],
        candidate_path: manifest["candidate_data_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"AWM selection artifact hash mismatch: {path}")
    frame = pd.read_parquet(candidate_path)
    extras = [dict(extra) for extra in frame["extra_info"].tolist()]
    ids = [str(extra["task_id"]) for extra in extras]
    if ids != manifest["task_ids"]:
        raise RuntimeError("AWM candidate parquet IDs do not match its manifest")
    if any(extra.get("selection_protocol_version") != SELECTION_PROTOCOL_VERSION for extra in extras):
        raise RuntimeError("AWM candidate rows do not match the selection protocol")
    if len(ids) != len(set(ids)) or len(ids) != int(manifest["target_tasks"]):
        raise RuntimeError("AWM candidate selection has duplicate or missing task IDs")
    records = list(manifest.get("records") or [])
    if [str(record.get("task_id")) for record in records] != ids:
        raise RuntimeError("AWM candidate manifest records do not match ordered task IDs")
    cutoff = int(manifest["native_prompt_cutoff"])
    if any(int(record.get("native_prompt_tokens", cutoff + 1)) > cutoff for record in records):
        raise RuntimeError("AWM candidate selection exceeds its native prompt cutoff")
    selection_mode = manifest.get("selection_mode")
    if selection_mode == SELECTION_MODE_ALL_ELIGIBLE and any(record.get("preflight") is not None for record in records):
        raise RuntimeError("AWM all-context selection must delegate task preflight")
    if selection_mode != SELECTION_MODE_ALL_ELIGIBLE and any(not (record.get("preflight") or {}).get("viable") for record in records):
        raise RuntimeError("AWM candidate selection contains a failed preflight")
    scenario_counts: dict[str, int] = {}
    for record in records:
        scenario = str(record["scenario"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    actual_counts = {
        "tasks": len(ids),
        "environments": len(scenario_counts),
        "max_tasks_per_environment": max(scenario_counts.values()),
    }
    if actual_counts != manifest.get("selected_counts"):
        raise RuntimeError("AWM candidate selected counts do not match its records")
    if selection_mode == SELECTION_MODE_ALL_ELIGIBLE:
        expected_tasks = int(manifest["audit_counts"]["eligible_tasks"])
        expected_environments = int(manifest["audit_counts"]["eligible_environments"])
        if actual_counts != {
            "tasks": expected_tasks,
            "environments": expected_environments,
            "max_tasks_per_environment": 10,
        }:
            raise RuntimeError("AWM all-context-eligible selection does not exactly cover the audited pool")
    if selection_mode == SELECTION_MODE_ONE_PER_ENVIRONMENT:
        expected_environments = int(manifest["audit_counts"]["eligible_environments"])
        if actual_counts != {
            "tasks": expected_environments,
            "environments": expected_environments,
            "max_tasks_per_environment": 1,
        }:
            raise RuntimeError("AWM one-task-per-environment selection does not exactly cover the eligible environments")
    print(json.dumps(manifest["selected_counts"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=16000)
    parser.add_argument(
        "--selection-mode",
        choices=(
            SELECTION_MODE_ONE_PER_ENVIRONMENT,
            SELECTION_MODE_ALL_ELIGIBLE,
            SELECTION_MODE_FIXED_TARGET,
        ),
        default=SELECTION_MODE_ALL_ELIGIBLE,
    )
    parser.add_argument("--target", type=int)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rebase-from", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        verify_selection(args.output_dir)
        return
    if args.rebase_from is not None:
        result = rebase_one_per_environment(args.rebase_from, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    for name in ("data", "manifest", "tokenizer"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.cutoff <= 0 or args.concurrency <= 0:
        parser.error("--cutoff and --concurrency must be positive")
    if args.selection_mode == SELECTION_MODE_FIXED_TARGET:
        if args.target is None or args.target <= 0:
            parser.error("--target must be positive in fixed-target mode")
    elif args.target is not None:
        parser.error("--target is only valid in fixed-target mode")
    asyncio.run(build_selection(args))


if __name__ == "__main__":
    main()
