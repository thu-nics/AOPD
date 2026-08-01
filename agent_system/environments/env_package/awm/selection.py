#!/usr/bin/env python3
"""Audit AWM scaffolds and select a deterministic 1K expert candidate pool."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from transformers import AutoTokenizer

from .actions import normalize_tools, tool_schema_hash
from .data import (
    DATASET_NAME,
    DATASET_REVISION,
    EXPECTED_SOURCE_SHA256,
)
from .native_rollout import (
    fixed_scaffold_token_count,
    model_artifact_identity,
    observation_dict,
    sha256_file,
)

SELECTION_PROTOCOL_VERSION = 1
EXPECTED_AUDIT_COUNTS = {
    "tasks": 10000,
    "eligible_tasks": 9834,
    "eligible_environments": 984,
    "all_tasks_eligible_environments": 983,
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
    eligible = [record for record in records if int(record["scaffold_tokens"]) <= cutoff]
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(str(record["scenario"]), []).append(record)
    return {
        "tasks": len(records),
        "eligible_tasks": len(eligible),
        "eligible_environments": sum(any(int(item["scaffold_tokens"]) <= cutoff for item in items) for items in by_scenario.values()),
        "all_tasks_eligible_environments": sum(all(int(item["scaffold_tokens"]) <= cutoff for item in items) for items in by_scenario.values()),
    }


def eligible_by_scenario(
    audit_records: list[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    cutoff: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for record in audit_records:
        if int(record["scaffold_tokens"]) > cutoff:
            continue
        task_id = str(record["task_id"])
        row = dict(rows_by_id[task_id])
        row["scaffold_tokens"] = int(record["scaffold_tokens"])
        row["tool_schema_hash"] = str(record["tool_schema_hash"])
        row["scaffold_reset_reward_type"] = record.get("reset_reward_type")
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
                raise RuntimeError(f"AWM scaffold audit reset failed for {scenario}: {reset_payload}")
            tools = normalize_tools(await env.list_tools(use_cache=False))
    schema_hash = tool_schema_hash(tools)
    return [
        {
            "task_id": row["task_id"],
            "scenario": scenario,
            "task_idx": int(row["task_idx"]),
            "scaffold_tokens": fixed_scaffold_token_count(tokenizer, row["task"], tools),
            "tool_schema_hash": schema_hash,
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
            tools = normalize_tools(await env.list_tools(use_cache=False))
            actual_hash = tool_schema_hash(tools)
            if actual_hash != row["tool_schema_hash"]:
                return {
                    "task_id": row["task_id"],
                    "viable": False,
                    "reason": "tool_schema_changed",
                    "expected_tool_schema_hash": row["tool_schema_hash"],
                    "actual_tool_schema_hash": actual_hash,
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
            "scaffold_tokens": int(row["scaffold_tokens"]),
            "tool_schema_hash": str(row["tool_schema_hash"]),
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
        "scaffold_cutoff": int(args.cutoff),
        "target_tasks": int(args.target),
        "selection": ("sha256 environment order; sha256 task order; one viable task per eligible environment, then deterministic environment round-robin"),
        "preflight": {
            "reset_and_list_tools": True,
            "tool_schema_hash_must_match": True,
            "untouched_code_verifier_must_not_complete": True,
            "reset_warning_is_allowed_and_recorded": True,
            "infrastructure_attempts": 3,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != identity:
            raise RuntimeError("AWM scaffold audit resume configuration mismatch")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    audit_path = args.output_dir / "scaffold_audit.jsonl"
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
            print(f"scaffold_audit {len(existing_audit)}/10000", flush=True)

    await asyncio.gather(*(audit_and_write(scenario) for scenario in pending_scenarios))
    if set(existing_audit) != set(rows_by_id):
        raise RuntimeError("AWM scaffold audit did not produce exactly one record per task")
    audit_records = [existing_audit[task_id] for task_id in base_ids]
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = audit_counts(audit_records, args.cutoff)
    if args.cutoff == 16000 and counts != EXPECTED_AUDIT_COUNTS:
        raise RuntimeError(f"AWM Qwen3 scaffold audit changed: expected {EXPECTED_AUDIT_COUNTS}, got {counts}")
    eligible = eligible_by_scenario(audit_records, rows_by_id, args.cutoff)
    rounds = selection_rounds(eligible)
    if len(rounds) < args.target:
        raise RuntimeError(f"only {len(rounds)} tasks satisfy the scaffold cutoff")

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
    first_rows = await asyncio.gather(*(first_viable(scenario) for scenario in scenario_order))
    selected = [row for row in first_rows if row is not None]
    selected_ids = {row["task_id"] for row in selected}
    scenario_counts = {row["scenario"]: 1 for row in selected}

    # Fill the small remainder with a second task per environment before any
    # third task, preserving deterministic environment balance.
    for _, _, row in rounds:
        if len(selected) >= args.target:
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
    if len(selected) != args.target:
        raise RuntimeError(f"preflight left {len(selected)} viable tasks, expected {args.target}")

    scenario_counts: dict[str, int] = {}
    for row in selected:
        scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
    if max(scenario_counts.values()) > 2:
        raise RuntimeError("1K candidate selection unexpectedly assigned more than two tasks to an environment")
    with preflight_path.open("w", encoding="utf-8") as handle:
        for record in sorted(preflight.values(), key=lambda item: str(item["task_id"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    selected_records = [
        {
            "task_id": row["task_id"],
            "scenario": row["scenario"],
            "task_idx": row["task_idx"],
            "scaffold_tokens": row["scaffold_tokens"],
            "tool_schema_hash": row["tool_schema_hash"],
            "preflight": preflight[row["task_id"]],
        }
        for row in selected
    ]
    candidate_path = args.output_dir / "awm_expert_candidates_1k.parquet"
    pd.DataFrame([_training_row(row) for row in selected]).to_parquet(candidate_path, index=False)
    manifest = {
        **identity,
        "kind": "awm_expert_candidate_selection",
        "audit_counts": counts,
        "selected_counts": {
            "tasks": len(selected),
            "environments": len(scenario_counts),
            "max_tasks_per_environment": max(scenario_counts.values()),
        },
        "scaffold_audit_sha256": sha256_file(audit_path),
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


def verify_selection(output_dir: Path) -> None:
    manifest_path = output_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM candidate manifest protocol mismatch")
    audit_path = output_dir / "scaffold_audit.jsonl"
    preflight_path = output_dir / "preflight.jsonl"
    candidate_path = output_dir / "awm_expert_candidates_1k.parquet"
    expected_hashes = {
        audit_path: manifest["scaffold_audit_sha256"],
        preflight_path: manifest["preflight_sha256"],
        candidate_path: manifest["candidate_data_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"AWM selection artifact hash mismatch: {path}")
    frame = pd.read_parquet(candidate_path)
    ids = [str(extra["task_id"]) for extra in frame["extra_info"].tolist()]
    if ids != manifest["task_ids"]:
        raise RuntimeError("AWM candidate parquet IDs do not match its manifest")
    if len(ids) != len(set(ids)) or len(ids) != int(manifest["target_tasks"]):
        raise RuntimeError("AWM candidate selection has duplicate or missing task IDs")
    records = list(manifest.get("records") or [])
    if [str(record.get("task_id")) for record in records] != ids:
        raise RuntimeError("AWM candidate manifest records do not match ordered task IDs")
    cutoff = int(manifest["scaffold_cutoff"])
    if any(int(record.get("scaffold_tokens", cutoff + 1)) > cutoff for record in records):
        raise RuntimeError("AWM candidate selection exceeds its scaffold cutoff")
    if any(not (record.get("preflight") or {}).get("viable") for record in records):
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
    print(json.dumps(manifest["selected_counts"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=16000)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_selection(args.output_dir)
        return
    for name in ("data", "manifest", "tokenizer"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.cutoff <= 0 or args.target <= 0 or args.concurrency <= 0:
        parser.error("--cutoff, --target, and --concurrency must be positive")
    asyncio.run(build_selection(args))


if __name__ == "__main__":
    main()
