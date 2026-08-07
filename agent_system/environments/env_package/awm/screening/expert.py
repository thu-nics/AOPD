#!/usr/bin/env python3
"""One-pass expert screening for AWM task/environment validity."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from transformers import AutoTokenizer

from ..data.integrity import (
    INTEGRITY_PROTOCOL_VERSION,
    PREFILTER_PROTOCOL_VERSION,
    TRAINING_POOL_PROTOCOL_VERSION,
    verify_integrity,
)
from ..runtime.logical_time import fetch_server_protocol
from ..runtime.rollout import (
    MODEL_CONTEXT_TOKENS,
    model_artifact_identity,
    run_native_trajectory,
    sha256_file,
)
from .common import DeepSeekExpertPolicy, environment_balanced, load_candidate_rows

EXPERT_SCREENING_PROTOCOL_VERSION = 2
SCREENING_SEED = 300
SCREENING_HISTORY_WINDOW = 6
SCREENING_MAX_DECISIONS = 20
FINAL_TASK_STATUSES = ("passed", "failed", "infrastructure_failed", "pending")
FINAL_POOL_FILENAME = "awm_training_pool.parquet"
FINAL_MANIFEST_FILENAME = "final_manifest.json"
POLICY_FAILURE_REWARD_TYPES = frozenset({"others", "incomplete", "agent_error"})
USAGE_FIELDS = (
    "requests",
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "total_tokens",
)


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
                continue
            raise RuntimeError(f"invalid expert-screening JSONL record {index + 1}") from exc
    if repair_torn_tail and lines and not lines[-1].endswith(("\n", "\r")):
        repair = True
    if repair:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def usage_from_result(result: Mapping[str, Any] | None) -> dict[str, int]:
    usage = {field: 0 for field in USAGE_FIELDS}
    for entry in (result or {}).get("trajectory") or []:
        usage["requests"] += 1
        entry_usage = entry.get("usage") or {}
        for field in USAGE_FIELDS[1:]:
            usage[field] += int(entry_usage.get(field, 0) or 0)
    return usage


def usage_from_trials(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    usage = {field: 0 for field in USAGE_FIELDS}
    for record in records:
        recorded = record.get("usage")
        if not isinstance(recorded, Mapping):
            recorded = usage_from_result(record.get("result") or record.get("last_result"))
        usage = add_usage(usage, recorded)
    return usage


def add_usage(*values: Mapping[str, Any] | None) -> dict[str, int]:
    usage = {field: 0 for field in USAGE_FIELDS}
    for value in values:
        for field in USAGE_FIELDS:
            usage[field] += int((value or {}).get(field, 0) or 0)
    return usage


def screening_rollout_protocol(max_response_tokens: int) -> dict[str, Any]:
    max_response_tokens = int(max_response_tokens)
    if not 0 < max_response_tokens < MODEL_CONTEXT_TOKENS:
        raise ValueError("screening response budget must be in (0, model_context_tokens)")
    return {
        "history_window": SCREENING_HISTORY_WINDOW,
        "history_unit": "complete_action_result_exchange",
        "history_prefix": "system_and_task_pinned",
        "model_context_tokens": MODEL_CONTEXT_TOKENS,
        "max_prompt_tokens": MODEL_CONTEXT_TOKENS - max_response_tokens,
        "context_response_reserve_tokens": max_response_tokens,
        "max_decisions": SCREENING_MAX_DECISIONS,
    }


def provider_identity_from_trials(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    identities = set()
    for record in records:
        result = record.get("result") or record.get("last_result") or {}
        for entry in result.get("trajectory") or []:
            if entry.get("model"):
                identities.add((str(entry["model"]), entry.get("system_fingerprint")))
    if len(identities) > 1:
        raise RuntimeError(f"expert screening mixes provider identities: {identities!r}")
    if not identities:
        return None
    model, fingerprint = next(iter(identities))
    return {"model": model, "system_fingerprint": fingerprint}


def validate_trial_records(records: Sequence[Mapping[str, Any]], candidate_ids: set[str]) -> None:
    seen = set()
    allowed = {"success", "failure", "infrastructure_exhausted"}
    for record in records:
        task_id = str(record.get("task_id") or "")
        if task_id not in candidate_ids:
            raise RuntimeError(f"expert-screening trial has unknown task ID {task_id!r}")
        if task_id in seen:
            raise RuntimeError(f"duplicate expert-screening trial for {task_id!r}")
        seen.add(task_id)
        if int(record.get("seed", -1)) != SCREENING_SEED:
            raise RuntimeError(f"expert-screening trial for {task_id!r} has a seed mismatch")
        status = str(record.get("status") or "")
        if status not in allowed:
            raise RuntimeError(f"expert-screening trial for {task_id!r} has invalid status {status!r}")
        result = record.get("result")
        if status in {"success", "failure"} and not isinstance(result, Mapping):
            raise RuntimeError(f"expert-screening trial for {task_id!r} is missing its result")
        if status == "success" and not result.get("success"):
            raise RuntimeError(f"expert-screening success for {task_id!r} has a failed result")
        if status == "failure" and result.get("success") and not record.get("legacy_status"):
            raise RuntimeError(f"expert-screening failure for {task_id!r} has a successful result")


def task_resolution(task_id: str, records_by_task: Mapping[str, Mapping[str, Any]]) -> str:
    record = records_by_task.get(task_id)
    if record is None:
        return "pending"
    return {
        "success": "passed",
        "failure": "failed",
        "infrastructure_exhausted": "infrastructure_failed",
    }[str(record["status"])]


def screening_result_status(result: Mapping[str, Any]) -> str:
    reward_type = str(result.get("reward_type") or "")
    if reward_type == "complete" and bool(result.get("success")):
        return "success"
    if bool(result.get("verify_infrastructure_error")) or any(bool(entry.get("runtime_infrastructure_error")) for entry in result.get("trajectory") or []):
        return "infrastructure_error"
    if reward_type in POLICY_FAILURE_REWARD_TYPES:
        return "failure"
    return "infrastructure_error"


def new_task_limit(
    total_tasks: int,
    *,
    max_new_tasks: int | None,
    max_new_task_fraction: float | None,
) -> int:
    if max_new_tasks is not None:
        return min(total_tasks, int(max_new_tasks))
    if max_new_task_fraction is not None:
        return min(total_tasks, max(1, math.ceil(total_tasks * float(max_new_task_fraction))))
    return total_tasks


def _final_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row["training_row"])
    extra = dict(output["extra_info"])
    extra.update(
        {
            "awm_expert_screening_protocol_version": EXPERT_SCREENING_PROTOCOL_VERSION,
            "awm_expert_screening_status": "passed",
            "awm_final_pool_status": "active",
        }
    )
    output["extra_info"] = extra
    return output


def write_screening_artifacts(
    *,
    rows: list[dict[str, Any]],
    candidate_manifest: Mapping[str, Any],
    integrity_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    integrity_manifest_path: Path,
    identity: Mapping[str, Any],
    trial_records: list[dict[str, Any]],
    output_dir: Path,
    provider_identity: Mapping[str, Any] | None,
    live_usage: Mapping[str, int],
    prior_cumulative_usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    candidate_snapshot = output_dir / "candidate_manifest.json"
    integrity_snapshot = output_dir / "integrity_manifest.json"
    candidate_snapshot.write_bytes(candidate_manifest_path.read_bytes())
    integrity_snapshot.write_bytes(integrity_manifest_path.read_bytes())
    candidate_ids = {row["task_id"] for row in rows}
    validate_trial_records(trial_records, candidate_ids)
    order = {row["task_id"]: index for index, row in enumerate(rows)}
    trial_records.sort(key=lambda record: order[str(record["task_id"])])
    trials_path = output_dir / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for record in trial_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    records_by_task = {str(record["task_id"]): record for record in trial_records}
    resolutions = {row["task_id"]: task_resolution(row["task_id"], records_by_task) for row in rows}
    accepted_rows = [row for row in rows if resolutions[row["task_id"]] == "passed"]
    pool_path = output_dir / FINAL_POOL_FILENAME
    if accepted_rows:
        pd.DataFrame([_final_training_row(row) for row in accepted_rows]).to_parquet(pool_path, index=False)
    elif pool_path.exists():
        pool_path.unlink()

    counts = {status: sum(value == status for value in resolutions.values()) for status in FINAL_TASK_STATUSES}
    trial_visible_usage = usage_from_trials(trial_records)
    cumulative_usage = add_usage(prior_cumulative_usage, live_usage)
    context_tasks = int(
        (integrity_manifest.get("selection_counts") or {}).get(
            "tasks",
            len(rows) + int((integrity_manifest.get("counts") or {}).get("quarantine", 0)),
        )
    )
    deterministic_quarantine = int((integrity_manifest.get("counts") or {}).get("quarantine", 0))
    manifest = {
        **identity,
        "kind": "awm_strict_task_pool",
        "selection_policy": "deterministic pass AND one-off expert success",
        "config_sha256": sha256_file(output_dir / "config.json"),
        "candidate_selection_counts": candidate_manifest.get("selected_counts"),
        "candidate_manifest_snapshot_sha256": sha256_file(candidate_snapshot),
        "integrity_manifest_snapshot_sha256": sha256_file(integrity_snapshot),
        "integrity_filter_counts": integrity_manifest.get("counts"),
        "provider_identity": dict(provider_identity) if provider_identity is not None else None,
        "live_usage": dict(live_usage),
        "cumulative_usage": cumulative_usage,
        "trial_visible_usage": trial_visible_usage,
        "counts": counts,
        "pipeline_counts": {
            "context_eligible": context_tasks,
            "deterministic_pass": len(rows),
            "deterministic_quarantine": deterministic_quarantine,
            "expert_pass": len(accepted_rows),
            "expert_failed": counts["failed"],
            "expert_infrastructure_failed": counts["infrastructure_failed"],
            "expert_reject": counts["failed"] + counts["infrastructure_failed"],
            "expert_pending": counts["pending"],
        },
        "task_status": resolutions,
        "accepted_task_ids": [row["task_id"] for row in accepted_rows],
        "rejected_task_ids": [row["task_id"] for row in rows if resolutions[row["task_id"]] in {"failed", "infrastructure_failed"}],
        "pending_task_ids": [row["task_id"] for row in rows if resolutions[row["task_id"]] == "pending"],
        "selection_counts": integrity_manifest.get("selection_counts"),
        "training_pool_task_ids": [row["task_id"] for row in accepted_rows],
        "training_pool_data_sha256": sha256_file(pool_path) if pool_path.is_file() else None,
        "training_pool_filename": pool_path.name,
        "trials_sha256": sha256_file(trials_path),
    }
    manifest_path = output_dir / FINAL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        **counts,
        "training_pool_tasks": len(accepted_rows),
        "complete": counts["pending"] == 0,
        "live_usage": dict(live_usage),
        "cumulative_usage": cumulative_usage,
        "trial_visible_usage": trial_visible_usage,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _screening_identity(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    integrity_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    integrity_manifest_path: Path,
    data_path: Path,
    tokenizer_identity: Mapping[str, Any],
    model: str,
    api_base: str,
    awm_base_url: str,
    awm_logical_time: Mapping[str, Any],
    max_tokens: int,
    infrastructure_attempts: int,
) -> dict[str, Any]:
    return {
        "protocol_version": EXPERT_SCREENING_PROTOCOL_VERSION,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "integrity_manifest_sha256": sha256_file(integrity_manifest_path),
        "candidate_data_sha256": sha256_file(data_path),
        "candidate_task_ids": [str(row["task_id"]) for row in rows],
        "candidate_scope": "deterministic_pass_only",
        "selection_protocol_version": candidate_manifest.get("protocol_version"),
        "integrity_protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
        "training_pool_protocol_version": TRAINING_POOL_PROTOCOL_VERSION,
        "screening_seed": SCREENING_SEED,
        "screening_order": "environment_balanced_sha256_task_id_within_environment",
        "screening_policy": "accept one-off expert success only; retry infrastructure errors without trajectory replay",
        "final_task_statuses": list(FINAL_TASK_STATUSES),
        "tokenizer": dict(tokenizer_identity),
        "model": model,
        "api_base": api_base,
        "awm_base_url": awm_base_url,
        "awm_logical_time": dict(awm_logical_time),
        **screening_rollout_protocol(max_tokens),
        "max_response_tokens": int(max_tokens),
        "thinking": True,
        "reasoning_effort": "max",
        "native_function_calling": True,
        "reasoning_history": "provider-native reasoning_content preserved per tool exchange",
        "parallel_tool_calls": False,
        "expert_multiple_calls": "execute_first",
        "verifier_mode": "code",
        "infrastructure_attempts": int(infrastructure_attempts),
    }


def _load_verified_candidates(args) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if args.integrity_manifest.name != "integrity_manifest.json":
        raise RuntimeError("expert screening requires the canonical integrity_manifest.json")
    verify_integrity(args.integrity_manifest.parent)
    rows, candidate_manifest, integrity_manifest = load_candidate_rows(
        args.data,
        args.candidate_manifest,
        args.integrity_manifest,
    )
    if integrity_manifest is None:
        raise RuntimeError("expert screening requires a hash-bound deterministic pool")
    if integrity_manifest.get("training_pool_protocol_version") != TRAINING_POOL_PROTOCOL_VERSION:
        raise RuntimeError("expert screening training-pool protocol mismatch")
    if sha256_file(args.data) != integrity_manifest.get("training_pool_data_sha256"):
        raise RuntimeError("expert screening requires the strict deterministic-pass pool")
    if integrity_manifest.get("counts", {}).get("pass") != len(rows):
        raise RuntimeError("expert screening candidate count differs from deterministic pass count")
    return rows, candidate_manifest, integrity_manifest


def _migrate_legacy_trial(record: Mapping[str, Any]) -> dict[str, Any]:
    status = str(record.get("status") or "")
    if status not in {"success", "policy_failure", "environment_failure", "infrastructure_exhausted"}:
        raise RuntimeError(f"unsupported legacy expert status {status!r}")
    output = {key: value for key, value in record.items() if key not in {"runtime_replay", "last_runtime_replay"}}
    for result_key in ("result", "last_result"):
        result = output.get(result_key)
        if not isinstance(result, Mapping):
            continue
        cleaned_result = dict(result)
        cleaned_result.pop("verify_observation_signature", None)
        cleaned_trajectory = []
        for entry in cleaned_result.get("trajectory") or []:
            cleaned_entry = dict(entry)
            cleaned_entry.pop("tool_observation_signature", None)
            cleaned_trajectory.append(cleaned_entry)
        cleaned_result["trajectory"] = cleaned_trajectory
        output[result_key] = cleaned_result
    if status == "success":
        output["status"] = "success"
    elif status == "infrastructure_exhausted":
        output["status"] = "infrastructure_exhausted"
    else:
        output["status"] = "failure"
    output["legacy_status"] = status
    return output


def migrate_screening(args) -> dict[str, Any]:
    rows, candidate_manifest, integrity_manifest = _load_verified_candidates(args)
    source_dir = args.migrate_from
    source_manifest_path = source_dir / "screening_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("protocol_version") != 1 or source_manifest.get("kind") != "awm_one_pass_expert_screening":
        raise RuntimeError("expert migration source must be a protocol-v1 one-pass screening")
    source_trials_path = source_dir / "trials.jsonl"
    if sha256_file(source_trials_path) != source_manifest.get("trials_sha256"):
        raise RuntimeError("legacy expert trials hash mismatch")
    source_config_path = source_dir / "config.json"
    if sha256_file(source_config_path) != source_manifest.get("config_sha256"):
        raise RuntimeError("legacy expert config hash mismatch")
    source_records = _load_jsonl(source_trials_path)
    source_by_id = {str(record.get("task_id")): record for record in source_records}
    candidate_ids = [row["task_id"] for row in rows]
    if not set(candidate_ids).issubset(source_by_id):
        raise RuntimeError("legacy expert trials do not cover every deterministic-pass task")
    migrated = [_migrate_legacy_trial(source_by_id[task_id]) for task_id in candidate_ids]

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    identity = _screening_identity(
        rows=rows,
        candidate_manifest=candidate_manifest,
        integrity_manifest=integrity_manifest,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
        data_path=args.data,
        tokenizer_identity=source_config["tokenizer"],
        model=str(source_config["model"]),
        api_base=str(source_config["api_base"]),
        awm_base_url=str(source_config["awm_base_url"]),
        awm_logical_time=source_config["awm_logical_time"],
        max_tokens=int(source_config["max_response_tokens"]),
        infrastructure_attempts=int(source_config["infrastructure_attempts"]),
    )
    identity["migration_provenance"] = {
        "kind": "protocol_v1_success_only_migration",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_protocol_version": 1,
        "api_calls": 0,
        "legacy_trials": len(source_records),
        "retained_trials": len(migrated),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = write_screening_artifacts(
        rows=rows,
        candidate_manifest=candidate_manifest,
        integrity_manifest=integrity_manifest,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
        identity=identity,
        trial_records=migrated,
        output_dir=args.output_dir,
        provider_identity=source_manifest.get("provider_identity"),
        live_usage={field: 0 for field in USAGE_FIELDS},
        prior_cumulative_usage=source_manifest.get("cumulative_usage"),
    )
    if summary["pending"]:
        raise RuntimeError("migrated strict expert screen is incomplete")
    return summary


async def screen(args) -> None:
    rows, candidate_manifest, integrity_manifest = _load_verified_candidates(args)
    identity = _screening_identity(
        rows=rows,
        candidate_manifest=candidate_manifest,
        integrity_manifest=integrity_manifest,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
        data_path=args.data,
        tokenizer_identity=model_artifact_identity(args.tokenizer),
        model=args.model,
        api_base=args.api_base,
        awm_base_url=args.awm_base_url,
        awm_logical_time=fetch_server_protocol(args.awm_base_url),
        max_tokens=args.max_tokens,
        infrastructure_attempts=args.infrastructure_attempts,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    trials_path = args.output_dir / "trials.jsonl"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("expert-screening resume configuration mismatch")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    prior_manifest_path = args.output_dir / FINAL_MANIFEST_FILENAME
    prior_manifest = None
    prior_cumulative_usage = None
    if prior_manifest_path.is_file():
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_cumulative_usage = prior_manifest.get("cumulative_usage")
    trial_records = _load_jsonl(trials_path, repair_torn_tail=True)
    if prior_manifest is not None:
        prior_status = prior_manifest.get("task_status") or {}
        recovered_records = [record for record in trial_records if prior_status.get(str(record["task_id"])) == "pending"]
        prior_cumulative_usage = add_usage(
            prior_cumulative_usage,
            usage_from_trials(recovered_records),
        )
    elif trial_records:
        prior_cumulative_usage = usage_from_trials(trial_records)
    candidate_ids = {row["task_id"] for row in rows}
    validate_trial_records(trial_records, candidate_ids)
    prior_identity = provider_identity_from_trials(trial_records)
    records_by_task = {str(record["task_id"]): record for record in trial_records}
    unresolved = [row for row in environment_balanced(rows) if row["task_id"] not in records_by_task]
    limit = new_task_limit(
        len(rows),
        max_new_tasks=args.max_new_tasks,
        max_new_task_fraction=args.max_new_task_fraction,
    )
    unresolved = unresolved[:limit]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    policy = DeepSeekExpertPolicy(
        model=args.model,
        api_key_env=args.api_key_env,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        expected_identity=prior_identity,
    )
    slots = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def record_trial(record: dict[str, Any]) -> None:
        async with write_lock:
            _append_jsonl(trials_path, record)
            trial_records.append(record)
            records_by_task[record["task_id"]] = record
            passed = sum(task_resolution(row["task_id"], records_by_task) == "passed" for row in rows)
            print(f"expert_screening_resolved {len(records_by_task)}/{len(rows)} passed={passed}", flush=True)

    async def screen_task(row: dict[str, Any]) -> None:
        async with slots:
            errors: list[str] = []
            last_result = None
            trial_usage = {field: 0 for field in USAGE_FIELDS}
            for infrastructure_attempt in range(1, args.infrastructure_attempts + 1):
                try:
                    result = await run_native_trajectory(
                        row,
                        generate_action=policy.generate,
                        tokenizer=tokenizer,
                        awm_base_url=args.awm_base_url,
                        seed=SCREENING_SEED,
                        verifier_mode="code",
                        max_decisions=SCREENING_MAX_DECISIONS,
                        history_window=SCREENING_HISTORY_WINDOW,
                        max_prompt_tokens=MODEL_CONTEXT_TOKENS - int(args.max_tokens),
                        preserve_reasoning_history=True,
                    )
                    last_result = result
                    trial_usage = add_usage(trial_usage, usage_from_result(result))
                    status = screening_result_status(result)
                    if status == "infrastructure_error":
                        errors.append(f"infrastructure reward_type={result.get('reward_type')!r}")
                        continue
                    await record_trial(
                        {
                            "task_id": row["task_id"],
                            "seed": SCREENING_SEED,
                            "status": status,
                            "infrastructure_attempts": infrastructure_attempt,
                            "infrastructure_errors": errors,
                            "usage": trial_usage,
                            "result": result,
                        }
                    )
                    return
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            await record_trial(
                {
                    "task_id": row["task_id"],
                    "seed": SCREENING_SEED,
                    "status": "infrastructure_exhausted",
                    "infrastructure_attempts": args.infrastructure_attempts,
                    "errors": errors,
                    "usage": trial_usage,
                    "last_result": last_result,
                }
            )

    try:
        await asyncio.gather(*(screen_task(row) for row in unresolved))
    finally:
        await policy.client.close()
    summary = write_screening_artifacts(
        rows=rows,
        candidate_manifest=candidate_manifest,
        integrity_manifest=integrity_manifest,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
        identity=identity,
        trial_records=trial_records,
        output_dir=args.output_dir,
        provider_identity=await policy.identity(),
        live_usage=await policy.stats(),
        prior_cumulative_usage=prior_cumulative_usage,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--integrity-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--infrastructure-attempts", type=int, default=3)
    parser.add_argument("--migrate-from", type=Path)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--max-new-tasks", type=int)
    limit.add_argument("--max-new-task-fraction", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.migrate_from is not None:
        summary = migrate_screening(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.tokenizer is None:
        parser.error("--tokenizer is required unless --migrate-from is used")
    positive = {
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "infrastructure_attempts": args.infrastructure_attempts,
    }
    if any(value <= 0 for value in positive.values()) or args.max_retries < 0:
        parser.error("concurrency, token/time/attempt limits must be positive; retries cannot be negative")
    if args.max_new_tasks is not None and args.max_new_tasks <= 0:
        parser.error("--max-new-tasks must be positive")
    if args.max_new_task_fraction is not None and not 0 < args.max_new_task_fraction <= 1:
        parser.error("--max-new-task-fraction must be in (0, 1]")
    if args.max_tokens >= MODEL_CONTEXT_TOKENS:
        parser.error("--max-tokens must be smaller than the model context budget")
    asyncio.run(screen(args))


if __name__ == "__main__":
    main()
