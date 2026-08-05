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

from .actions import AWMAction, tool_schema_audit, tool_schema_hash
from .integrity import (
    INTEGRITY_PROTOCOL_VERSION,
    PREFILTER_PROTOCOL_VERSION,
    TRAINING_POOL_PROTOCOL_VERSION,
)
from .logical_time import fetch_server_protocol
from .native_rollout import (
    MODEL_CONTEXT_TOKENS,
    model_artifact_identity,
    observation_dict,
    run_native_trajectory,
    sha256_file,
)
from .qualification import DeepSeekExpertPolicy, environment_balanced, load_candidate_rows
from .runtime_failures import (
    deterministic_error_signature,
    infrastructure_error,
    replay_observation_signature,
)

EXPERT_SCREENING_PROTOCOL_VERSION = 1
SCREENING_SEED = 300
SCREENING_HISTORY_WINDOW = 6
SCREENING_MAX_DECISIONS = 20
FINAL_TASK_STATUSES = (
    "accepted_success",
    "accepted_policy_failure",
    "rejected_environment",
    "infrastructure_pending",
    "pending",
)
ACCEPTED_TASK_STATUSES = frozenset({"accepted_success", "accepted_policy_failure"})
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                continue
            raise RuntimeError(f"invalid expert-screening JSONL record {index + 1}") from exc
    return records


def usage_from_trials(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    usage = {field: 0 for field in USAGE_FIELDS}
    for record in records:
        result = record.get("result") or record.get("last_result") or {}
        for entry in result.get("trajectory") or []:
            usage["requests"] += 1
            entry_usage = entry.get("usage") or {}
            for field in USAGE_FIELDS[1:]:
                usage[field] += int(entry_usage.get(field, 0) or 0)
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
    allowed = {"success", "policy_failure", "environment_failure", "infrastructure_exhausted"}
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
        if status in {"success", "policy_failure", "environment_failure"} and not isinstance(result, Mapping):
            raise RuntimeError(f"expert-screening trial for {task_id!r} is missing its result")
        if status == "success" and not result.get("success"):
            raise RuntimeError(f"expert-screening success for {task_id!r} has a failed result")
        if status == "policy_failure" and result.get("success"):
            raise RuntimeError(f"expert-screening policy failure for {task_id!r} has a successful result")
        if status == "environment_failure" and (record.get("runtime_replay") or {}).get("status") != "confirmed":
            raise RuntimeError(f"expert-screening environment failure for {task_id!r} lacks confirmed replay")


def task_resolution(task_id: str, records_by_task: Mapping[str, Mapping[str, Any]]) -> str:
    record = records_by_task.get(task_id)
    if record is None:
        return "pending"
    return {
        "success": "accepted_success",
        "policy_failure": "accepted_policy_failure",
        "environment_failure": "rejected_environment",
        "infrastructure_exhausted": "infrastructure_pending",
    }[str(record["status"])]


def screening_result_status(result: Mapping[str, Any]) -> str:
    reward_type = str(result.get("reward_type") or "")
    if reward_type == "complete" and bool(result.get("success")):
        return "success"
    if reward_type in POLICY_FAILURE_REWARD_TYPES:
        return "policy_failure"
    return "infrastructure_error"


def _parsed_tool_action(entry: Mapping[str, Any]) -> AWMAction:
    payload = json.loads(str(entry["parsed_action"]))
    if payload.get("kind") != "tool":
        raise ValueError("trajectory replay expected a tool action")
    return AWMAction(
        kind="tool",
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
    )


def runtime_failure_candidate(result: Mapping[str, Any]) -> dict[str, Any] | None:
    for index, entry in enumerate(result.get("trajectory") or []):
        if entry.get("runtime_infrastructure_error"):
            return {
                "phase": "tool",
                "trajectory_index": index,
                "expected_signature": entry.get("runtime_error_signature"),
            }
    if result.get("verify_infrastructure_error"):
        return {
            "phase": "verify",
            "trajectory_index": None,
            "expected_signature": result.get("verify_error_signature"),
        }
    return None


async def replay_runtime_failure(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    awm_base_url: str,
) -> dict[str, Any]:
    """Replay one strong failure without model calls; only exact repeats confirm defects."""
    candidate = runtime_failure_candidate(result)
    if candidate is None:
        return {"status": "none"}

    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    try:
        async with AWMEnv(base_url=awm_base_url) as env:
            reset = await env.reset(
                scenario=str(row["scenario"]),
                task_idx=int(row["task_idx"]),
                seed=SCREENING_SEED,
            )
            reset_payload = observation_dict(reset)
            if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                return {"status": "pending", "error": f"replay reset failed: {reset_payload}"}
            if str(reset_payload.get("task") or "") != str(row["task"]):
                return {"status": "pending", "error": "replay task changed"}
            schema = tool_schema_audit(await env.list_tools(use_cache=False))
            if tool_schema_hash(schema["canonical_tools"]) != str(result["tool_schema_hash"]):
                return {"status": "pending", "error": "replay canonical tool schema changed"}
            if str(schema["raw_tool_schema_hash"]) != str(result["raw_tool_schema_hash"]):
                return {"status": "pending", "error": "replay raw tool schema changed"}

            for index, entry in enumerate(result.get("trajectory") or []):
                if entry.get("action_kind") != "tool":
                    continue
                action = _parsed_tool_action(entry)
                replay_step = await env.step(
                    CallToolAction(
                        tool_name=action.name or "",
                        arguments=action.arguments or {},
                    )
                )
                payload = observation_dict(replay_step)
                if candidate["phase"] == "tool" and index == candidate["trajectory_index"]:
                    signature = deterministic_error_signature(
                        payload,
                        phase="tool",
                        tool_name=action.name,
                    )
                    if candidate["expected_signature"] is not None and signature == candidate["expected_signature"]:
                        return {"status": "confirmed", "phase": "tool", "signature": signature}
                    if not infrastructure_error(payload, phase="tool"):
                        return {"status": "resolved", "phase": "tool"}
                    return {
                        "status": "pending",
                        "phase": "tool",
                        "signature": signature,
                        "error": ("runtime tool error lacks a deterministic confirmation signature" if candidate["expected_signature"] is None else "runtime tool error changed during replay"),
                    }
                if replay_observation_signature(payload) != entry.get("tool_observation_signature"):
                    return {
                        "status": "pending",
                        "phase": candidate["phase"],
                        "error": "replay prefix observation changed",
                    }

            if candidate["phase"] != "verify":
                return {"status": "pending", "phase": "tool", "error": "failing tool action was not replayed"}
            verify = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": "code", "final_answer": result.get("final_answer")},
                )
            )
            payload = observation_dict(verify)
            signature = deterministic_error_signature(payload, phase="verify", tool_name="verify")
            if candidate["expected_signature"] is not None and signature == candidate["expected_signature"]:
                return {"status": "confirmed", "phase": "verify", "signature": signature}
            if not infrastructure_error(payload, phase="verify"):
                return {"status": "resolved", "phase": "verify"}
            return {
                "status": "pending",
                "phase": "verify",
                "signature": signature,
                "error": ("runtime verifier error lacks a deterministic confirmation signature" if candidate["expected_signature"] is None else "runtime verifier error changed during replay"),
            }
    except Exception as exc:
        return {"status": "pending", "phase": candidate["phase"], "error": f"{type(exc).__name__}: {exc}"}


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


def write_screening_artifacts(
    *,
    rows: list[dict[str, Any]],
    candidate_manifest: Mapping[str, Any],
    integrity_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    trial_records: list[dict[str, Any]],
    output_dir: Path,
    provider_identity: Mapping[str, Any] | None,
    live_usage: Mapping[str, int],
    prior_cumulative_usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
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
    accepted_rows = [row for row in rows if resolutions[row["task_id"]] in ACCEPTED_TASK_STATUSES]
    pool_path = output_dir / "awm_expert_screened_pool.parquet"
    if accepted_rows:
        pd.DataFrame([row["training_row"] for row in accepted_rows]).to_parquet(pool_path, index=False)
    elif pool_path.exists():
        pool_path.unlink()

    counts = {status: sum(value == status for value in resolutions.values()) for status in FINAL_TASK_STATUSES}
    trial_visible_usage = usage_from_trials(trial_records)
    cumulative_usage = add_usage(prior_cumulative_usage, live_usage)
    manifest = {
        **identity,
        "kind": "awm_one_pass_expert_screening",
        "config_sha256": sha256_file(output_dir / "config.json"),
        "candidate_selection_counts": candidate_manifest.get("selected_counts"),
        "integrity_filter_counts": integrity_manifest.get("counts"),
        "provider_identity": dict(provider_identity) if provider_identity is not None else None,
        "live_usage": dict(live_usage),
        "cumulative_usage": cumulative_usage,
        "trial_visible_usage": trial_visible_usage,
        "counts": counts,
        "task_status": resolutions,
        "accepted_task_ids": [row["task_id"] for row in accepted_rows],
        "selection_counts": integrity_manifest.get("selection_counts"),
        "training_pool_task_ids": [row["task_id"] for row in accepted_rows],
        "training_pool_data_sha256": sha256_file(pool_path) if pool_path.is_file() else None,
        "training_pool_filename": pool_path.name,
        "rejected_environment_task_ids": [task_id for task_id, status in resolutions.items() if status == "rejected_environment"],
        "infrastructure_pending_task_ids": [task_id for task_id, status in resolutions.items() if status == "infrastructure_pending"],
        "trials_sha256": sha256_file(trials_path),
        "screened_pool_sha256": sha256_file(pool_path) if pool_path.is_file() else None,
    }
    manifest_path = output_dir / "screening_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        **counts,
        "screened_pool_tasks": len(accepted_rows),
        "live_usage": dict(live_usage),
        "cumulative_usage": cumulative_usage,
        "trial_visible_usage": trial_visible_usage,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


async def screen(args) -> None:
    rows, candidate_manifest, integrity_manifest = load_candidate_rows(
        args.data,
        args.candidate_manifest,
        args.integrity_manifest,
    )
    if integrity_manifest is None:
        raise RuntimeError("expert screening requires a hash-bound deterministic training pool")
    if integrity_manifest.get("training_pool_protocol_version") != TRAINING_POOL_PROTOCOL_VERSION:
        raise RuntimeError("expert screening training-pool protocol mismatch")
    if sha256_file(args.data) != integrity_manifest.get("training_pool_data_sha256"):
        raise RuntimeError("expert screening requires the current deterministic training pool")

    identity = {
        "protocol_version": EXPERT_SCREENING_PROTOCOL_VERSION,
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "integrity_manifest_sha256": sha256_file(args.integrity_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "candidate_task_ids": [row["task_id"] for row in rows],
        "candidate_scope": "deterministic_filter_non_quarantine",
        "selection_protocol_version": candidate_manifest.get("protocol_version"),
        "integrity_protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
        "training_pool_protocol_version": TRAINING_POOL_PROTOCOL_VERSION,
        "screening_seed": SCREENING_SEED,
        "screening_order": "environment_balanced_sha256_task_id_within_environment",
        "screening_policy": "accept expert success and ordinary policy failure; reject only exact fresh-reset replay-confirmed strong environment errors",
        "final_task_statuses": list(FINAL_TASK_STATUSES),
        "tokenizer": model_artifact_identity(args.tokenizer),
        "model": args.model,
        "api_base": args.api_base,
        "awm_base_url": args.awm_base_url,
        "awm_logical_time": fetch_server_protocol(args.awm_base_url),
        **screening_rollout_protocol(args.max_tokens),
        "max_response_tokens": int(args.max_tokens),
        "thinking": True,
        "reasoning_effort": "max",
        "native_function_calling": True,
        "reasoning_history": "provider-native reasoning_content preserved per tool exchange",
        "parallel_tool_calls": False,
        "expert_multiple_calls": "execute_first",
        "verifier_mode": "code",
        "infrastructure_attempts": int(args.infrastructure_attempts),
    }
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

    prior_manifest_path = args.output_dir / "screening_manifest.json"
    prior_cumulative_usage = None
    if prior_manifest_path.is_file():
        prior_cumulative_usage = json.loads(prior_manifest_path.read_text(encoding="utf-8")).get("cumulative_usage")

    trial_records = _load_jsonl(trials_path)
    candidate_ids = {row["task_id"] for row in rows}
    validate_trial_records(trial_records, candidate_ids)
    prior_identity = provider_identity_from_trials(trial_records)
    if prior_identity is not None and prior_identity["model"] != args.model:
        raise RuntimeError(f"resumed provider model {prior_identity['model']!r} does not match {args.model!r}")
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
            accepted = sum(task_resolution(row["task_id"], records_by_task) in ACCEPTED_TASK_STATUSES for row in rows)
            print(
                f"expert_screening_resolved {len(records_by_task)}/{len(rows)} accepted={accepted}",
                flush=True,
            )

    async def screen_task(row: dict[str, Any]) -> None:
        async with slots:
            errors = []
            last_result = None
            last_replay = None
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
                    replay = await replay_runtime_failure(
                        row,
                        result,
                        awm_base_url=args.awm_base_url,
                    )
                    last_replay = replay
                    if replay["status"] == "confirmed":
                        await record_trial(
                            {
                                "task_id": row["task_id"],
                                "seed": SCREENING_SEED,
                                "status": "environment_failure",
                                "infrastructure_attempts": infrastructure_attempt,
                                "infrastructure_errors": errors,
                                "runtime_replay": replay,
                                "result": result,
                            }
                        )
                        return
                    if replay["status"] in {"pending", "resolved"}:
                        errors.append(f"runtime replay {replay['status']}: {replay.get('error') or replay.get('phase')}")
                        continue
                    status = screening_result_status(result)
                    if status == "infrastructure_error":
                        errors.append(f"verifier infrastructure reward_type={result.get('reward_type')!r}")
                        continue
                    await record_trial(
                        {
                            "task_id": row["task_id"],
                            "seed": SCREENING_SEED,
                            "status": status,
                            "infrastructure_attempts": infrastructure_attempt,
                            "infrastructure_errors": errors,
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
                    "last_result": last_result,
                    "last_runtime_replay": last_replay,
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
    parser.add_argument("--tokenizer", required=True)
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
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--max-new-tasks", type=int)
    limit.add_argument("--max-new-task-fraction", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
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
