#!/usr/bin/env python3
"""Qualify AWM tasks with four independent DeepSeek expert trajectories."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from examples.awm.native_rollout import (
    model_artifact_identity,
    run_native_trajectory,
    sha256_file,
)
from examples.awm.select_tasks import (
    SELECTION_PROTOCOL_VERSION,
    stable_rank,
)

QUALIFICATION_PROTOCOL_VERSION = 1
TRIAL_SEEDS = (300, 301, 302, 303)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                # The final ordered rewrite drops a torn append safely.
                continue
            raise RuntimeError(f"invalid qualification JSONL record {index + 1}") from exc
    return records


def provider_identity_from_trials(
    trial_records: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    identities = {(str(entry.get("model") or ""), entry.get("system_fingerprint")) for record in trial_records for entry in ((record.get("result") or {}).get("trajectory") or []) if entry.get("model")}
    if len(identities) > 1:
        raise RuntimeError(f"qualification trials mix provider identities: {identities!r}")
    if not identities:
        return None
    model, fingerprint = next(iter(identities))
    return {"model": model, "system_fingerprint": fingerprint}


def cumulative_usage_from_trials(
    trial_records: list[Mapping[str, Any]],
) -> dict[str, int]:
    usage = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in trial_records:
        for entry in (record.get("result") or {}).get("trajectory") or []:
            usage["requests"] += 1
            entry_usage = entry.get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[name] += int(entry_usage.get(name, 0) or 0)
    return usage


def load_candidate_rows(data_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM candidate selection protocol mismatch")
    if sha256_file(data_path) != manifest.get("candidate_data_sha256"):
        raise RuntimeError("AWM candidate parquet hash mismatch")
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
                "scaffold_tokens": int(extra["scaffold_tokens"]),
                "tool_schema_hash": str(extra["tool_schema_hash"]),
                "training_row": raw.to_dict(),
            }
        )
    if [row["task_id"] for row in rows] != manifest.get("task_ids"):
        raise RuntimeError("AWM candidate manifest and parquet IDs differ")
    return rows, manifest


class DeepSeekExpertPolicy:
    """One strict asynchronous DeepSeek policy shared by qualification tasks."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        api_base: str,
        max_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        expected_identity: Mapping[str, Any] | None = None,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.model = str(model)
        self.max_tokens = int(max_tokens)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._identity = dict(expected_identity) if expected_identity is not None else None
        self._identity_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self._stats = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            extra_body={
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
        )
        identity = {
            "model": str(response.model or ""),
            "system_fingerprint": response.system_fingerprint,
        }
        if identity["model"] != self.model:
            raise RuntimeError(f"DeepSeek returned model {identity['model']!r}, expected {self.model!r}")
        async with self._identity_lock:
            if self._identity is not None and self._identity != identity:
                raise RuntimeError(f"DeepSeek provider identity changed: {self._identity!r} -> {identity!r}")
            self._identity = identity
        message = response.choices[0].message
        usage = response.usage.model_dump() if response.usage is not None else {}
        async with self._stats_lock:
            self._stats["requests"] += 1
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._stats[name] += int(usage.get(name, 0) or 0)
        return {
            "content": message.content or "",
            "reasoning_content": getattr(message, "reasoning_content", "") or "",
            "model": identity["model"],
            "system_fingerprint": identity["system_fingerprint"],
            "usage": usage,
        }

    async def identity(self) -> dict[str, Any] | None:
        async with self._identity_lock:
            return dict(self._identity) if self._identity is not None else None

    async def stats(self) -> dict[str, int]:
        async with self._stats_lock:
            return dict(self._stats)


def task_resolution(
    task_id: str,
    trials_by_task: Mapping[str, list[Mapping[str, Any]]],
) -> str:
    records = trials_by_task.get(task_id, [])
    if any(record.get("status") == "infrastructure_exhausted" for record in records):
        return "infrastructure_exhausted"
    if any(record.get("status") == "policy_failure" for record in records):
        return "policy_failure"
    successes = {int(record["trial_index"]) for record in records if record.get("status") == "success"}
    return "qualified" if successes == set(range(4)) else "pending"


def validate_trial_records(
    trial_records: list[Mapping[str, Any]],
    candidate_task_ids: set[str],
) -> None:
    seen = set()
    allowed_statuses = {"success", "policy_failure", "infrastructure_exhausted"}
    for record in trial_records:
        task_id = str(record.get("task_id") or "")
        if task_id not in candidate_task_ids:
            raise RuntimeError(f"qualification trial has unknown task ID {task_id!r}")
        try:
            trial_index = int(record["trial_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"qualification trial for {task_id} has invalid index") from exc
        if trial_index not in range(len(TRIAL_SEEDS)):
            raise RuntimeError(f"qualification trial for {task_id} has out-of-range index")
        if int(record.get("seed", -1)) != TRIAL_SEEDS[trial_index]:
            raise RuntimeError(f"qualification trial for {task_id} has a seed mismatch")
        key = (task_id, trial_index)
        if key in seen:
            raise RuntimeError(f"duplicate qualification trial {key!r}")
        seen.add(key)
        status = record.get("status")
        if status not in allowed_statuses:
            raise RuntimeError(f"qualification trial {key!r} has invalid status {status!r}")
        result = record.get("result")
        if status in {"success", "policy_failure"} and not isinstance(result, Mapping):
            raise RuntimeError(f"qualification policy trial {key!r} is missing its result")
        if status == "success" and not result.get("success"):
            raise RuntimeError(f"qualification success trial {key!r} has a failed result")
        if status == "policy_failure" and result.get("success"):
            raise RuntimeError(f"qualification failure trial {key!r} has a successful result")


def environment_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    for scenario in by_scenario:
        by_scenario[scenario].sort(key=lambda row: stable_rank(row["task_id"]))
    scenarios = sorted(by_scenario, key=stable_rank)
    maximum = max((len(items) for items in by_scenario.values()), default=0)
    return [by_scenario[scenario][rank] for rank in range(maximum) for scenario in scenarios if rank < len(by_scenario[scenario])]


def select_qwen_diagnostic(
    qualified_rows: list[dict[str, Any]],
    trials_by_task: Mapping[str, list[Mapping[str, Any]]],
    *,
    target: int = 32,
) -> list[dict[str, Any]]:
    if not qualified_rows:
        return []
    target = min(int(target), len(qualified_rows))
    values = np.asarray([row["scaffold_tokens"] for row in qualified_rows], dtype=np.float64)
    boundaries = np.quantile(values, [0.25, 0.5, 0.75])
    records = []
    for row in qualified_rows:
        successful = [record for record in trials_by_task[row["task_id"]] if record.get("status") == "success"]
        records.append(
            {
                **row,
                "scaffold_quartile": min(
                    int(np.searchsorted(boundaries, row["scaffold_tokens"], side="right")),
                    3,
                ),
                "expert_max_decisions": max(int(record["result"]["decisions"]) for record in successful),
            }
        )
    selected = []
    used_scenarios = set()
    for quartile in range(4):
        ranked = sorted(
            (record for record in records if record["scaffold_quartile"] == quartile),
            key=lambda record: (
                record["expert_max_decisions"],
                stable_rank(record["task_id"]),
            ),
        )
        for record in ranked:
            if len([item for item in selected if item["scaffold_quartile"] == quartile]) >= 8:
                break
            if record["scenario"] in used_scenarios:
                continue
            selected.append(record)
            used_scenarios.add(record["scenario"])
    if len(selected) < target:
        ranked = sorted(
            records,
            key=lambda record: (
                record["expert_max_decisions"],
                stable_rank(record["task_id"]),
            ),
        )
        selected_ids = {record["task_id"] for record in selected}
        for record in ranked:
            if len(selected) >= target:
                break
            if record["task_id"] in selected_ids or record["scenario"] in used_scenarios:
                continue
            selected.append(record)
            selected_ids.add(record["task_id"])
            used_scenarios.add(record["scenario"])
    return selected


async def qualify(args) -> None:
    rows, candidate_manifest = load_candidate_rows(args.data, args.candidate_manifest)
    identity = {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_data_sha256": sha256_file(args.data),
        "candidate_task_ids": [row["task_id"] for row in rows],
        "tokenizer": model_artifact_identity(args.tokenizer),
        "model": args.model,
        "api_base": args.api_base,
        "awm_base_url": args.awm_base_url,
        "trial_seeds": list(TRIAL_SEEDS),
        "trials_required": 4,
        "qualification": "4/4 with first policy failure early stop",
        "max_decisions": 20,
        "max_response_tokens": int(args.max_tokens),
        "thinking": True,
        "reasoning_effort": "max",
        "verifier_mode": "code",
        "infrastructure_attempts": int(args.infrastructure_attempts),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    trials_path = args.output_dir / "trials.jsonl"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != identity:
            raise RuntimeError("AWM qualification resume configuration mismatch")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    trial_records = _load_jsonl(trials_path)
    validate_trial_records(trial_records, {row["task_id"] for row in rows})
    prior_identity = provider_identity_from_trials(trial_records)
    if prior_identity is not None and prior_identity["model"] != args.model:
        raise RuntimeError(f"resumed provider model {prior_identity['model']!r} does not match {args.model!r}")
    trials_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        trials_by_task.setdefault(str(record["task_id"]), []).append(record)
    unresolved = [row for row in rows if task_resolution(row["task_id"], trials_by_task) == "pending"]
    if args.max_new_tasks is not None:
        unresolved = unresolved[: args.max_new_tasks]

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
    task_slots = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def record_trial(record):
        async with write_lock:
            _append_jsonl(trials_path, record)
            trial_records.append(record)
            trials_by_task.setdefault(record["task_id"], []).append(record)
            resolved = sum(task_resolution(row["task_id"], trials_by_task) != "pending" for row in rows)
            print(f"qualification_resolved {resolved}/{len(rows)}", flush=True)

    async def qualify_task(row):
        async with task_slots:
            existing = trials_by_task.get(row["task_id"], [])
            if any(record.get("status") != "success" for record in existing):
                return
            successful_indices = {int(record["trial_index"]) for record in existing if record.get("status") == "success"}
            for trial_index, seed in enumerate(TRIAL_SEEDS):
                if trial_index in successful_indices:
                    continue
                errors = []
                result = None
                for infrastructure_attempt in range(1, args.infrastructure_attempts + 1):
                    try:
                        result = await run_native_trajectory(
                            row,
                            generate_action=policy.generate,
                            tokenizer=tokenizer,
                            awm_base_url=args.awm_base_url,
                            seed=seed,
                        )
                        break
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                if result is None:
                    await record_trial(
                        {
                            "task_id": row["task_id"],
                            "trial_index": trial_index,
                            "seed": seed,
                            "status": "infrastructure_exhausted",
                            "infrastructure_attempts": args.infrastructure_attempts,
                            "errors": errors,
                        }
                    )
                    return
                status = "success" if result["success"] else "policy_failure"
                await record_trial(
                    {
                        "task_id": row["task_id"],
                        "trial_index": trial_index,
                        "seed": seed,
                        "status": status,
                        "infrastructure_attempts": len(errors) + 1,
                        "infrastructure_errors": errors,
                        "result": result,
                    }
                )
                if status != "success":
                    return

    try:
        await asyncio.gather(*(qualify_task(row) for row in unresolved))
    finally:
        await policy.client.close()

    # Rewrite in candidate/trial order once all append-only records are durable.
    order = {row["task_id"]: index for index, row in enumerate(rows)}
    trial_records.sort(key=lambda record: (order[record["task_id"]], int(record["trial_index"])))
    with trials_path.open("w", encoding="utf-8") as handle:
        for record in trial_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    resolutions = {row["task_id"]: task_resolution(row["task_id"], trials_by_task) for row in rows}
    qualified_rows = [row for row in rows if resolutions[row["task_id"]] == "qualified"]
    balanced = environment_balanced(qualified_rows)
    trim = len(balanced) % 8
    train_rows = balanced[:-trim] if trim else balanced
    all_path = args.output_dir / "awm_expert_qualified_all.parquet"
    train_path = args.output_dir / "awm_expert_qualified_train_b8.parquet"
    if qualified_rows:
        pd.DataFrame([row["training_row"] for row in qualified_rows]).to_parquet(all_path, index=False)
    if train_rows:
        pd.DataFrame([row["training_row"] for row in train_rows]).to_parquet(train_path, index=False)
    diagnostic = select_qwen_diagnostic(qualified_rows, trials_by_task, target=32)
    diagnostic_manifest = {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "kind": "awm_qwen_function_call_diagnostic",
        "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
        "qualification_trials_sha256": sha256_file(trials_path),
        "selection": ("up to 8 tasks per fixed-scaffold quartile; distinct environments; lower expert max-decision count first; sha256 task-id tie break"),
        "task_ids": [row["task_id"] for row in diagnostic],
        "records": [
            {
                "task_id": row["task_id"],
                "scenario": row["scenario"],
                "scaffold_tokens": row["scaffold_tokens"],
                "scaffold_quartile": row.get("scaffold_quartile"),
                "expert_max_decisions": row.get("expert_max_decisions"),
            }
            for row in diagnostic
        ],
    }
    diagnostic_path = args.output_dir / "qwen_diagnostic_manifest.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for resolution in resolutions.values():
        counts[resolution] = counts.get(resolution, 0) + 1
    live_usage = await policy.stats()
    cumulative_usage = cumulative_usage_from_trials(trial_records)
    manifest = {
        **identity,
        "kind": "awm_expert_qualification",
        "candidate_selection_counts": candidate_manifest["selected_counts"],
        "provider_identity": await policy.identity(),
        "live_usage": live_usage,
        "cumulative_usage": cumulative_usage,
        "counts": {
            **dict(sorted(counts.items())),
            "qualified_train_b8": len(train_rows),
            "batch_trimmed": trim,
            "qwen_diagnostic_tasks": len(diagnostic),
        },
        "task_status": resolutions,
        "qualified_task_ids": [row["task_id"] for row in qualified_rows],
        "qualified_train_b8_task_ids": [row["task_id"] for row in train_rows],
        "trials_sha256": sha256_file(trials_path),
        "qualified_all_sha256": sha256_file(all_path) if all_path.is_file() else None,
        "qualified_train_b8_sha256": sha256_file(train_path) if train_path.is_file() else None,
        "qwen_diagnostic_manifest_sha256": sha256_file(diagnostic_path),
    }
    (args.output_dir / "qualification_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        **manifest["counts"],
        "live_usage": live_usage,
        "cumulative_usage": cumulative_usage,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--infrastructure-attempts", type=int, default=3)
    parser.add_argument("--max-new-tasks", type=int)
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
    asyncio.run(qualify(args))


if __name__ == "__main__":
    main()
