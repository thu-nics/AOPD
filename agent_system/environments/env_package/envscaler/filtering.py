"""Deterministic EnvScaler task integrity audit and parquet materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .source import (
    DEFAULT_SOURCE_ROOT,
    build_environment_instance,
    checker_summary,
    evaluate_checkers,
    load_envscaler_source,
    state_dict,
    validate_tool_contract,
)

FILTER_PROTOCOL_VERSION = 1
MAX_CHECKERS = 100


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def audit_task(
    task_index: int,
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    source = load_envscaler_source(source_root)
    task = deepcopy(source.tasks[int(task_index)])
    task_id = str(task.get("task_id") or "")
    env_id = str(task.get("env_id") or "")
    reasons = []
    diagnostics: dict[str, Any] = {}
    checkers = task.get("checklist_with_func")
    if not isinstance(checkers, list) or not checkers:
        reasons.append("missing_checkers")
        checkers = []
    if len(checkers) > MAX_CHECKERS:
        reasons.append("checker_count_exceeds_limit")
    checker_codes = [str(item.get("check_func") or "") for item in checkers]
    if any(not code for code in checker_codes):
        reasons.append("empty_checker_code")
    duplicates = len(checker_codes) - len(set(checker_codes))
    if duplicates:
        reasons.append("exact_duplicate_checker")
    diagnostics["duplicate_checker_count"] = duplicates
    diagnostics["checker_count"] = len(checkers)

    environment = source.environments.get(env_id)
    if environment is None:
        reasons.append("missing_environment")
    else:
        try:
            first = build_environment_instance(environment, task)
            second = build_environment_instance(environment, task)
            first_state = state_dict(first)
            second_state = state_dict(second)
            diagnostics["initial_state_sha256"] = hashlib.sha256(_canonical(first_state).encode("utf-8")).hexdigest()
            if _canonical(first_state) != _canonical(second_state):
                reasons.append("fresh_reset_state_mismatch")
            tools = validate_tool_contract(environment, first)
            diagnostics["tool_count"] = len(tools)
            checks = evaluate_checkers(task, first_state, first_state)
            summary = checker_summary(checks)
            diagnostics.update(
                {
                    "initial_checker_passed": summary["checker_passed"],
                    "initial_checker_fraction": summary["checker_fraction"],
                    "checker_error_count": len(summary["checker_errors"]),
                    "checker_errors": summary["checker_errors"],
                }
            )
            if summary["checker_errors"]:
                reasons.append("checker_runtime_or_contract_error")
            if summary["state_complete"]:
                reasons.append("initial_state_already_complete")
        except Exception as exc:
            reasons.append("environment_or_tool_contract_error")
            diagnostics["environment_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "protocol_version": FILTER_PROTOCOL_VERSION,
        "task_index": int(task_index),
        "task_id": task_id,
        "env_id": env_id,
        "status": "quarantine" if reasons else "pass",
        "reasons": sorted(set(reasons)),
        "diagnostics": diagnostics,
    }


def training_row(task_index: int, task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "data_source": "envscaler",
        "prompt": [{"role": "user", "content": str(task["task"])}],
        "ability": "agentic_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "env_family": "envscaler",
            "task_index": int(task_index),
            "task_id": str(task["task_id"]),
            "env_id": str(task["env_id"]),
            "task": str(task["task"]),
        },
        "env_kwargs": {
            "env_family": "envscaler",
            "task_index": int(task_index),
            "task_id": str(task["task_id"]),
        },
    }


def run_deterministic_audit(
    *,
    source_root: str | Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    source = load_envscaler_source(source_root)
    count = len(source.tasks) if limit is None else min(int(limit), len(source.tasks))
    if count <= 0:
        raise ValueError("deterministic audit limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "task_audit.jsonl"
    records = [audit_task(index, source_root=source_root) for index in range(count)]
    audit_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    passed = [item for item in records if item["status"] == "pass"]
    manifest = {
        "kind": "envscaler_deterministic_audit",
        "protocol_version": FILTER_PROTOCOL_VERSION,
        "source": source.identity,
        "total_tasks": count,
        "pass_tasks": len(passed),
        "quarantine_tasks": count - len(passed),
        "pass_task_indices": [item["task_index"] for item in passed],
        "quarantine": [
            {
                "task_index": item["task_index"],
                "task_id": item["task_id"],
                "reasons": item["reasons"],
            }
            for item in records
            if item["status"] == "quarantine"
        ],
    }
    (output_dir / "deterministic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = run_deterministic_audit(
        source_root=args.source_root,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
