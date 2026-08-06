#!/usr/bin/env python3
"""Verify AWM data artifacts and materialize deterministic training slices."""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from ..runtime.rollout import sha256_file

TRAINING_SLICE_PROTOCOL_VERSION = 1
TRAINING_SCHEDULE_PROTOCOL_VERSION = 1


def _validate_semantic_partition_identity(
    source_all_ids: list[str],
    manifest_source_ids: list[str],
    partition_ids: dict[str, list[str]],
    partition_names: tuple[str, ...],
) -> None:
    if manifest_source_ids != source_all_ids:
        raise RuntimeError("AWM verifier-reliable pinned all-task IDs mismatch")
    flattened = [task_id for name in partition_names for task_id in partition_ids[name]]
    if len(source_all_ids) != 10000 or len(set(source_all_ids)) != 10000 or len(flattened) != 10000 or len(flattened) != len(set(flattened)) or set(flattened) != set(source_all_ids):
        raise RuntimeError("AWM verifier-reliable partitions must exactly cover the pinned 10,000 tasks")
    for name in partition_names:
        selected = set(partition_ids[name])
        expected_order = [task_id for task_id in source_all_ids if task_id in selected]
        if partition_ids[name] != expected_order:
            raise RuntimeError(f"AWM verifier-reliable {name} partition order mismatch")


def _verify_semantic_pool(data: Path, manifest_path: Path, manifest: dict) -> dict:
    from ..screening.semantic.pipeline import (
        PARTITIONS,
        POOL_FILENAME,
        SEMANTIC_AUDIT_PROTOCOL_VERSION,
        verify_consensus_ledger,
    )
    from .selection import validate_base_manifest

    if manifest.get("protocol_version") != SEMANTIC_AUDIT_PROTOCOL_VERSION:
        raise RuntimeError("AWM verifier-reliable pool protocol mismatch")
    if data.name != POOL_FILENAME or manifest.get("training_pool_filename") != POOL_FILENAME:
        raise RuntimeError("AWM verifier-reliable pool filename mismatch")
    if sha256_file(data) != manifest.get("training_pool_data_sha256"):
        raise RuntimeError("AWM verifier-reliable pool Parquet hash mismatch")
    root = manifest_path.parent
    for field, filename in (
        ("review_plan_sha256", "review_plan.json"),
        ("review_queue_sha256", "review_queue.jsonl"),
        ("review_consensus_sha256", "review_consensus.jsonl"),
    ):
        if sha256_file(root / filename) != manifest.get(field):
            raise RuntimeError(f"AWM verifier-reliable provenance hash mismatch: {filename}")
    source_manifest_filename = manifest.get("source_all_manifest_filename")
    if source_manifest_filename != "source_all_manifest.json":
        raise RuntimeError("AWM verifier-reliable source manifest filename mismatch")
    source_manifest_path = root / source_manifest_filename
    if sha256_file(source_manifest_path) != manifest.get("source_all_manifest_sha256"):
        raise RuntimeError("AWM verifier-reliable source manifest hash mismatch")
    source_manifest = validate_base_manifest(source_manifest_path)
    source_all_ids = [str(value) for value in source_manifest["split_task_ids"]["all"]]
    review_plan = json.loads((root / "review_plan.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads(Path(review_plan["candidate_manifest_path"]).read_text(encoding="utf-8"))
    if manifest.get("source_all_manifest_sha256") != candidate_manifest.get("base_manifest_sha256"):
        raise RuntimeError("AWM verifier-reliable source manifest differs from context selection")
    if manifest.get("source_all_data_sha256") != candidate_manifest.get("base_data_sha256"):
        raise RuntimeError("AWM verifier-reliable source parquet differs from context selection")
    partition_ids = {}
    for name in PARTITIONS:
        path = root / f"{name}_task_ids.json"
        if sha256_file(path) != manifest.get(f"{name}_task_ids_sha256"):
            raise RuntimeError(f"AWM verifier-reliable {name} partition hash mismatch")
        partition_ids[name] = [str(value) for value in json.loads(path.read_text(encoding="utf-8"))]
        if len(partition_ids[name]) != int((manifest.get("counts") or {}).get(name, -1)):
            raise RuntimeError(f"AWM verifier-reliable {name} partition count mismatch")
    _validate_semantic_partition_identity(
        source_all_ids,
        [str(value) for value in manifest.get("source_all_task_ids") or []],
        partition_ids,
        PARTITIONS,
    )
    if partition_ids["included"] != manifest.get("training_pool_task_ids"):
        raise RuntimeError("AWM verifier-reliable included IDs mismatch")
    frame = pd.read_parquet(data)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    if task_ids != partition_ids["included"] or not task_ids:
        raise RuntimeError("AWM verifier-reliable pool task IDs mismatch")
    if set(manifest.get("task_reasons") or {}) != set(source_all_ids):
        raise RuntimeError("AWM verifier-reliable task reasons do not cover all tasks")
    verify_consensus_ledger(root)
    return {"tasks": len(task_ids), "data": str(data), "kind": "verifier_reliable_training_pool"}


def _verify_training_pool(data: Path, manifest_path: Path, manifest: dict) -> dict:
    from .integrity import (
        INTEGRITY_PROTOCOL_VERSION,
        TRAINING_POOL_FILENAME,
        TRAINING_POOL_PROTOCOL_VERSION,
        verify_integrity,
    )

    if manifest.get("protocol_version") != INTEGRITY_PROTOCOL_VERSION:
        raise RuntimeError("AWM deterministic training-pool protocol mismatch")
    if manifest.get("training_pool_protocol_version") != TRAINING_POOL_PROTOCOL_VERSION:
        raise RuntimeError("AWM training-pool partition protocol mismatch")
    if data.name != TRAINING_POOL_FILENAME:
        raise RuntimeError(f"AWM training pool filename must be {TRAINING_POOL_FILENAME}")
    verify_integrity(manifest_path.parent)
    if sha256_file(data) != manifest.get("training_pool_data_sha256"):
        raise RuntimeError("AWM training-pool Parquet hash mismatch")
    frame = pd.read_parquet(data)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    if task_ids != manifest.get("training_pool_task_ids"):
        raise RuntimeError("AWM training-pool task IDs do not match its manifest")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("AWM training-pool Parquet contains duplicate task IDs")
    return {"tasks": len(task_ids), "data": str(data), "kind": "deterministic_training_pool"}


def _verify_expert_screened_pool(data: Path, manifest_path: Path, manifest: dict) -> dict:
    from ..screening.expert import (
        ACCEPTED_TASK_STATUSES,
        EXPERT_SCREENING_PROTOCOL_VERSION,
        FINAL_TASK_STATUSES,
        _load_jsonl,
        task_resolution,
        validate_trial_records,
    )

    if manifest.get("protocol_version") != EXPERT_SCREENING_PROTOCOL_VERSION:
        raise RuntimeError("AWM expert-screening protocol mismatch")
    if manifest.get("kind") != "awm_one_pass_expert_screening":
        raise RuntimeError("AWM expert-screening manifest kind mismatch")
    config_path = manifest_path.parent / "config.json"
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise RuntimeError("AWM expert-screening config hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in config.items()):
        raise RuntimeError("AWM expert-screening manifest/config identity mismatch")
    candidate_ids = [str(value) for value in config.get("candidate_task_ids") or []]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("AWM expert-screening candidate task IDs must be unique")
    if data.name != "awm_expert_screened_pool.parquet":
        raise RuntimeError("AWM expert-screened pool filename mismatch")
    if manifest.get("training_pool_filename") != data.name:
        raise RuntimeError("AWM expert-screened manifest filename mismatch")
    if sha256_file(data) != manifest.get("training_pool_data_sha256"):
        raise RuntimeError("AWM expert-screened pool Parquet hash mismatch")
    trials_path = manifest_path.parent / "trials.jsonl"
    if sha256_file(trials_path) != manifest.get("trials_sha256"):
        raise RuntimeError("AWM expert-screening trials hash mismatch")
    trial_records = _load_jsonl(trials_path)
    validate_trial_records(trial_records, set(candidate_ids))
    records_by_task = {str(record["task_id"]): record for record in trial_records}
    statuses = {task_id: task_resolution(task_id, records_by_task) for task_id in candidate_ids}
    if manifest.get("task_status") != statuses:
        raise RuntimeError("AWM expert-screening task-status derivation mismatch")
    counts = {status: sum(value == status for value in statuses.values()) for status in FINAL_TASK_STATUSES}
    if manifest.get("counts") != counts:
        raise RuntimeError("AWM expert-screening status counts mismatch")
    accepted_ids = [task_id for task_id in candidate_ids if statuses[task_id] in ACCEPTED_TASK_STATUSES]
    if manifest.get("accepted_task_ids") != accepted_ids:
        raise RuntimeError("AWM expert-screening accepted task IDs mismatch")
    frame = pd.read_parquet(data)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    expected_ids = [str(value) for value in manifest.get("training_pool_task_ids") or []]
    if task_ids != expected_ids or task_ids != manifest.get("accepted_task_ids"):
        raise RuntimeError("AWM expert-screened pool task IDs do not match its manifest")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("AWM expert-screened pool must contain unique tasks")
    return {"tasks": len(task_ids), "data": str(data), "kind": "expert_screened_training_pool"}


def verify_training_pool(data: Path, manifest_path: Path) -> dict:
    """Verify only the current deterministic pool accepted by training."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") == "awm_task_integrity_filter":
        return _verify_training_pool(data, manifest_path, manifest)
    if manifest.get("kind") == "awm_one_pass_expert_screening":
        return _verify_expert_screened_pool(data, manifest_path, manifest)
    if manifest.get("kind") == "awm_verifier_reliable_task_pool":
        return _verify_semantic_pool(data, manifest_path, manifest)
    raise RuntimeError("AWM training requires a current deterministic, expert-screened, or verifier-reliable pool manifest")


def verify(data: Path, manifest_path: Path) -> dict:
    return verify_training_pool(data, manifest_path)


def _canonical_fraction(value: str | None) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, None
    try:
        fraction = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("training-task fraction must be a decimal number") from exc
    if not fraction.is_finite() or fraction <= 0 or fraction > 1:
        raise ValueError("training-task fraction must be in (0, 1]")
    return fraction, format(fraction.normalize(), "f")


def _slice_spec(
    source_manifest: dict[str, Any],
    *,
    available_tasks: int,
    task_count: int | None,
    fraction: str | None,
) -> dict[str, Any]:
    if task_count is not None and fraction is not None:
        raise ValueError("set only one of training-task count and fraction")
    if task_count is not None:
        if isinstance(task_count, bool) or int(task_count) <= 0:
            raise ValueError("training-task count must be positive")
        selected_tasks = int(task_count)
        if selected_tasks > available_tasks:
            raise ValueError(f"training-task count {selected_tasks} exceeds the verified pool size {available_tasks}")
        return {
            "requested_task_count": selected_tasks,
            "requested_fraction": None,
            "fraction_base_tasks": None,
            "selected_tasks": selected_tasks,
        }

    fraction_value, fraction_text = _canonical_fraction(fraction)
    if fraction_value is None:
        return {
            "requested_task_count": None,
            "requested_fraction": None,
            "fraction_base_tasks": None,
            "selected_tasks": available_tasks,
        }
    selection_counts = source_manifest.get("selection_counts") or {}
    fraction_base = int(selection_counts.get("tasks", available_tasks))
    if fraction_base < available_tasks:
        raise RuntimeError("AWM training pool exceeds its context-selection task count")
    requested = int((Decimal(fraction_base) * fraction_value).to_integral_value(rounding=ROUND_FLOOR))
    if requested <= 0:
        raise ValueError("training-task fraction selects zero tasks")
    return {
        "requested_task_count": None,
        "requested_fraction": fraction_text,
        "fraction_base_tasks": fraction_base,
        "selected_tasks": min(requested, available_tasks),
    }


def verify_training_slice(
    *,
    source_data: Path,
    source_manifest_path: Path,
    output_data: Path,
    output_manifest_path: Path,
    task_count: int | None = None,
    fraction: str | None = None,
) -> dict[str, Any]:
    verify_training_pool(source_data, source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_ids = [str(item) for item in source_manifest["training_pool_task_ids"]]
    spec = _slice_spec(
        source_manifest,
        available_tasks=len(source_ids),
        task_count=task_count,
        fraction=fraction,
    )
    manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "protocol_version": TRAINING_SLICE_PROTOCOL_VERSION,
        "kind": "awm_training_pool_slice",
        "selection": "ordered_prefix_after_deterministic_quarantine",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_data_sha256": sha256_file(source_data),
        "source_available_tasks": len(source_ids),
        **spec,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"AWM training-slice manifest mismatch: {key}")
    if sha256_file(output_data) != manifest.get("data_sha256"):
        raise RuntimeError("AWM training-slice Parquet hash mismatch")
    frame = pd.read_parquet(output_data)
    output_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    expected_ids = source_ids[: int(spec["selected_tasks"])]
    if output_ids != expected_ids or output_ids != manifest.get("task_ids"):
        raise RuntimeError("AWM training slice is not the expected ordered prefix")
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("AWM training slice contains duplicate task IDs")
    return {
        "tasks": len(output_ids),
        "source_tasks": len(source_ids),
        "data": str(output_data),
        "kind": "deterministic_training_pool_slice",
    }


def materialize_training_slice(
    *,
    source_data: Path,
    source_manifest_path: Path,
    output_data: Path,
    output_manifest_path: Path,
    task_count: int | None = None,
    fraction: str | None = None,
) -> dict[str, Any]:
    verify_training_pool(source_data, source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_frame = pd.read_parquet(source_data)
    source_ids = [str(item["task_id"]) for item in source_frame["extra_info"].tolist()]
    spec = _slice_spec(
        source_manifest,
        available_tasks=len(source_ids),
        task_count=task_count,
        fraction=fraction,
    )
    if output_data.exists() or output_manifest_path.exists():
        if not output_data.is_file() or not output_manifest_path.is_file():
            raise RuntimeError("AWM training-slice artifacts are incomplete")
        return verify_training_slice(
            source_data=source_data,
            source_manifest_path=source_manifest_path,
            output_data=output_data,
            output_manifest_path=output_manifest_path,
            task_count=task_count,
            fraction=fraction,
        )
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected_frame = source_frame.iloc[: int(spec["selected_tasks"])].copy()
    selected_frame.to_parquet(output_data, index=False)
    manifest = {
        "protocol_version": TRAINING_SLICE_PROTOCOL_VERSION,
        "kind": "awm_training_pool_slice",
        "selection": "ordered_prefix_after_deterministic_quarantine",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_data_sha256": sha256_file(source_data),
        "source_available_tasks": len(source_ids),
        **spec,
        "task_ids": source_ids[: int(spec["selected_tasks"])],
        "data_sha256": sha256_file(output_data),
    }
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verify_training_slice(
        source_data=source_data,
        source_manifest_path=source_manifest_path,
        output_data=output_data,
        output_manifest_path=output_manifest_path,
        task_count=task_count,
        fraction=fraction,
    )


def _verified_schedule_source(data: Path, manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") in {
        "awm_task_integrity_filter",
        "awm_one_pass_expert_screening",
        "awm_verifier_reliable_task_pool",
    }:
        verify_training_pool(data, manifest_path)
        expected_hash = manifest["training_pool_data_sha256"]
        expected_ids = [str(value) for value in manifest["training_pool_task_ids"]]
    elif manifest.get("kind") == "awm_training_pool_slice":
        if manifest.get("protocol_version") != TRAINING_SLICE_PROTOCOL_VERSION:
            raise RuntimeError("AWM training-slice protocol mismatch")
        expected_hash = manifest.get("data_sha256")
        expected_ids = [str(value) for value in manifest.get("task_ids") or []]
    else:
        raise RuntimeError("AWM schedule requires a verified pool or deterministic slice")
    if sha256_file(data) != expected_hash:
        raise RuntimeError("AWM schedule source Parquet hash mismatch")
    frame = pd.read_parquet(data)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    if task_ids != expected_ids:
        raise RuntimeError("AWM schedule source task IDs do not match its manifest")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("AWM schedule source must contain unique tasks")
    return frame, task_ids


def verify_training_schedule(
    *,
    source_data: Path,
    source_manifest_path: Path,
    output_data: Path,
    output_manifest_path: Path,
    train_steps: int,
    train_batch_size: int,
) -> dict[str, Any]:
    _, source_ids = _verified_schedule_source(source_data, source_manifest_path)
    if train_steps <= 0 or train_batch_size <= 0:
        raise ValueError("train_steps and train_batch_size must be positive")
    total_rows = int(train_steps) * int(train_batch_size)
    expected_ids = [source_ids[index % len(source_ids)] for index in range(total_rows)]
    manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    identity = {
        "protocol_version": TRAINING_SCHEDULE_PROTOCOL_VERSION,
        "kind": "awm_deterministic_cyclic_training_schedule",
        "selection": "verified_source_order_cyclic",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_data_sha256": sha256_file(source_data),
        "source_tasks": len(source_ids),
        "train_steps": int(train_steps),
        "train_batch_size": int(train_batch_size),
        "total_rows": total_rows,
    }
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"AWM training-schedule manifest mismatch: {key}")
    if manifest.get("task_ids") != expected_ids:
        raise RuntimeError("AWM training schedule task order mismatch")
    if sha256_file(output_data) != manifest.get("data_sha256"):
        raise RuntimeError("AWM training-schedule Parquet hash mismatch")
    frame = pd.read_parquet(output_data)
    actual_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    if actual_ids != expected_ids:
        raise RuntimeError("AWM training-schedule Parquet order mismatch")
    return {
        "kind": identity["kind"],
        "source_tasks": len(source_ids),
        "rows": total_rows,
        "steps": int(train_steps),
        "data": str(output_data),
    }


def materialize_training_schedule(
    *,
    source_data: Path,
    source_manifest_path: Path,
    output_data: Path,
    output_manifest_path: Path,
    train_steps: int,
    train_batch_size: int,
) -> dict[str, Any]:
    source_frame, source_ids = _verified_schedule_source(source_data, source_manifest_path)
    if train_steps <= 0 or train_batch_size <= 0:
        raise ValueError("train_steps and train_batch_size must be positive")
    if output_data.exists() or output_manifest_path.exists():
        if not output_data.is_file() or not output_manifest_path.is_file():
            raise RuntimeError("AWM training-schedule artifacts are incomplete")
        return verify_training_schedule(
            source_data=source_data,
            source_manifest_path=source_manifest_path,
            output_data=output_data,
            output_manifest_path=output_manifest_path,
            train_steps=train_steps,
            train_batch_size=train_batch_size,
        )

    total_rows = int(train_steps) * int(train_batch_size)
    indices = [index % len(source_frame) for index in range(total_rows)]
    task_ids = [source_ids[index] for index in indices]
    schedule = source_frame.iloc[indices].reset_index(drop=True).copy()
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(output_data, index=False)
    quotient, remainder = divmod(total_rows, len(source_ids))
    manifest = {
        "protocol_version": TRAINING_SCHEDULE_PROTOCOL_VERSION,
        "kind": "awm_deterministic_cyclic_training_schedule",
        "selection": "verified_source_order_cyclic",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_data_sha256": sha256_file(source_data),
        "source_tasks": len(source_ids),
        "train_steps": int(train_steps),
        "train_batch_size": int(train_batch_size),
        "total_rows": total_rows,
        "complete_source_passes": quotient,
        "partial_next_pass_tasks": remainder,
        "minimum_task_occurrences": quotient,
        "maximum_task_occurrences": quotient + int(remainder > 0),
        "task_ids": task_ids,
        "data_sha256": sha256_file(output_data),
    }
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verify_training_schedule(
        source_data=source_data,
        source_manifest_path=source_manifest_path,
        output_data=output_data,
        output_manifest_path=output_manifest_path,
        train_steps=train_steps,
        train_batch_size=train_batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
