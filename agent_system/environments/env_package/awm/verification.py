#!/usr/bin/env python3
"""Verify AWM data artifacts and materialize deterministic training slices."""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from .native_rollout import sha256_file
from .qualification import QUALIFICATION_PROTOCOL_VERSION, qualification_rollout_protocol

TRAINING_SLICE_PROTOCOL_VERSION = 1


def _verify_rollout_protocol(manifest: dict) -> None:
    expected = qualification_rollout_protocol()
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"AWM qualification rollout protocol mismatch: expected {expected!r}, got {actual!r}")


def _verify_migration(manifest: dict, manifest_path: Path) -> None:
    provenance = manifest.get("migration_provenance")
    if provenance is None:
        return
    if provenance.get("protocol_version") != 2 or provenance.get("api_calls") != 0:
        raise RuntimeError("AWM qualification migration provenance mismatch")
    if provenance.get("to_qualification_protocol") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification migration target-protocol mismatch")
    source_protocol = provenance.get("from_qualification_protocol")
    if source_protocol not in (6, 7, QUALIFICATION_PROTOCOL_VERSION):
        raise RuntimeError("AWM qualification migration source-protocol mismatch")
    context_binding = provenance.get("source_context_binding") or {}
    expected_binding_status = "manifest_bound" if source_protocol == QUALIFICATION_PROTOCOL_VERSION else "operator_confirmed"
    if context_binding.get("status") != expected_binding_status or context_binding.get("rollout_protocol") != qualification_rollout_protocol():
        raise RuntimeError("AWM qualification migration source-context binding mismatch")
    root = manifest_path.parent.resolve()
    archive_dir = (manifest_path.parent / str(provenance["archive_subdir"])).resolve()
    if not archive_dir.is_relative_to(root):
        raise RuntimeError("AWM qualification migration archive escapes its root")
    archive_manifest_path = archive_dir / "archive_manifest.json"
    if sha256_file(archive_manifest_path) != provenance.get("archive_manifest_sha256"):
        raise RuntimeError("AWM qualification migration archive-manifest hash mismatch")
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if archive_manifest.get("protocol_version") != 2:
        raise RuntimeError("AWM qualification migration archive protocol mismatch")
    if archive_manifest.get("from_qualification_protocol") != source_protocol or archive_manifest.get("to_qualification_protocol") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification migration archive transition mismatch")
    if archive_manifest.get("source_qualification_manifest_sha256") != provenance.get("source_qualification_manifest_sha256"):
        raise RuntimeError("AWM qualification migration source-manifest mismatch")
    archive_root = archive_dir.resolve()
    for relative, expected_hash in (archive_manifest.get("files") or {}).items():
        path = (archive_dir / relative).resolve()
        if not path.is_relative_to(archive_root):
            raise RuntimeError("AWM qualification migration archive path escapes its root")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"AWM qualification migration archive hash mismatch: {relative}")
    archived_manifest = archive_dir / "qualification_manifest.json"
    if sha256_file(archived_manifest) != provenance.get("source_qualification_manifest_sha256"):
        raise RuntimeError("AWM qualification archived source manifest hash mismatch")


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


def verify_training_pool(data: Path, manifest_path: Path) -> dict:
    """Verify only the current deterministic pool accepted by training."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "awm_task_integrity_filter":
        raise RuntimeError("AWM training requires the current deterministic training-pool manifest; legacy qualification manifests are unsupported")
    return _verify_training_pool(data, manifest_path, manifest)


def verify(data: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") == "awm_task_integrity_filter":
        return _verify_training_pool(data, manifest_path, manifest)
    if manifest.get("protocol_version") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification protocol mismatch")
    _verify_rollout_protocol(manifest)
    _verify_migration(manifest, manifest_path)
    if data.name == "awm_expert_qualified_train_b8.parquet":
        hash_key = "qualified_train_b8_sha256"
        ids_key = "qualified_train_b8_task_ids"
    elif data.name == "awm_expert_qualified_all.parquet":
        hash_key = "qualified_all_sha256"
        ids_key = "qualified_task_ids"
    else:
        raise RuntimeError("qualified data filename must be awm_expert_qualified_all.parquet or awm_expert_qualified_train_b8.parquet")
    expected_hash = manifest.get(hash_key)
    if not expected_hash or sha256_file(data) != expected_hash:
        raise RuntimeError("AWM qualified Parquet hash mismatch")
    frame = pd.read_parquet(data)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"].tolist()]
    if task_ids != manifest.get(ids_key):
        raise RuntimeError("AWM qualified Parquet task IDs do not match its manifest")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("AWM qualified Parquet contains duplicate task IDs")
    return {"tasks": len(task_ids), "data": str(data)}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
