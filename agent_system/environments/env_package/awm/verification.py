#!/usr/bin/env python3
"""Verify an AWM expert-qualified Parquet against its strict manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .native_rollout import sha256_file
from .qualification import QUALIFICATION_PROTOCOL_VERSION, qualification_rollout_protocol


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


def verify(data: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
