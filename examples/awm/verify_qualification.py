#!/usr/bin/env python3
"""Verify an AWM expert-qualified Parquet against its strict manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from examples.awm.native_rollout import sha256_file
from examples.awm.qualify_expert import QUALIFICATION_PROTOCOL_VERSION


def verify(data: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification protocol mismatch")
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
