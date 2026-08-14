"""Deterministic per-family, per-environment mixed training schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from agent_system.environments.static_feasibility import STATIC_FEASIBILITY_PROTOCOL_VERSION

from .envs import interleave_families


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EnvironmentRoundRobin:
    def __init__(self, rows: list[dict[str, Any]], *, family: str):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            extra = row.get("extra_info") or {}
            if family == "awm":
                environment_id = str(extra.get("scenario") or "")
                task_id = str(extra.get("task_id") or "")
            else:
                environment_id = str(extra.get("env_id") or "")
                task_id = str(extra.get("task_id") or "")
            if not environment_id or not task_id:
                raise ValueError(f"{family} row is missing environment/task identity")
            groups[environment_id].append(deepcopy(row))
        if not groups:
            raise ValueError(f"{family} schedule source is empty")
        self.environment_ids = sorted(groups)
        self.groups = {
            key: sorted(
                values,
                key=lambda row: str((row.get("extra_info") or {}).get("task_id") or ""),
            )
            for key, values in groups.items()
        }
        self.environment_position = 0
        self.task_positions = {key: 0 for key in groups}

    def next(self) -> dict[str, Any]:
        environment_id = self.environment_ids[self.environment_position]
        self.environment_position = (self.environment_position + 1) % len(self.environment_ids)
        rows = self.groups[environment_id]
        position = self.task_positions[environment_id]
        self.task_positions[environment_id] = (position + 1) % len(rows)
        return deepcopy(rows[position])


def materialize_mixed_schedule(
    *,
    awm_data: Path,
    envscaler_data: Path,
    envscaler_manifest: Path,
    output_data: Path,
    output_manifest: Path,
    train_steps: int,
    awm_per_step: int,
    envscaler_per_step: int,
) -> dict[str, Any]:
    train_steps = int(train_steps)
    counts = {
        "awm": int(awm_per_step),
        "envscaler": int(envscaler_per_step),
    }
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    labels = interleave_families(counts)
    awm_frame = pd.read_parquet(awm_data)
    envscaler_frame = pd.read_parquet(envscaler_data)
    health = json.loads(envscaler_manifest.read_text())
    if health.get("kind") != "envscaler_healthy_task_pool":
        raise RuntimeError("unexpected EnvScaler health manifest kind")
    if health.get("protocol_version") != STATIC_FEASIBILITY_PROTOCOL_VERSION:
        raise RuntimeError("EnvScaler health manifest protocol mismatch")
    if health.get("expert_outcome_membership_gate") is not False:
        raise RuntimeError("EnvScaler health-pool membership must not depend on expert outcome")
    training_artifact = (health.get("artifacts") or {}).get("training_pool") or {}
    expected_pool_hash = str(training_artifact.get("sha256") or "")
    if not expected_pool_hash:
        raise RuntimeError("EnvScaler health manifest lacks training-pool hash")
    actual_pool_hash = _sha256(envscaler_data)
    if actual_pool_hash != expected_pool_hash:
        raise RuntimeError("EnvScaler training parquet hash does not match health manifest")
    accepted_ids = [str(value) for value in health.get("accepted_task_ids") or []]
    actual_ids = [str((item or {}).get("task_id") or "") for item in envscaler_frame["extra_info"].tolist()]
    if actual_ids != accepted_ids:
        raise RuntimeError("EnvScaler training parquet does not match health manifest order")
    schedulers = {
        "awm": EnvironmentRoundRobin(awm_frame.to_dict(orient="records"), family="awm"),
        "envscaler": EnvironmentRoundRobin(envscaler_frame.to_dict(orient="records"), family="envscaler"),
    }
    output = []
    for step in range(train_steps):
        for slot, family in enumerate(labels):
            row = schedulers[family].next()
            kwargs = dict(row.get("env_kwargs") or {})
            kwargs["env_family"] = family
            row["env_kwargs"] = kwargs
            extra = dict(row.get("extra_info") or {})
            extra["env_family"] = family
            extra["schedule_step"] = step
            extra["schedule_slot"] = slot
            row["extra_info"] = extra
            output.append(row)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_parquet(output_data, index=False)
    manifest = {
        "kind": "awm_envscaler_training_schedule",
        "protocol_version": 1,
        "counts_per_step": counts,
        "family_slot_order": labels,
        "train_steps": train_steps,
        "rows": len(output),
        "sources": {
            "awm_data": str(awm_data),
            "awm_data_sha256": _sha256(awm_data),
            "envscaler_data": str(envscaler_data),
            "envscaler_data_sha256": _sha256(envscaler_data),
            "envscaler_manifest": str(envscaler_manifest),
            "envscaler_manifest_sha256": _sha256(envscaler_manifest),
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--awm-data", type=Path, required=True)
    parser.add_argument("--envscaler-data", type=Path, required=True)
    parser.add_argument("--envscaler-manifest", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--awm-per-step", type=int, default=48)
    parser.add_argument("--envscaler-per-step", type=int, default=16)
    args = parser.parse_args()
    manifest = materialize_mixed_schedule(
        awm_data=args.awm_data,
        envscaler_data=args.envscaler_data,
        envscaler_manifest=args.envscaler_manifest,
        output_data=args.output_data,
        output_manifest=args.output_manifest,
        train_steps=args.train_steps,
        awm_per_step=args.awm_per_step,
        envscaler_per_step=args.envscaler_per_step,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
