"""Deterministic AWM scenario, database, and SQL-verifier health audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..runtime.rollout import sha256_file
from .health import (
    SCENARIO_HEALTH_FILENAME,
    _load_candidate_rows,
    _load_multimap,
    _normalize_scenario,
    audit_scenarios,
    audit_sql_verifier,
)
from .integrity import _load_jsonl, _write_jsonl
from .prepare import DATASET_NAME, DATASET_REVISION, EXPECTED_SOURCE_SHA256
from .selection import SELECTION_PROTOCOL_VERSION

PROTOCOL_VERSION = 1
KIND = "awm_deterministic_health_audit"
MANIFEST_FILENAME = "deterministic_manifest.json"
TASK_FILENAME = "task_audit.jsonl"
CONFIG_FILENAME = "config.json"


def _identity(
    *,
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    candidate_manifest: Path,
    data: Path,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": KIND,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": dict(source_hashes),
        "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(candidate_manifest),
        "candidate_data_sha256": sha256_file(data),
        "selection_counts": selection["selected_counts"],
        "candidate_task_ids": [row["task_id"] for row in rows],
        "policy": ("context eligible AND unique ten-task scenario sources AND strict database build AND one exact-task compilable SQL verifier defining verify_task"),
    }


def verify(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("AWM deterministic-audit protocol mismatch")
    if manifest.get("kind") != KIND:
        raise RuntimeError("AWM deterministic manifest has the wrong kind")
    paths = {
        "scenario_health_sha256": output_dir / SCENARIO_HEALTH_FILENAME,
        "task_audit_sha256": output_dir / TASK_FILENAME,
    }
    for field, path in paths.items():
        if not path.is_file() or sha256_file(path) != manifest.get(field):
            raise RuntimeError(f"AWM deterministic-audit artifact hash mismatch: {path}")

    scenario_records = _load_jsonl(output_dir / SCENARIO_HEALTH_FILENAME)
    for record in scenario_records:
        status = record.get("status")
        reasons = list(record.get("status_reasons") or [])
        database_errors = list(record.get("database_errors") or [])
        if status not in {"healthy", "quarantine"}:
            raise RuntimeError("AWM scenario-health audit is not binary")
        if (status == "healthy") != (not reasons and not database_errors):
            raise RuntimeError("AWM scenario-health status disagrees with its evidence")

    task_records = _load_jsonl(output_dir / TASK_FILENAME)
    candidate_ids = [str(value) for value in manifest.get("candidate_task_ids") or []]
    if [str(record["task_id"]) for record in task_records] != candidate_ids:
        raise RuntimeError("AWM deterministic task-audit order mismatch")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("AWM deterministic candidates contain duplicate task IDs")
    for record in task_records:
        status = record.get("status")
        reasons = list(record.get("status_reasons") or [])
        if status not in {"healthy", "quarantine"}:
            raise RuntimeError("AWM deterministic task audit is not binary")
        if (status == "healthy") != (not reasons):
            raise RuntimeError("AWM deterministic task status disagrees with its evidence")
        if status == "healthy" and not record.get("sql_verifier_sha256"):
            raise RuntimeError("AWM deterministic-pass task lacks SQL-verifier identity")
    healthy = [record for record in task_records if record["status"] == "healthy"]
    expected_counts = {
        "context_eligible": len(candidate_ids),
        "healthy": len(healthy),
        "quarantine": len(candidate_ids) - len(healthy),
        "healthy_environments": len({str(record["scenario"]) for record in healthy}),
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError("AWM deterministic-audit manifest counts mismatch")
    return {"tasks": len(healthy), "quarantine": len(candidate_ids) - len(healthy), "kind": KIND}


def build(
    *,
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    data_dir: Path,
    candidate_manifest: Path,
    data: Path,
    output_dir: Path,
) -> dict[str, Any]:
    identity = _identity(
        rows=rows,
        selection=selection,
        source_hashes=source_hashes,
        candidate_manifest=candidate_manifest,
        data=data,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / CONFIG_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("AWM deterministic-audit configuration mismatch")
        verify(output_dir)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_files = {path.name for path in output_dir.iterdir()}
    allowed_partial = {CONFIG_FILENAME, SCENARIO_HEALTH_FILENAME, TASK_FILENAME}
    if existing_files:
        if existing_files - allowed_partial:
            raise FileExistsError(f"refusing to overwrite unexpected deterministic audit artifacts in {output_dir}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("AWM incomplete deterministic-audit configuration mismatch")
    else:
        config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    scenario_order = list(dict.fromkeys(row["scenario"] for row in rows))
    scenario_records = audit_scenarios(data_dir, scenario_order)
    scenario_path = output_dir / SCENARIO_HEALTH_FILENAME
    _write_jsonl(scenario_path, scenario_records)
    scenario_by_name = {record["scenario"]: record for record in scenario_records}
    verifier_index = _load_multimap(
        data_dir / "gen_verifier.jsonl",
        lambda record: (_normalize_scenario(record["scenario"]), int(record["task_idx"])),
    )
    task_records = []
    for row in rows:
        reasons = []
        if scenario_by_name[row["scenario"]]["status"] != "healthy":
            reasons.append("scenario_quarantine")
        key = (_normalize_scenario(row["scenario"]), int(row["task_idx"]))
        verifier_reasons, verifier_sha256 = audit_sql_verifier(row, verifier_index.get(key) or [])
        reasons.extend(verifier_reasons)
        task_records.append(
            {
                "task_id": str(row["task_id"]),
                "scenario": row["scenario"],
                "task_idx": int(row["task_idx"]),
                "status": "healthy" if not reasons else "quarantine",
                "status_reasons": sorted(set(reasons)),
                "sql_verifier_sha256": verifier_sha256,
            }
        )
    task_path = output_dir / TASK_FILENAME
    _write_jsonl(task_path, task_records)
    healthy = [record for record in task_records if record["status"] == "healthy"]
    manifest = {
        **identity,
        "counts": {
            "context_eligible": len(rows),
            "healthy": len(healthy),
            "quarantine": len(rows) - len(healthy),
            "healthy_environments": len({record["scenario"] for record in healthy}),
        },
        "scenario_health_sha256": sha256_file(scenario_path),
        "task_audit_sha256": sha256_file(task_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify(output_dir)
    return manifest


def build_from_candidates(
    *,
    data: Path,
    candidate_manifest: Path,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows, selection = _load_candidate_rows(data, candidate_manifest)
    source_hashes = {name: sha256_file(data_dir / name) for name in EXPECTED_SOURCE_SHA256}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("AWM deterministic audit requires the pinned source file hashes")
    return build(
        rows=rows,
        selection=selection,
        source_hashes=source_hashes,
        data_dir=data_dir,
        candidate_manifest=candidate_manifest,
        data=data,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--awm-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        summary = verify(args.output_dir)
    else:
        manifest = build_from_candidates(
            data=args.data,
            candidate_manifest=args.candidate_manifest,
            data_dir=args.awm_data_dir,
            output_dir=args.output_dir,
        )
        summary = {
            "tasks": manifest["counts"]["healthy"],
            "quarantine": manifest["counts"]["quarantine"],
            "kind": manifest["kind"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
