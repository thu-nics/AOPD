"""Offline migration from legacy AWM qualification caches to the prefilter protocol."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .integrity import PREFILTER_PROTOCOL_VERSION, verify_integrity
from .native_rollout import sha256_file
from .qualification import (
    FINAL_TASK_STATUSES,
    QUALIFICATION_PROTOCOL_VERSION,
    _load_jsonl,
    load_candidate_rows,
    provider_identity_from_trials,
    validate_trial_records,
    write_qualification_artifacts,
)

LEGACY_QUALIFICATION_PROTOCOL_VERSION = 6
MIGRATION_PROTOCOL_VERSION = 1


def _archive_inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "archive_manifest.json"}


def migrate_v6_to_v7(
    *,
    qualification_dir: Path,
    data_path: Path,
    candidate_manifest_path: Path,
    integrity_manifest_path: Path,
) -> dict[str, Any]:
    """Reuse compatible v6 trials and rebuild v7 artifacts without API calls."""
    verify_integrity(integrity_manifest_path.parent)
    rows, candidate_manifest, integrity_manifest = load_candidate_rows(
        data_path,
        candidate_manifest_path,
        integrity_manifest_path,
    )
    if integrity_manifest is None:
        raise RuntimeError("AWM qualification migration requires an integrity manifest")
    if sha256_file(data_path) != integrity_manifest.get("prefilter_data_sha256"):
        raise RuntimeError("AWM qualification migration requires the hash-bound prefilter pool")
    if integrity_manifest.get("prefilter_protocol_version") != PREFILTER_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification migration requires materialized prefilter artifacts")

    config_path = qualification_dir / "config.json"
    qualification_manifest_path = qualification_dir / "qualification_manifest.json"
    trials_path = qualification_dir / "trials.jsonl"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    if old_config.get("protocol_version") == QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification cache is already on the current protocol")
    if old_config.get("protocol_version") != LEGACY_QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("unsupported AWM qualification source protocol")
    old_manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("protocol_version") != LEGACY_QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError("AWM qualification config/manifest protocol mismatch")
    if sha256_file(trials_path) != old_manifest.get("trials_sha256"):
        raise RuntimeError("legacy AWM qualification trials hash mismatch")

    old_trials = _load_jsonl(trials_path)
    old_candidate_ids = {str(task_id) for task_id in old_config["candidate_task_ids"]}
    validate_trial_records(old_trials, old_candidate_ids)
    new_candidate_ids = {row["task_id"] for row in rows}
    rejected_prefilter_ids = set(integrity_manifest["rejected_prefilter_task_ids"])
    removed_trials = [record for record in old_trials if record["task_id"] not in new_candidate_ids]
    removed_task_ids = sorted({str(record["task_id"]) for record in removed_trials})
    if not set(removed_task_ids).issubset(rejected_prefilter_ids):
        raise RuntimeError("legacy trials outside the new pool are not rejected-prefilter tasks")
    retained_trials = [record for record in old_trials if record["task_id"] in new_candidate_ids]

    source_manifest_sha256 = sha256_file(qualification_manifest_path)
    archive_subdir = Path("protocol_migrations") / "v6_to_v7" / source_manifest_sha256
    archive_dir = qualification_dir / archive_subdir
    if archive_dir.exists():
        raise FileExistsError(f"qualification migration archive already exists: {archive_dir}")
    archive_dir.mkdir(parents=True)
    for source in sorted(qualification_dir.iterdir()):
        if source.name == "protocol_migrations":
            continue
        destination = archive_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    archive_manifest = {
        "protocol_version": MIGRATION_PROTOCOL_VERSION,
        "from_qualification_protocol": LEGACY_QUALIFICATION_PROTOCOL_VERSION,
        "to_qualification_protocol": QUALIFICATION_PROTOCOL_VERSION,
        "source_qualification_manifest_sha256": source_manifest_sha256,
        "files": _archive_inventory(archive_dir),
    }
    archive_manifest_path = archive_dir / "archive_manifest.json"
    archive_manifest_path.write_text(
        json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    new_identity = dict(old_config)
    new_identity.update(
        {
            "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "integrity_manifest_sha256": sha256_file(integrity_manifest_path),
            "candidate_data_sha256": sha256_file(data_path),
            "candidate_task_ids": [row["task_id"] for row in rows],
            "candidate_scope": "cheap_deterministic_prefilter_non_quarantine",
            "prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
            "final_task_statuses": list(FINAL_TASK_STATUSES),
        }
    )
    migration_provenance = {
        "protocol_version": MIGRATION_PROTOCOL_VERSION,
        "from_qualification_protocol": LEGACY_QUALIFICATION_PROTOCOL_VERSION,
        "to_qualification_protocol": QUALIFICATION_PROTOCOL_VERSION,
        "archive_subdir": str(archive_subdir),
        "archive_manifest_sha256": sha256_file(archive_manifest_path),
        "source_qualification_manifest_sha256": source_manifest_sha256,
        "source_trials_sha256": old_manifest["trials_sha256"],
        "source_candidate_tasks": len(old_candidate_ids),
        "target_candidate_tasks": len(rows),
        "source_trial_records": len(old_trials),
        "retained_trial_records": len(retained_trials),
        "removed_trial_records": len(removed_trials),
        "removed_trial_task_ids": removed_task_ids,
        "api_calls": 0,
    }

    for legacy_name in ("integrity_feedback.json", "source_integrity_snapshot"):
        legacy_path = qualification_dir / legacy_name
        if legacy_path.is_dir():
            shutil.rmtree(legacy_path)
        elif legacy_path.exists():
            legacy_path.unlink()
    config_path.write_text(
        json.dumps(new_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = write_qualification_artifacts(
        rows=rows,
        candidate_manifest=candidate_manifest,
        integrity_manifest=integrity_manifest,
        identity=new_identity,
        trial_records=retained_trials,
        output_dir=qualification_dir,
        provider_identity=provider_identity_from_trials(retained_trials),
        live_usage={"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        migration_provenance=migration_provenance,
    )
    return {
        "migration": migration_provenance,
        "qualification": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--integrity-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = migrate_v6_to_v7(
        qualification_dir=args.qualification_dir,
        data_path=args.data,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
