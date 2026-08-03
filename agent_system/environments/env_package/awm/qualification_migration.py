"""Offline migration and candidate-pool rebase for AWM qualification caches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .integrity import PREFILTER_PROTOCOL_VERSION, verify_integrity
from .native_rollout import sha256_file
from .qualification import (
    FINAL_TASK_STATUSES,
    QUALIFICATION_PROTOCOL_VERSION,
    _load_jsonl,
    load_candidate_rows,
    provider_identity_from_trials,
    qualification_rollout_protocol,
    validate_trial_records,
    write_qualification_artifacts,
)

LEGACY_QUALIFICATION_PROTOCOL_VERSIONS = (6, 7)
MIGRATION_PROTOCOL_VERSION = 2


def _archive_inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "archive_manifest.json"}


def _validate_source_artifacts(
    qualification_dir: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    for key, value in config.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"legacy AWM qualification manifest/config mismatch: {key}")
    trials_path = qualification_dir / "trials.jsonl"
    if sha256_file(trials_path) != manifest.get("trials_sha256"):
        raise RuntimeError("legacy AWM qualification trials hash mismatch")
    artifacts = (
        (
            "awm_expert_qualified_all.parquet",
            "qualified_all_sha256",
            "qualified_task_ids",
        ),
        (
            "awm_expert_qualified_train_b8.parquet",
            "qualified_train_b8_sha256",
            "qualified_train_b8_task_ids",
        ),
    )
    for filename, hash_key, ids_key in artifacts:
        path = qualification_dir / filename
        expected_hash = manifest.get(hash_key)
        if expected_hash is None:
            if path.exists():
                raise RuntimeError(f"legacy AWM qualification has an unbound artifact: {filename}")
            continue
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"legacy AWM qualification artifact hash mismatch: {filename}")
        frame = pd.read_parquet(path)
        task_ids = [str(extra["task_id"]) for extra in frame["extra_info"].tolist()]
        if task_ids != manifest.get(ids_key):
            raise RuntimeError(f"legacy AWM qualification artifact task IDs mismatch: {filename}")
    diagnostic_path = qualification_dir / "qwen_diagnostic_manifest.json"
    expected_diagnostic_hash = manifest.get("qwen_diagnostic_manifest_sha256")
    if expected_diagnostic_hash is not None and (not diagnostic_path.is_file() or sha256_file(diagnostic_path) != expected_diagnostic_hash):
        raise RuntimeError("legacy AWM qualification diagnostic-manifest hash mismatch")


def _legacy_rollout_evidence(trials: list[Mapping[str, Any]]) -> dict[str, int]:
    protocol = qualification_rollout_protocol()
    max_decisions = 0
    max_prompt_tokens = 0
    trajectory_records = 0
    prompt_usage_records = 0
    for record in trials:
        result = record.get("result") or record.get("last_result")
        if not isinstance(result, Mapping):
            continue
        trajectory = result.get("trajectory") or []
        decisions = int(result.get("decisions", len(trajectory)) or 0)
        if decisions > protocol["max_decisions"]:
            raise RuntimeError("legacy AWM qualification trial exceeds the confirmed decision budget")
        max_decisions = max(max_decisions, decisions)
        for entry in trajectory:
            if not isinstance(entry, Mapping):
                continue
            trajectory_records += 1
            usage = entry.get("usage")
            if not isinstance(usage, Mapping) or usage.get("prompt_tokens") is None:
                continue
            prompt_tokens = int(usage["prompt_tokens"] or 0)
            if prompt_tokens > protocol["max_prompt_tokens"]:
                raise RuntimeError("legacy AWM qualification trial exceeds the confirmed prompt budget")
            prompt_usage_records += 1
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
    return {
        "max_observed_decisions": max_decisions,
        "max_observed_prompt_tokens": max_prompt_tokens,
        "trajectory_records": trajectory_records,
        "prompt_usage_records": prompt_usage_records,
    }


def migrate_qualification(
    *,
    qualification_dir: Path,
    data_path: Path,
    candidate_manifest_path: Path,
    integrity_manifest_path: Path,
    confirm_legacy_context: bool = False,
) -> dict[str, Any]:
    """Reuse compatible trials and rebuild v8 artifacts without API calls."""
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
    source_protocol = old_config.get("protocol_version")
    current_protocol_rebase = source_protocol == QUALIFICATION_PROTOCOL_VERSION
    if source_protocol not in LEGACY_QUALIFICATION_PROTOCOL_VERSIONS and not current_protocol_rebase:
        raise RuntimeError("unsupported AWM qualification source protocol")
    old_manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("protocol_version") != source_protocol:
        raise RuntimeError("AWM qualification config/manifest protocol mismatch")
    _validate_source_artifacts(qualification_dir, old_config, old_manifest)
    if not current_protocol_rebase and not confirm_legacy_context:
        raise RuntimeError("legacy AWM qualification migration requires explicit confirmation that retained trials used history_window=3 and the fixed 32k context protocol")

    old_trials = _load_jsonl(trials_path)
    old_candidate_order = [str(task_id) for task_id in old_config["candidate_task_ids"]]
    old_candidate_ids = set(old_candidate_order)
    validate_trial_records(old_trials, old_candidate_ids)
    source_provider_identity = provider_identity_from_trials(old_trials)
    if old_manifest.get("provider_identity") is not None and old_manifest.get("provider_identity") != source_provider_identity:
        raise RuntimeError("legacy AWM qualification provider identity mismatch")
    rollout_protocol = qualification_rollout_protocol()
    for key, expected in rollout_protocol.items():
        if current_protocol_rebase and old_config.get(key) != expected:
            raise RuntimeError(f"current AWM qualification rollout mismatch: {key}")
        if not current_protocol_rebase and key in old_config and old_config[key] != expected:
            raise RuntimeError(f"legacy AWM qualification rollout mismatch: {key}")
    rollout_evidence = _legacy_rollout_evidence(old_trials)

    new_candidate_order = [row["task_id"] for row in rows]
    new_candidate_ids = set(new_candidate_order)
    rejected_prefilter_ids = set(integrity_manifest["rejected_prefilter_task_ids"])
    if current_protocol_rebase and not new_candidate_ids.issubset(old_candidate_ids):
        raise RuntimeError("current AWM qualification rebase requires a candidate subset")
    if current_protocol_rebase and [task_id for task_id in old_candidate_order if task_id in new_candidate_ids] != new_candidate_order:
        raise RuntimeError("current AWM qualification rebase requires an ordered candidate subset")
    if source_protocol == 7:
        expected_bindings = {
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "integrity_manifest_sha256": sha256_file(integrity_manifest_path),
            "candidate_data_sha256": sha256_file(data_path),
            "candidate_task_ids": [row["task_id"] for row in rows],
            "candidate_scope": "cheap_deterministic_prefilter_non_quarantine",
            "prefilter_protocol_version": PREFILTER_PROTOCOL_VERSION,
        }
        for key, expected in expected_bindings.items():
            if old_config.get(key) != expected:
                raise RuntimeError(f"v7 AWM qualification source binding mismatch: {key}")

    removed_trials = [record for record in old_trials if record["task_id"] not in new_candidate_ids]
    removed_task_ids = sorted({str(record["task_id"]) for record in removed_trials})
    if not current_protocol_rebase and not set(removed_task_ids).issubset(rejected_prefilter_ids):
        raise RuntimeError("legacy trials outside the new pool are not rejected-prefilter tasks")
    if source_protocol == 7 and removed_trials:
        raise RuntimeError("v7 AWM qualification unexpectedly contains trials outside its bound pool")
    retained_trials = [record for record in old_trials if record["task_id"] in new_candidate_ids]

    source_manifest_sha256 = sha256_file(qualification_manifest_path)
    transition = f"v{source_protocol}_candidate_rebase" if current_protocol_rebase else f"v{source_protocol}_to_v{QUALIFICATION_PROTOCOL_VERSION}"
    archive_subdir = Path("protocol_migrations") / transition / source_manifest_sha256
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
        "from_qualification_protocol": source_protocol,
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
            **rollout_protocol,
        }
    )
    removed_candidate_ids = sorted(old_candidate_ids - new_candidate_ids)
    source_context_binding = (
        {
            "status": "manifest_bound",
            "basis": "source v8 config and manifest exactly bind the rollout protocol",
            "rollout_protocol": rollout_protocol,
            "artifact_evidence": rollout_evidence,
        }
        if current_protocol_rebase
        else {
            "status": "operator_confirmed",
            "basis": ("legacy qualification protocols did not serialize context constants; the operator confirmed the fixed implementation used for these trials"),
            "rollout_protocol": rollout_protocol,
            "artifact_evidence": rollout_evidence,
        }
    )
    migration_provenance = {
        "protocol_version": MIGRATION_PROTOCOL_VERSION,
        "from_qualification_protocol": source_protocol,
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
        "removed_candidate_tasks": len(removed_candidate_ids),
        "removed_candidate_task_ids": removed_candidate_ids,
        "source_context_binding": source_context_binding,
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
        provider_identity=source_provider_identity,
        live_usage={
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
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
    parser.add_argument(
        "--confirm-legacy-context",
        action="store_true",
        help=("Confirm that retained v6/v7 trials used history_window=3, max_decisions=20, and the fixed 32k context budget. Current-v8 candidate rebases do not require this flag. This performs no model calls."),
    )
    args = parser.parse_args()
    result = migrate_qualification(
        qualification_dir=args.qualification_dir,
        data_path=args.data,
        candidate_manifest_path=args.candidate_manifest,
        integrity_manifest_path=args.integrity_manifest,
        confirm_legacy_context=args.confirm_legacy_context,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
