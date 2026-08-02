"""Feed deterministic qualification-time environment defects into integrity quarantine."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .integrity import verify_integrity
from .native_rollout import sha256_file

QUALIFICATION_FEEDBACK_PROTOCOL_VERSION = 1
_QUARANTINE_REASON = "qualification:repeated_judge_confirmed_environment_error"
_HTTP_ERROR_RE = re.compile(r"Status code:\s*([45]\d\d)")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}") from exc
    return records


def deterministic_environment_error(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return conservative evidence for a repeated, judge-confirmed environment defect."""
    if record.get("status") != "infrastructure_exhausted":
        return None
    errors = [str(item) for item in record.get("errors") or []]
    server_error_attempts = sum("verifier infrastructure error: 'server_error'" in item for item in errors)
    if int(record.get("infrastructure_attempts", 0) or 0) < 3 or server_error_attempts < 2:
        return None
    result = record.get("last_result")
    if not isinstance(result, Mapping) or result.get("reward_type") != "server_error":
        return None
    verify_result = result.get("verify_result")
    if not isinstance(verify_result, Mapping):
        return None
    judge = verify_result.get("llm_judge")
    if not isinstance(judge, Mapping):
        return None
    if str(judge.get("classification") or "").lower() != "server_error":
        return None
    confidence = judge.get("confidence_score") or []
    try:
        server_error_confidence = float(confidence[2])
    except (IndexError, TypeError, ValueError):
        return None
    if server_error_confidence < 80:
        return None

    tool_errors = []
    status_codes = []
    for step in result.get("trajectory") or []:
        if not isinstance(step, Mapping) or not step.get("tool_response_is_error"):
            continue
        response = str(step.get("tool_response") or "")
        codes = [int(item) for item in _HTTP_ERROR_RE.findall(response)]
        if not codes:
            continue
        status_codes.extend(codes)
        tool_errors.append(
            {
                "decision": int(step.get("decision", 0) or 0),
                "parsed_action": str(step.get("parsed_action") or "")[:1000],
                "status_codes": codes,
                "tool_response": response[:2000],
            }
        )
    if not tool_errors:
        return None
    return {
        "task_id": str(record["task_id"]),
        "trial_index": int(record["trial_index"]),
        "seed": int(record["seed"]),
        "infrastructure_attempts": int(record["infrastructure_attempts"]),
        "server_error_attempts": server_error_attempts,
        "judge_server_error_confidence": server_error_confidence,
        "http_status_codes": status_codes,
        "tool_errors": tool_errors,
        "judge_reasoning": str(judge.get("reasoning") or "")[:4000],
        "judge_evidence": judge.get("evidence") or {},
    }


def build_qualification_feedback(
    config: Mapping[str, Any],
    trial_records: Sequence[Mapping[str, Any]],
    *,
    trials_sha256: str,
) -> dict[str, Any]:
    candidate_ids = [str(item) for item in config.get("candidate_task_ids") or []]
    allowed = set(candidate_ids)
    records_by_id = {}
    infrastructure_ids = set()
    for record in trial_records:
        task_id = str(record.get("task_id") or "")
        if task_id not in allowed:
            raise RuntimeError(f"qualification feedback found unknown task ID {task_id!r}")
        if record.get("status") != "infrastructure_exhausted":
            continue
        infrastructure_ids.add(task_id)
        evidence = deterministic_environment_error(record)
        if evidence is not None:
            records_by_id[task_id] = evidence
    deterministic_ids = [task_id for task_id in candidate_ids if task_id in records_by_id]
    pending_ids = [task_id for task_id in candidate_ids if task_id in infrastructure_ids and task_id not in records_by_id]
    return {
        "protocol_version": QUALIFICATION_FEEDBACK_PROTOCOL_VERSION,
        "kind": "awm_qualification_integrity_feedback",
        "qualification_protocol_version": config.get("protocol_version"),
        "qualification_config_sha256": sha256_file_from_json(config),
        "qualification_trials_sha256": trials_sha256,
        "source_integrity_manifest_sha256": config.get("integrity_manifest_sha256"),
        "source_candidate_data_sha256": config.get("candidate_data_sha256"),
        "classification_policy": {
            "minimum_infrastructure_attempts": 3,
            "minimum_server_error_attempts": 2,
            "required_judge_classification": "server_error",
            "minimum_judge_server_error_confidence": 80,
            "required_trajectory_evidence": "at least one tool response with HTTP 4xx/5xx",
            "timeouts_only": "infrastructure_pending",
        },
        "deterministic_quarantine_task_ids": deterministic_ids,
        "infrastructure_pending_task_ids": pending_ids,
        "records": [records_by_id[task_id] for task_id in deterministic_ids],
    }


def sha256_file_from_json(value: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_qualification_feedback(
    output_dir: Path,
    config: Mapping[str, Any],
    trial_records: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    trials_path = output_dir / "trials.jsonl"
    feedback = build_qualification_feedback(
        config,
        trial_records,
        trials_sha256=sha256_file(trials_path),
    )
    path = output_dir / "integrity_feedback.json"
    _write_json(path, feedback)
    return path, feedback


def ensure_qualification_feedback(qualification_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = qualification_dir / "config.json"
    manifest_path = qualification_dir / "qualification_manifest.json"
    trials_path = qualification_dir / "trials.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != config.get("protocol_version"):
        raise RuntimeError("qualification manifest/config protocol mismatch")
    if sha256_file(trials_path) != manifest.get("trials_sha256"):
        raise RuntimeError("qualification trials hash mismatch")
    for field in ("integrity_manifest_sha256", "candidate_data_sha256", "candidate_manifest_sha256"):
        if manifest.get(field) != config.get(field):
            raise RuntimeError(f"qualification manifest/config mismatch: {field}")
    trial_records = _load_jsonl(trials_path)
    feedback_path, feedback = write_qualification_feedback(qualification_dir, config, trial_records)
    manifest.update(
        {
            "integrity_feedback_sha256": sha256_file(feedback_path),
            "deterministic_environment_quarantine_task_ids": feedback["deterministic_quarantine_task_ids"],
            "qualification_infrastructure_pending_task_ids": feedback["infrastructure_pending_task_ids"],
        }
    )
    _write_json(manifest_path, manifest)
    return feedback, manifest


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing provenance snapshot differs: {destination}")
        return
    shutil.copy2(source, destination)


def _source_integrity_files(
    integrity_dir: Path,
    qualification_dir: Path,
    source_manifest_sha256: str,
    current_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    current_manifest_path = integrity_dir / "integrity_manifest.json"
    current_data_path = integrity_dir / "awm_integrity_filtered.parquet"
    if sha256_file(current_manifest_path) == source_manifest_sha256:
        return current_manifest_path, current_data_path
    snapshot = qualification_dir / "source_integrity_snapshot"
    snapshot_manifest = snapshot / "integrity_manifest.json"
    snapshot_data = snapshot / "awm_integrity_filtered.parquet"
    if snapshot_manifest.is_file() and sha256_file(snapshot_manifest) == source_manifest_sha256:
        source_manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        if snapshot_data.is_file() and sha256_file(snapshot_data) == source_manifest.get("filtered_data_sha256"):
            return snapshot_manifest, snapshot_data
    for provenance in current_manifest.get("qualification_feedback_provenance") or []:
        if provenance.get("qualification_source_integrity_manifest_sha256") != source_manifest_sha256:
            continue
        archive = integrity_dir / str(provenance["archive_subdir"])
        source_manifest_path = archive / "qualification_source_integrity_manifest.json"
        source_data_path = archive / "qualification_source_filtered.parquet"
        if source_manifest_path.is_file() and sha256_file(source_manifest_path) == source_manifest_sha256:
            return source_manifest_path, source_data_path
    raise RuntimeError("qualification source integrity snapshot is unavailable")


def _feedback_lineage_accepts(current_manifest: Mapping[str, Any], source_sha256: str, current_sha256: str) -> bool:
    if current_sha256 == source_sha256:
        return True
    return any(item.get("qualification_source_integrity_manifest_sha256") == source_sha256 for item in current_manifest.get("qualification_feedback_provenance") or [])


def apply_qualification_feedback(integrity_dir: Path, qualification_dir: Path) -> dict[str, Any]:
    """Apply deterministic feedback and preserve source snapshots for strict resume."""
    verify_integrity(integrity_dir)
    feedback, qualification_manifest = ensure_qualification_feedback(qualification_dir)
    feedback_path = qualification_dir / "integrity_feedback.json"
    qualification_manifest_path = qualification_dir / "qualification_manifest.json"
    qualification_manifest_sha256 = sha256_file(qualification_manifest_path)
    current_manifest_path = integrity_dir / "integrity_manifest.json"
    current_data_path = integrity_dir / "awm_integrity_filtered.parquet"
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    current_manifest_sha256 = sha256_file(current_manifest_path)
    source_manifest_sha256 = str(feedback.get("source_integrity_manifest_sha256") or "")
    if not source_manifest_sha256 or not _feedback_lineage_accepts(
        current_manifest,
        source_manifest_sha256,
        current_manifest_sha256,
    ):
        raise RuntimeError("qualification feedback does not descend from the current integrity lineage")

    records = _load_jsonl(integrity_dir / "integrity_audit.jsonl")
    records_by_id = {str(record["task_id"]): record for record in records}
    feedback_by_id = {str(record["task_id"]): record for record in feedback["records"]}
    requested_ids = [str(item) for item in feedback["deterministic_quarantine_task_ids"]]
    applied_ids = []
    for task_id in requested_ids:
        record = records_by_id.get(task_id)
        if record is None:
            raise RuntimeError(f"qualification feedback task is absent from integrity audit: {task_id}")
        existing = record.get("qualification_feedback") or []
        if record.get("status") == "quarantine" and any(item.get("reason") == _QUARANTINE_REASON for item in existing):
            continue
        if record.get("status") != "pass":
            raise RuntimeError(f"qualification feedback cannot change {task_id} from {record.get('status')!r}")
        record["status"] = "quarantine"
        record["status_reasons"] = sorted(set(record.get("status_reasons") or []) | {_QUARANTINE_REASON})
        record["qualification_feedback"] = [
            *existing,
            {
                "protocol_version": QUALIFICATION_FEEDBACK_PROTOCOL_VERSION,
                "qualification_manifest_sha256": qualification_manifest_sha256,
                "qualification_trials_sha256": feedback["qualification_trials_sha256"],
                "reason": _QUARANTINE_REASON,
                "evidence": feedback_by_id[task_id],
            },
        ]
        applied_ids.append(task_id)

    if not applied_ids:
        return {
            "applied": 0,
            "deterministic_quarantine": len(requested_ids),
            "infrastructure_pending": len(feedback["infrastructure_pending_task_ids"]),
            "counts": current_manifest["counts"],
        }

    source_manifest_path, source_data_path = _source_integrity_files(
        integrity_dir,
        qualification_dir,
        source_manifest_sha256,
        current_manifest,
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(source_data_path) != source_manifest.get("filtered_data_sha256"):
        raise RuntimeError("qualification source filtered-data hash mismatch")
    resume_snapshot = qualification_dir / "source_integrity_snapshot"
    _copy_exact(source_manifest_path, resume_snapshot / "integrity_manifest.json")
    _copy_exact(source_data_path, resume_snapshot / "awm_integrity_filtered.parquet")

    archive_subdir = f"qualification_feedback/{qualification_manifest_sha256}"
    archive = integrity_dir / archive_subdir
    _copy_exact(current_manifest_path, archive / "base_integrity_manifest.json")
    _copy_exact(current_data_path, archive / "base_filtered.parquet")
    _copy_exact(source_manifest_path, archive / "qualification_source_integrity_manifest.json")
    _copy_exact(source_data_path, archive / "qualification_source_filtered.parquet")
    _copy_exact(qualification_manifest_path, archive / "qualification_manifest.json")
    _copy_exact(feedback_path, archive / "integrity_feedback.json")

    _write_jsonl(integrity_dir / "integrity_audit.jsonl", records)
    status_ids = {status: [str(record["task_id"]) for record in records if record.get("status") == status] for status in ("quarantine", "needs_review", "infrastructure_pending")}
    for status, name in (
        ("quarantine", "quarantine_task_ids.json"),
        ("needs_review", "needs_review_task_ids.json"),
        ("infrastructure_pending", "infrastructure_pending_task_ids.json"),
    ):
        _write_json(integrity_dir / name, status_ids[status])

    removed = set(applied_ids)
    frame = pd.read_parquet(current_data_path)
    keep = [str(extra["task_id"]) not in removed for extra in frame["extra_info"].tolist()]
    filtered = frame.loc[keep].reset_index(drop=True)
    temporary_parquet = current_data_path.with_name(f".{current_data_path.name}.tmp.parquet")
    filtered.to_parquet(temporary_parquet, index=False)
    temporary_parquet.replace(current_data_path)
    filtered_ids = [str(extra["task_id"]) for extra in filtered["extra_info"].tolist()]

    counts = Counter(str(record["status"]) for record in records)
    provenance = {
        "protocol_version": QUALIFICATION_FEEDBACK_PROTOCOL_VERSION,
        "archive_subdir": archive_subdir,
        "prior_integrity_manifest_sha256": current_manifest_sha256,
        "prior_filtered_data_sha256": current_manifest["filtered_data_sha256"],
        "qualification_source_integrity_manifest_sha256": source_manifest_sha256,
        "qualification_source_filtered_data_sha256": source_manifest["filtered_data_sha256"],
        "qualification_manifest_sha256": qualification_manifest_sha256,
        "qualification_trials_sha256": feedback["qualification_trials_sha256"],
        "integrity_feedback_sha256": sha256_file(feedback_path),
        "applied_task_ids": applied_ids,
    }
    current_manifest.update(
        {
            "counts": dict(sorted(counts.items())),
            "filtered_task_ids": filtered_ids,
            "integrity_audit_sha256": sha256_file(integrity_dir / "integrity_audit.jsonl"),
            "filtered_data_sha256": sha256_file(current_data_path),
            "quarantine_task_ids_sha256": sha256_file(integrity_dir / "quarantine_task_ids.json"),
            "needs_review_task_ids_sha256": sha256_file(integrity_dir / "needs_review_task_ids.json"),
            "infrastructure_pending_task_ids_sha256": sha256_file(integrity_dir / "infrastructure_pending_task_ids.json"),
            "qualification_feedback_quarantine_task_ids": [str(record["task_id"]) for record in records if record.get("qualification_feedback")],
            "qualification_feedback_provenance": [
                *(current_manifest.get("qualification_feedback_provenance") or []),
                provenance,
            ],
        }
    )
    _write_json(current_manifest_path, current_manifest)
    verify_integrity(integrity_dir)
    return {
        "applied": len(applied_ids),
        "applied_task_ids": applied_ids,
        "deterministic_quarantine": len(requested_ids),
        "infrastructure_pending": len(feedback["infrastructure_pending_task_ids"]),
        "counts": current_manifest["counts"],
    }


def verify_feedback_provenance(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    records = {str(record["task_id"]): record for record in _load_jsonl(output_dir / "integrity_audit.jsonl")}
    for item in manifest.get("qualification_feedback_provenance") or []:
        if item.get("protocol_version") != QUALIFICATION_FEEDBACK_PROTOCOL_VERSION:
            raise RuntimeError("AWM qualification-feedback provenance protocol mismatch")
        archive_subdir = str(item.get("archive_subdir") or "")
        archive = output_dir / archive_subdir
        if archive.parent != output_dir / "qualification_feedback":
            raise RuntimeError("AWM qualification-feedback archive path is invalid")
        paths = {
            "prior_integrity_manifest_sha256": archive / "base_integrity_manifest.json",
            "prior_filtered_data_sha256": archive / "base_filtered.parquet",
            "qualification_source_integrity_manifest_sha256": archive / "qualification_source_integrity_manifest.json",
            "qualification_source_filtered_data_sha256": archive / "qualification_source_filtered.parquet",
            "qualification_manifest_sha256": archive / "qualification_manifest.json",
            "integrity_feedback_sha256": archive / "integrity_feedback.json",
        }
        for field, path in paths.items():
            if sha256_file(path) != item.get(field):
                raise RuntimeError(f"AWM qualification-feedback provenance hash mismatch: {path}")
        for task_id in item.get("applied_task_ids") or []:
            record = records.get(str(task_id))
            if record is None or record.get("status") != "quarantine":
                raise RuntimeError(f"AWM qualification-feedback quarantine is missing: {task_id}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity-dir", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            apply_qualification_feedback(args.integrity_dir, args.qualification_dir),
            indent=2,
            sort_keys=True,
        )
    )
