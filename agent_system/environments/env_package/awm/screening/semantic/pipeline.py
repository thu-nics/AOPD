"""Resumable queue and final-pool pipeline for AWM semantic task review."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ...data.integrity import INTEGRITY_PROTOCOL_VERSION
from ...data.prepare import EXPECTED_SOURCE_SHA256
from ...data.selection import validate_base_manifest
from ...runtime.logical_time import fetch_server_protocol
from ...runtime.rollout import sha256_file
from ..common import load_candidate_rows
from ..expert import (
    EXPERT_SCREENING_PROTOCOL_VERSION,
    FINAL_TASK_STATUSES,
    validate_trial_records,
)
from .evidence import (
    ReplayDriftError,
    SourceCatalog,
    build_evidence_packet,
    load_jsonl,
    validate_evidence_packet,
)
from .judgments import reviewer_consensus, validate_judgment
from .prompts import REVIEWER_PROMPT_PROTOCOL_VERSION, render_review_prompt

SEMANTIC_AUDIT_PROTOCOL_VERSION = 2
SUCCESS_CONTROL_FRACTION = 0.10
POOL_FILENAME = "awm_verified_task_pool.parquet"
POOL_MANIFEST_FILENAME = "verified_pool_manifest.json"
PARTITIONS = ("included", "excluded", "pending", "out_of_context")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _stable_rank(value: str) -> tuple[str, str]:
    return hashlib.sha256(value.encode()).hexdigest(), value


def _trial_result(trial: Mapping[str, Any]) -> Mapping[str, Any]:
    return trial.get("result") or trial.get("last_result") or {}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid finalized JSONL record {line_number}: {path}") from exc


def _compact_plan_trial(record: Mapping[str, Any]) -> dict[str, Any]:
    result = _trial_result(record)
    return {
        "task_id": str(record["task_id"]),
        "status": str(record["status"]),
        "result": {
            "decisions": int(result.get("decisions") or 0),
            "trajectory": [{"runtime_error_signature": entry.get("runtime_error_signature")} for entry in result.get("trajectory") or [] if entry.get("runtime_error_signature")],
        },
    }


def _jsonl_offsets(path: Path) -> dict[str, int]:
    offsets = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id") or "")
            if not task_id or task_id in offsets:
                raise RuntimeError(f"invalid or duplicate expert trial while indexing: {task_id!r}")
            offsets[task_id] = offset
    return offsets


def _read_jsonl_at(handle, offset: int) -> dict[str, Any]:
    handle.seek(offset)
    return json.loads(handle.readline())


def _success_stratum(row: Mapping[str, Any], trial: Mapping[str, Any], token_quartile: int) -> str:
    decisions = int(_trial_result(trial).get("decisions") or 0)
    decision_band = "1-5" if decisions <= 5 else "6-10" if decisions <= 10 else "11-20"
    return f"prompt_q{token_quartile}:decisions_{decision_band}"


def select_success_controls(
    success_rows: Sequence[Mapping[str, Any]],
    trials_by_id: Mapping[str, Mapping[str, Any]],
    *,
    fraction: float = SUCCESS_CONTROL_FRACTION,
) -> tuple[list[str], dict[str, str]]:
    """Select deterministic 10% controls within prompt-length/decision strata."""
    if not 0 < fraction <= 1:
        raise ValueError("control fraction must be in (0, 1]")
    ordered_tokens = sorted((int(row["native_prompt_tokens"]), str(row["task_id"])) for row in success_rows)
    quartile_by_id = {task_id: min(3, (rank * 4) // max(1, len(ordered_tokens))) for rank, (_, task_id) in enumerate(ordered_tokens)}
    strata: dict[str, list[str]] = defaultdict(list)
    stratum_by_id = {}
    for row in success_rows:
        task_id = str(row["task_id"])
        stratum = _success_stratum(row, trials_by_id[task_id], quartile_by_id[task_id])
        strata[stratum].append(task_id)
        stratum_by_id[task_id] = stratum
    selected = []
    for stratum in sorted(strata):
        task_ids = sorted(strata[stratum], key=_stable_rank)
        selected.extend(task_ids[: max(1, math.ceil(len(task_ids) * fraction))])
    selected_set = set(selected)
    return [str(row["task_id"]) for row in success_rows if str(row["task_id"]) in selected_set], stratum_by_id


def verify_finalized_screening(
    *,
    screening_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = screening_dir / "config.json"
    trials_path = screening_dir / "trials.jsonl"
    manifest_path = screening_dir / "screening_manifest.json"
    for path in (config_path, trials_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing finalized expert-screening artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != EXPERT_SCREENING_PROTOCOL_VERSION:
        raise RuntimeError("expert-screening protocol mismatch")
    if manifest.get("kind") != "awm_one_pass_expert_screening":
        raise RuntimeError("expert-screening manifest kind mismatch")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise RuntimeError("expert-screening config hash mismatch")
    if sha256_file(trials_path) != manifest.get("trials_sha256"):
        raise RuntimeError("expert-screening trials hash mismatch")
    if any(manifest.get(key) != value for key, value in config.items()):
        raise RuntimeError("expert-screening manifest/config mismatch")
    candidate_ids = [str(row["task_id"]) for row in rows]
    if config.get("candidate_task_ids") != candidate_ids:
        raise RuntimeError("expert-screening candidate task IDs mismatch")
    candidate_set = set(candidate_ids)
    compact_trials = []
    raw_statuses = {}
    for record in _iter_jsonl(trials_path):
        validate_trial_records([record], candidate_set)
        task_id = str(record["task_id"])
        if task_id in raw_statuses:
            raise RuntimeError(f"duplicate expert-screening trial for {task_id!r}")
        raw_statuses[task_id] = str(record["status"])
        compact_trials.append(_compact_plan_trial(record))
    resolution = {
        "success": "accepted_success",
        "policy_failure": "accepted_policy_failure",
        "environment_failure": "rejected_environment",
        "infrastructure_exhausted": "infrastructure_pending",
    }
    statuses = {task_id: resolution[raw_statuses[task_id]] if task_id in raw_statuses else "pending" for task_id in candidate_ids}
    if statuses != manifest.get("task_status"):
        raise RuntimeError("expert-screening task-status derivation mismatch")
    counts = {status: sum(value == status for value in statuses.values()) for status in FINAL_TASK_STATUSES}
    if counts != manifest.get("counts"):
        raise RuntimeError("expert-screening counts mismatch")
    if counts.get("pending", 0):
        raise RuntimeError(f"expert screening is not finalized: {counts['pending']} tasks remain pending")
    return manifest, compact_trials


def _source_cohort_keys(catalog: SourceCatalog, row: Mapping[str, Any]) -> list[str]:
    hashes = catalog.task_cohort_hashes(
        str(row["scenario"]),
        int(row["task_idx"]),
    )
    return [
        f"environment_source:{hashes['environment']}",
        f"code_verifier:{hashes['code_verifier']}",
    ]


def _plan_identity(
    *,
    data: Path,
    candidate_manifest: Path,
    integrity_manifest: Path,
    screening_dir: Path,
    awm_data_dir: Path,
    screening_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_hashes = {filename: sha256_file(awm_data_dir / filename) for filename in sorted(EXPECTED_SOURCE_SHA256)}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("semantic audit AWM sources do not match the pinned revision")
    return {
        "protocol_version": SEMANTIC_AUDIT_PROTOCOL_VERSION,
        "kind": "awm_codex_semantic_audit_plan",
        "reviewer_prompt_protocol_version": REVIEWER_PROMPT_PROTOCOL_VERSION,
        "success_control_fraction": SUCCESS_CONTROL_FRACTION,
        "review_policy": "all policy/environment failures plus deterministic stratified 10% successes; independent A/B review",
        "data_path": str(data.resolve()),
        "data_sha256": sha256_file(data),
        "candidate_manifest_path": str(candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "integrity_manifest_path": str(integrity_manifest.resolve()),
        "integrity_manifest_sha256": sha256_file(integrity_manifest),
        "screening_dir": str(screening_dir.resolve()),
        "screening_manifest_sha256": sha256_file(screening_dir / "screening_manifest.json"),
        "screening_trials_sha256": str(screening_manifest["trials_sha256"]),
        "awm_data_dir": str(awm_data_dir.resolve()),
        "awm_source_sha256": source_hashes,
        "awm_logical_time": screening_manifest.get("awm_logical_time"),
        "replay_seed": 300,
        "replay_verifier_mode": "code",
        "replay_concurrency": 1,
    }


def _load_evidence_packet(
    output_dir: Path,
    plan: Mapping[str, Any],
    task_id: str,
) -> tuple[Path, dict[str, Any]]:
    path = output_dir / "evidence" / f"{task_id.replace(':', '__')}.json"
    packet = json.loads(path.read_text(encoding="utf-8"))
    return path, validate_evidence_packet(
        packet,
        output_dir=output_dir,
        task_id=task_id,
        screening_manifest_sha256=str(plan["screening_manifest_sha256"]),
    )


def _validated_judgment_file(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    task_id: str,
    slot: str,
) -> tuple[Path, dict[str, Any]]:
    evidence_path, packet = _load_evidence_packet(output_dir, plan, task_id)
    path = output_dir / "judgments" / slot / f"{task_id.replace(':', '__')}.json"
    value = validate_judgment(
        json.loads(path.read_text(encoding="utf-8")),
        task_id=task_id,
        slot=slot,
        evidence_sha256=sha256_file(evidence_path),
        allowed_cohort_keys=list(packet["cohort_keys"]),
    )
    for pointer in value["evidence_refs"]:
        _resolve_json_pointer(packet, pointer)
    return path, value


def _rebuild_queue(
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    persist: bool = True,
    write_prompts: bool = True,
) -> list[dict[str, Any]]:
    records = []
    for target in plan["targets"]:
        task_id = str(target["task_id"])
        evidence_path = output_dir / "evidence" / f"{task_id.replace(':', '__')}.json"
        evidence_error = None
        try:
            valid_evidence_path, _ = _load_evidence_packet(output_dir, plan, task_id)
            evidence_sha = sha256_file(valid_evidence_path)
        except Exception as exc:
            evidence_sha = None
            if evidence_path.exists():
                evidence_error = f"{type(exc).__name__}: {exc}"
        for slot in ("A", "B"):
            judgment_path = output_dir / "judgments" / slot / f"{task_id.replace(':', '__')}.json"
            judgment_error = None
            if evidence_sha is None:
                status = "capture_invalid" if evidence_error else "capture_pending"
            elif not judgment_path.is_file():
                status = "ready"
            else:
                try:
                    _validated_judgment_file(
                        output_dir=output_dir,
                        plan=plan,
                        task_id=task_id,
                        slot=slot,
                    )
                    status = "complete"
                except Exception as exc:
                    status = "invalid"
                    judgment_error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "queue_id": f"{task_id}:{slot}",
                    "task_id": task_id,
                    "review_slot": slot,
                    "target_basis": target["basis"],
                    "success_control": bool(target.get("success_control")),
                    "evidence_path": str(evidence_path.resolve()),
                    "evidence_sha256": evidence_sha,
                    "judgment_path": str(judgment_path.resolve()),
                    "status": status,
                    "validation_error": judgment_error or evidence_error,
                }
            )
    if persist:
        _write_jsonl(output_dir / "review_queue.jsonl", records)
    if not write_prompts:
        return records
    prompt_dir = output_dir / "prompts"
    for record in records:
        prompt_path = prompt_dir / record["review_slot"] / f"{record['task_id'].replace(':', '__')}.txt"
        if record["evidence_sha256"] is not None:
            _write_text_if_changed(
                prompt_path,
                render_review_prompt(record, Path(record["evidence_path"])),
            )
        elif prompt_path.exists():
            prompt_path.unlink()
    return records


def create_plan(args) -> dict[str, Any]:
    rows, _, integrity = load_candidate_rows(args.data, args.candidate_manifest, args.integrity_manifest)
    if integrity is None or integrity.get("protocol_version") != INTEGRITY_PROTOCOL_VERSION:
        raise RuntimeError("semantic audit requires the current hash-bound integrity pool")
    screening_manifest, trials = verify_finalized_screening(screening_dir=args.screening_dir, rows=rows)
    expected_screening_inputs = {
        "candidate_data_sha256": sha256_file(args.data),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "integrity_manifest_sha256": sha256_file(args.integrity_manifest),
    }
    for field, expected in expected_screening_inputs.items():
        if screening_manifest.get(field) != expected:
            raise RuntimeError(f"expert screening input provenance mismatch: {field}")
    identity = _plan_identity(
        data=args.data,
        candidate_manifest=args.candidate_manifest,
        integrity_manifest=args.integrity_manifest,
        screening_dir=args.screening_dir,
        awm_data_dir=args.awm_data_dir,
        screening_manifest=screening_manifest,
    )
    output_dir = args.output_dir
    plan_path = output_dir / "review_plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if existing.get(key) != value:
                raise RuntimeError(f"semantic audit plan identity mismatch: {key}")
        _rebuild_queue(output_dir, existing)
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to create a semantic plan in non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_by_id = {str(record["task_id"]): record for record in trials}
    policy_ids = [str(row["task_id"]) for row in rows if trials_by_id[str(row["task_id"])]["status"] == "policy_failure"]
    environment_ids = [str(row["task_id"]) for row in rows if trials_by_id[str(row["task_id"])]["status"] == "environment_failure"]
    success_rows = [row for row in rows if trials_by_id[str(row["task_id"])]["status"] == "success"]
    control_ids, strata = select_success_controls(success_rows, trials_by_id)
    target_meta = {
        **{task_id: {"basis": "policy_failure", "success_control": False} for task_id in policy_ids},
        **{task_id: {"basis": "environment_failure", "success_control": False} for task_id in environment_ids},
        **{
            task_id: {
                "basis": f"success_control:{strata[task_id]}",
                "success_control": True,
            }
            for task_id in control_ids
        },
    }
    catalog = SourceCatalog(args.awm_data_dir)
    cohort_task_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        task_id = str(row["task_id"])
        trial = trials_by_id[task_id]
        if screening_manifest["task_status"][task_id] != "infrastructure_pending":
            for key in _source_cohort_keys(catalog, row):
                cohort_task_ids[key].append(task_id)
        for entry in _trial_result(trial).get("trajectory") or []:
            if entry.get("runtime_error_signature"):
                cohort_task_ids[f"runtime_error:{entry['runtime_error_signature']}"].append(task_id)
    plan = {
        **identity,
        "candidate_task_ids": [str(row["task_id"]) for row in rows],
        "screening_task_status": screening_manifest["task_status"],
        "targets": [{"task_id": str(row["task_id"]), **target_meta[str(row["task_id"])]} for row in rows if str(row["task_id"]) in target_meta],
        "cohort_task_ids": {key: values for key, values in sorted(cohort_task_ids.items())},
        "initial_target_counts": {
            "policy_failure": len(policy_ids),
            "environment_failure": len(environment_ids),
            "success_controls": len(control_ids),
        },
        "expansions": [],
    }
    _write_json(plan_path, plan)
    _rebuild_queue(output_dir, plan)
    return plan


def _load_bound_plan(
    output_dir: Path,
    *,
    verify_external: bool = True,
) -> dict[str, Any]:
    plan_path = output_dir / "review_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("protocol_version") != SEMANTIC_AUDIT_PROTOCOL_VERSION:
        raise RuntimeError("semantic-audit plan protocol mismatch")
    if plan.get("awm_source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("semantic-audit plan is not bound to the pinned AWM sources")
    if not verify_external:
        return plan
    bound_paths = {
        "data_sha256": Path(plan["data_path"]),
        "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "integrity_manifest_sha256": Path(plan["integrity_manifest_path"]),
        "screening_manifest_sha256": Path(plan["screening_dir"]) / "screening_manifest.json",
        "screening_trials_sha256": Path(plan["screening_dir"]) / "trials.jsonl",
    }
    for field, path in bound_paths.items():
        if sha256_file(path) != plan[field]:
            raise RuntimeError(f"semantic-audit source hash drift: {field}")
    for filename, expected in plan["awm_source_sha256"].items():
        if sha256_file(Path(plan["awm_data_dir"]) / filename) != expected:
            raise RuntimeError(f"semantic-audit AWM source hash drift: {filename}")
    return plan


async def capture_evidence(args) -> dict[str, Any]:
    output_dir = args.output_dir
    plan = _load_bound_plan(output_dir)
    if fetch_server_protocol(args.awm_base_url) != plan["awm_logical_time"]:
        raise RuntimeError("semantic evidence server logical-time protocol drift")
    rows, _, _ = load_candidate_rows(
        Path(plan["data_path"]),
        Path(plan["candidate_manifest_path"]),
        Path(plan["integrity_manifest_path"]),
    )
    rows_by_id = {str(row["task_id"]): row for row in rows}
    trials_path = Path(plan["screening_dir"]) / "trials.jsonl"
    trial_offsets = _jsonl_offsets(trials_path)
    pending = []
    for target in plan["targets"]:
        task_id = str(target["task_id"])
        try:
            _load_evidence_packet(output_dir, plan, task_id)
        except Exception:
            pending.append(task_id)
    if args.max_tasks is not None:
        pending = pending[: args.max_tasks]
    catalog = SourceCatalog(Path(plan["awm_data_dir"]))
    errors = {str(item["task_id"]): item for item in load_jsonl(output_dir / "capture_errors.jsonl")} if (output_dir / "capture_errors.jsonl").is_file() else {}
    captured = 0
    with trials_path.open("rb") as trials_handle:
        for task_id in pending:
            try:
                packet = await build_evidence_packet(
                    row=rows_by_id[task_id],
                    trial=_read_jsonl_at(trials_handle, trial_offsets[task_id]),
                    catalog=catalog,
                    output_dir=output_dir,
                    awm_base_url=args.awm_base_url,
                    screening_manifest_sha256=plan["screening_manifest_sha256"],
                )
                validate_evidence_packet(
                    packet,
                    output_dir=output_dir,
                    task_id=task_id,
                    screening_manifest_sha256=plan["screening_manifest_sha256"],
                )
                path = output_dir / "evidence" / f"{task_id.replace(':', '__')}.json"
                _write_json(path, packet)
                errors.pop(task_id, None)
                captured += 1
                print(
                    f"semantic_evidence_captured {captured}/{len(pending)} task_id={task_id}",
                    flush=True,
                )
            except Exception as exc:
                errors[task_id] = {
                    "task_id": task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "classification": ("replay_drift_pending" if isinstance(exc, ReplayDriftError) else "replay_or_infrastructure_pending"),
                }
                print(
                    f"semantic_evidence_pending task_id={task_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
    _write_jsonl(
        output_dir / "capture_errors.jsonl",
        [errors[key] for key in sorted(errors)],
    )
    queue = _rebuild_queue(output_dir, plan)
    return {
        "requested": len(pending),
        "captured": captured,
        "capture_errors": len(errors),
        "review_ready": sum(item["status"] == "ready" for item in queue),
    }


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"evidence ref is not a JSON pointer: {pointer!r}")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, Mapping):
            value = value[token]
        else:
            raise KeyError(pointer)
    return value


def record_judgment(args) -> dict[str, Any]:
    plan = _load_bound_plan(args.output_dir, verify_external=False)
    raw = json.loads(args.judgment.read_text(encoding="utf-8"))
    task_id = str(raw.get("task_id") or "")
    slot = str(raw.get("review_slot") or "")
    targets = {str(item["task_id"]) for item in plan["targets"]}
    if task_id not in targets:
        raise ValueError("judgment task is not in the review plan")
    evidence_path, packet = _load_evidence_packet(args.output_dir, plan, task_id)
    value = validate_judgment(
        raw,
        task_id=task_id,
        slot=slot,
        evidence_sha256=sha256_file(evidence_path),
        allowed_cohort_keys=list(packet["cohort_keys"]),
    )
    for pointer in value["evidence_refs"]:
        _resolve_json_pointer(packet, pointer)
    output = args.output_dir / "judgments" / slot / f"{task_id.replace(':', '__')}.json"
    if output.exists() and not args.replace:
        raise FileExistsError(f"judgment already exists: {output}")
    _write_json(output, value)
    return value


def next_review(args) -> str:
    plan = _load_bound_plan(args.output_dir, verify_external=False)
    queue_path = args.output_dir / "review_queue.jsonl"
    if not queue_path.is_file():
        raise FileNotFoundError(f"missing semantic review queue: {queue_path}")
    queue_stat = queue_path.stat()
    queue_identity = {
        "size": queue_stat.st_size,
        "mtime_ns": queue_stat.st_mtime_ns,
    }
    cursor_path = args.output_dir / f"review_cursor_{args.slot}.json"
    offset = 0
    if cursor_path.is_file():
        cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
        if cursor.get("queue_identity") == queue_identity:
            offset = int(cursor.get("offset") or 0)
            if not 0 <= offset <= queue_stat.st_size:
                offset = 0

    target_ids = {str(item["task_id"]) for item in plan["targets"]}
    with queue_path.open("rb") as handle:
        handle.seek(offset)
        while line := handle.readline():
            next_offset = handle.tell()
            if not line.strip():
                offset = next_offset
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid semantic review queue at byte {offset}") from exc
            task_id = str(item.get("task_id") or "")
            slot = str(item.get("review_slot") or "")
            if task_id not in target_ids or slot not in {"A", "B"}:
                raise RuntimeError(f"semantic review queue record is outside the bound plan: {item!r}")
            if slot != args.slot:
                offset = next_offset
                continue
            try:
                evidence_path, _ = _load_evidence_packet(args.output_dir, plan, task_id)
            except Exception:
                # Capture rebuilds the queue atomically, which invalidates this cursor.
                # Until then, skip unavailable evidence so one failed replay does not
                # block review of later successfully captured tasks.
                offset = next_offset
                continue
            judgment_path = args.output_dir / "judgments" / slot / f"{task_id.replace(':', '__')}.json"
            if judgment_path.is_file():
                try:
                    _validated_judgment_file(
                        output_dir=args.output_dir,
                        plan=plan,
                        task_id=task_id,
                        slot=slot,
                    )
                except Exception:
                    status = "invalid"
                else:
                    offset = next_offset
                    continue
            else:
                status = "ready"
            _write_json(
                cursor_path,
                {"queue_identity": queue_identity, "offset": offset},
            )
            live_item = {
                **item,
                "evidence_path": str(evidence_path.resolve()),
                "evidence_sha256": sha256_file(evidence_path),
                "judgment_path": str(judgment_path.resolve()),
                "status": status,
            }
            return render_review_prompt(live_item, evidence_path)
    _write_json(
        cursor_path,
        {"queue_identity": queue_identity, "offset": offset},
    )
    return ""


def _validated_task_judgments(
    output_dir: Path,
    task_id: str,
    *,
    plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = dict(plan) if plan is not None else _load_bound_plan(output_dir, verify_external=False)
    try:
        _load_evidence_packet(output_dir, plan, task_id)
    except Exception:
        return []
    values = []
    for slot in ("A", "B"):
        path = output_dir / "judgments" / slot / f"{task_id.replace(':', '__')}.json"
        if not path.is_file():
            continue
        try:
            _, value = _validated_judgment_file(
                output_dir=output_dir,
                plan=plan,
                task_id=task_id,
                slot=slot,
            )
            values.append(value)
        except Exception:
            continue
    return values


def _apply_control_expansion(output_dir: Path, plan: dict[str, Any]) -> list[str]:
    targets_by_id = {str(item["task_id"]): item for item in plan["targets"]}
    additions = set()
    triggers = []
    for task_id, target in targets_by_id.items():
        if not target.get("success_control"):
            continue
        consensus = reviewer_consensus(_validated_task_judgments(output_dir, task_id, plan=plan))
        if consensus.membership != "excluded":
            continue
        cohort_additions = set()
        for key in consensus.cohort_keys:
            cohort_additions.update(plan["cohort_task_ids"].get(key) or [])
        cohort_additions -= set(targets_by_id)
        additions.update(cohort_additions)
        triggers.append(
            {
                "control_task_id": task_id,
                "verdict": consensus.verdict,
                "cohort_keys": list(consensus.cohort_keys),
                "added_task_ids": sorted(cohort_additions),
            }
        )
    if not additions:
        return []
    candidate_order = {task_id: index for index, task_id in enumerate(plan["candidate_task_ids"])}
    for task_id in sorted(additions, key=candidate_order.__getitem__):
        plan["targets"].append(
            {
                "task_id": task_id,
                "basis": "success_control_cohort_expansion",
                "success_control": False,
            }
        )
    plan["targets"].sort(key=lambda item: candidate_order[str(item["task_id"])])
    plan["expansions"].append({"triggers": triggers, "added_task_ids": sorted(additions)})
    _write_json(output_dir / "review_plan.json", plan)
    _rebuild_queue(output_dir, plan)
    _write_json(output_dir / "cohort_expansion.json", plan["expansions"][-1])
    return sorted(additions)


def _derive_consensus_record(
    output_dir: Path,
    plan: Mapping[str, Any],
    task_id: str,
) -> tuple[dict[str, Any], Any]:
    judgments = _validated_task_judgments(
        output_dir,
        task_id,
        plan=plan,
    )
    consensus = reviewer_consensus(judgments)
    try:
        evidence_path, _ = _load_evidence_packet(output_dir, plan, task_id)
        evidence_sha256 = sha256_file(evidence_path)
    except Exception:
        evidence_sha256 = None
    record = {
        "task_id": task_id,
        "evidence_sha256": evidence_sha256,
        "judgments": [
            {
                "review_slot": judgment["review_slot"],
                "sha256": sha256_file(output_dir / "judgments" / judgment["review_slot"] / f"{task_id.replace(':', '__')}.json"),
            }
            for judgment in judgments
        ],
        "membership": consensus.membership,
        "reason": consensus.reason,
        "verdict": consensus.verdict,
        "path_relevance": consensus.path_relevance,
        "cohort_keys": list(consensus.cohort_keys),
    }
    return record, consensus


def verify_consensus_ledger(
    output_dir: Path,
    *,
    plan: Mapping[str, Any] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Re-derive every review consensus from bound evidence and A/B files."""
    plan = dict(plan) if plan is not None else _load_bound_plan(output_dir)
    if records is None:
        records = load_jsonl(output_dir / "review_consensus.jsonl")
    actual = [dict(record) for record in records]
    expected_ids = [str(target["task_id"]) for target in plan["targets"]]
    if [str(record.get("task_id")) for record in actual] != expected_ids:
        raise RuntimeError("semantic review consensus task order mismatch")
    for record, task_id in zip(actual, expected_ids, strict=True):
        expected, _ = _derive_consensus_record(output_dir, plan, task_id)
        if record != expected:
            raise RuntimeError(f"semantic review consensus provenance mismatch: {task_id}")
    return actual


def finalize_pool(args) -> dict[str, Any]:
    output_dir = args.output_dir
    plan = _load_bound_plan(output_dir)
    additions = _apply_control_expansion(output_dir, plan)
    if additions:
        return {
            "status": "cohort_expanded",
            "added_tasks": len(additions),
            "next": "capture evidence and complete A/B review for the expanded targets, then finalize again",
        }
    rows, candidate_manifest, integrity_manifest = load_candidate_rows(
        Path(plan["data_path"]),
        Path(plan["candidate_manifest_path"]),
        Path(plan["integrity_manifest_path"]),
    )
    targets = {str(item["task_id"]): item for item in plan["targets"]}
    membership = {}
    reasons = {}
    consensus_records = []
    for row in rows:
        task_id = str(row["task_id"])
        status = str(plan["screening_task_status"][task_id])
        if status == "infrastructure_pending":
            membership[task_id] = "pending"
            reasons[task_id] = "expert_screening_infrastructure_pending"
        elif task_id in targets:
            record, consensus = _derive_consensus_record(output_dir, plan, task_id)
            membership[task_id] = consensus.membership
            reasons[task_id] = consensus.reason
            consensus_records.append(record)
        else:
            membership[task_id] = "included"
            reasons[task_id] = "unreviewed_expert_success_outside_stratified_control"

    base_manifest = validate_base_manifest(args.all_manifest)
    if sha256_file(args.all_manifest) != candidate_manifest.get("base_manifest_sha256"):
        raise RuntimeError("semantic finalization base manifest differs from context selection")
    if sha256_file(args.all_data) != candidate_manifest.get("base_data_sha256"):
        raise RuntimeError("semantic finalization all-data parquet differs from context selection")
    expected_all_ids = [str(task_id) for task_id in base_manifest["split_task_ids"]["all"]]
    all_frame = pd.read_parquet(args.all_data)
    all_ids = [str(extra["task_id"]) for extra in all_frame["extra_info"].tolist()]
    if all_ids != expected_all_ids or len(set(all_ids)) != 10000:
        raise RuntimeError("semantic finalization all-data IDs do not match the pinned base manifest")
    selected_ids = [str(value) for value in candidate_manifest["task_ids"]]
    selected_set = set(selected_ids)
    candidate_set = set(plan["candidate_task_ids"])
    all_set = set(all_ids)
    if not candidate_set <= selected_set <= all_set:
        raise RuntimeError("semantic finalization source partitions are not nested")
    deterministic_excluded = selected_set - candidate_set
    out_of_context = all_set - selected_set
    for task_id in deterministic_excluded:
        reasons[task_id] = "deterministic_integrity_exclusion"
    for task_id in out_of_context:
        reasons[task_id] = "fixed_context_budget_exclusion"
    partitions = {
        "included": [task_id for task_id in all_ids if membership.get(task_id) == "included"],
        "excluded": [task_id for task_id in all_ids if task_id in deterministic_excluded or membership.get(task_id) == "excluded"],
        "pending": [task_id for task_id in all_ids if membership.get(task_id) == "pending"],
        "out_of_context": [task_id for task_id in all_ids if task_id in out_of_context],
    }
    flattened = [task_id for name in PARTITIONS for task_id in partitions[name]]
    if len(flattened) != 10000 or set(flattened) != all_set or len(flattened) != len(set(flattened)):
        raise RuntimeError("semantic final partitions do not form an exact 10,000-task partition")
    rows_by_id = {str(row["task_id"]): row for row in rows}
    pool_path = output_dir / POOL_FILENAME
    pd.DataFrame([rows_by_id[task_id]["training_row"] for task_id in partitions["included"]]).to_parquet(pool_path, index=False)
    partition_hashes = {}
    for name in PARTITIONS:
        path = output_dir / f"{name}_task_ids.json"
        _write_json(path, partitions[name])
        partition_hashes[f"{name}_task_ids_sha256"] = sha256_file(path)
    consensus_path = output_dir / "review_consensus.jsonl"
    _write_jsonl(consensus_path, consensus_records)
    verify_consensus_ledger(output_dir, plan=plan, records=consensus_records)
    copied_base_manifest = output_dir / "source_all_manifest.json"
    shutil.copyfile(args.all_manifest, copied_base_manifest)
    _rebuild_queue(output_dir, plan, write_prompts=False)
    manifest = {
        "protocol_version": SEMANTIC_AUDIT_PROTOCOL_VERSION,
        "kind": "awm_verifier_reliable_task_pool",
        "review_plan_sha256": sha256_file(output_dir / "review_plan.json"),
        "review_queue_sha256": sha256_file(output_dir / "review_queue.jsonl"),
        "review_consensus_sha256": sha256_file(consensus_path),
        "source_all_data_sha256": sha256_file(args.all_data),
        "source_all_manifest_filename": copied_base_manifest.name,
        "source_all_manifest_sha256": sha256_file(copied_base_manifest),
        "source_all_task_ids": all_ids,
        "candidate_manifest_sha256": plan["candidate_manifest_sha256"],
        "integrity_manifest_sha256": plan["integrity_manifest_sha256"],
        "screening_manifest_sha256": plan["screening_manifest_sha256"],
        "screening_trials_sha256": plan["screening_trials_sha256"],
        "membership_policy": "one verifier-reliable pool shared by semantic and outcome methods",
        "counts": {name: len(partitions[name]) for name in PARTITIONS},
        "task_reasons": reasons,
        "training_pool_filename": POOL_FILENAME,
        "training_pool_task_ids": partitions["included"],
        "training_pool_data_sha256": sha256_file(pool_path),
        **partition_hashes,
    }
    _write_json(output_dir / POOL_MANIFEST_FILENAME, manifest)
    return {"status": "finalized", "counts": manifest["counts"], "pool": str(pool_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--data", type=Path, required=True)
    plan.add_argument("--candidate-manifest", type=Path, required=True)
    plan.add_argument("--integrity-manifest", type=Path, required=True)
    plan.add_argument("--screening-dir", type=Path, required=True)
    plan.add_argument("--awm-data-dir", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    capture.add_argument("--max-tasks", type=int)

    next_parser = subparsers.add_parser("next")
    next_parser.add_argument("--output-dir", type=Path, required=True)
    next_parser.add_argument("--slot", choices=("A", "B"), required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--output-dir", type=Path, required=True)
    record.add_argument("--judgment", type=Path, required=True)
    record.add_argument("--replace", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--all-data", type=Path, required=True)
    finalize.add_argument("--all-manifest", type=Path, required=True)
    args = parser.parse_args()

    if getattr(args, "max_tasks", None) is not None and args.max_tasks <= 0:
        parser.error("--max-tasks must be positive")
    if args.command == "plan":
        result = create_plan(args)
        result = {"targets": len(result["targets"]), "initial_target_counts": result["initial_target_counts"]}
    elif args.command == "capture":
        result = asyncio.run(capture_evidence(args))
    elif args.command == "next":
        print(next_review(args), end="")
        return
    elif args.command == "record":
        result = record_judgment(args)
    else:
        result = finalize_pool(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
