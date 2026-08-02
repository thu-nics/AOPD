import json
from pathlib import Path

import pandas as pd
import pytest

from agent_system.environments.env_package.awm.integrity import (
    INTEGRITY_PROTOCOL_VERSION,
    _write_prefilter_artifacts,
)
from agent_system.environments.env_package.awm.native_rollout import sha256_file
from agent_system.environments.env_package.awm.qualification_migration import migrate_v6_to_v7
from agent_system.environments.env_package.awm.selection import SELECTION_PROTOCOL_VERSION
from agent_system.environments.env_package.awm.verification import verify


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(task_id: str, status: str) -> tuple[dict, dict]:
    scenario, raw_index = task_id.split(":")
    training_row = {
        "extra_info": {
            "task_id": task_id,
            "task": f"Complete {task_id}",
            "native_prompt_tokens": 100,
            "tool_schema_hash": "canonical",
            "raw_tool_schema_hash": "raw",
            "tool_schema_repair_count": 0,
            "awm_integrity_status": status,
        },
        "env_kwargs": {"scenario": scenario, "task_idx": int(raw_index)},
    }
    return training_row, {
        "task_id": task_id,
        "scenario": scenario,
        "task_idx": int(raw_index),
        "task": training_row["extra_info"]["task"],
        "native_prompt_tokens": 100,
        "tool_schema_hash": "canonical",
        "raw_tool_schema_hash": "raw",
        "tool_schema_repair_count": 0,
        "training_row": training_row,
    }


def _build_integrity(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    selection_dir = tmp_path / "selection"
    integrity_dir = tmp_path / "integrity"
    selection_dir.mkdir()
    integrity_dir.mkdir()
    specs = [
        ("keep:0", "pass"),
        ("drop:0", "quarantine"),
        ("new:0", "needs_review"),
    ]
    pairs = [_row(*spec) for spec in specs]
    training_rows = [pair[0] for pair in pairs]
    candidate_rows = [pair[1] for pair in pairs]
    selection_data = selection_dir / "candidates.parquet"
    pd.DataFrame(training_rows).to_parquet(selection_data, index=False)
    candidate_manifest_path = selection_dir / "candidate_manifest.json"
    _write_json(
        candidate_manifest_path,
        {
            "protocol_version": SELECTION_PROTOCOL_VERSION,
            "candidate_data_sha256": sha256_file(selection_data),
            "task_ids": [spec[0] for spec in specs],
            "selected_counts": {"tasks": 3},
        },
    )

    static_path = integrity_dir / "static_audit.jsonl"
    judge_path = integrity_dir / "judge_audit.jsonl"
    audit_path = integrity_dir / "integrity_audit.jsonl"
    static_path.write_text("", encoding="utf-8")
    judge_path.write_text("", encoding="utf-8")
    records = [{"task_id": task_id, "status": status, "status_reasons": []} for task_id, status in specs]
    audit_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    filtered_path = integrity_dir / "awm_integrity_filtered.parquet"
    pd.DataFrame([training_rows[0]]).to_parquet(filtered_path, index=False)
    lists = {
        "quarantine_task_ids.json": ["drop:0"],
        "needs_review_task_ids.json": ["new:0"],
        "infrastructure_pending_task_ids.json": [],
    }
    for name, values in lists.items():
        _write_json(integrity_dir / name, values)
    manifest = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_task_ids": [spec[0] for spec in specs],
        "filtered_task_ids": ["keep:0"],
        "counts": {"needs_review": 1, "pass": 1, "quarantine": 1},
        "static_audit_sha256": sha256_file(static_path),
        "judge_audit_sha256": sha256_file(judge_path),
        "integrity_audit_sha256": sha256_file(audit_path),
        "filtered_data_sha256": sha256_file(filtered_path),
        "quarantine_task_ids_sha256": sha256_file(integrity_dir / "quarantine_task_ids.json"),
        "needs_review_task_ids_sha256": sha256_file(integrity_dir / "needs_review_task_ids.json"),
        "infrastructure_pending_task_ids_sha256": sha256_file(integrity_dir / "infrastructure_pending_task_ids.json"),
    }
    manifest.update(_write_prefilter_artifacts(candidate_rows, records, integrity_dir))
    integrity_manifest_path = integrity_dir / "integrity_manifest.json"
    _write_json(integrity_manifest_path, manifest)
    return (
        candidate_manifest_path,
        integrity_manifest_path,
        integrity_dir / "awm_prefilter_candidates.parquet",
        filtered_path,
    )


def test_v6_migration_rejects_legacy_pass_only_pool(tmp_path):
    candidate_manifest_path, integrity_manifest_path, _, filtered_data = _build_integrity(tmp_path)

    with pytest.raises(RuntimeError, match="hash-bound prefilter pool"):
        migrate_v6_to_v7(
            qualification_dir=tmp_path / "qualification",
            data_path=filtered_data,
            candidate_manifest_path=candidate_manifest_path,
            integrity_manifest_path=integrity_manifest_path,
        )


def test_v6_migration_reuses_compatible_trials_without_api_calls(tmp_path):
    candidate_manifest_path, integrity_manifest_path, prefilter_data, _ = _build_integrity(tmp_path)
    qualification_dir = tmp_path / "qualification"
    qualification_dir.mkdir()
    trials = [
        {
            "task_id": "keep:0",
            "trial_index": index,
            "seed": 300 + index,
            "status": "success",
            "result": {
                "success": True,
                "decisions": 1,
                "trajectory": [
                    {
                        "model": "deepseek-v4-flash",
                        "system_fingerprint": "revision",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                    }
                ],
            },
        }
        for index in range(4)
    ]
    trials.append(
        {
            "task_id": "drop:0",
            "trial_index": 0,
            "seed": 300,
            "status": "infrastructure_exhausted",
            "infrastructure_attempts": 3,
            "errors": ["server error"] * 3,
        }
    )
    trials_path = qualification_dir / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in trials),
        encoding="utf-8",
    )
    old_trials_sha256 = sha256_file(trials_path)
    old_config = {
        "protocol_version": 6,
        "candidate_task_ids": ["keep:0", "drop:0"],
        "model": "deepseek-v4-flash",
    }
    _write_json(qualification_dir / "config.json", old_config)
    _write_json(
        qualification_dir / "qualification_manifest.json",
        {
            **old_config,
            "trials_sha256": old_trials_sha256,
        },
    )

    result = migrate_v6_to_v7(
        qualification_dir=qualification_dir,
        data_path=prefilter_data,
        candidate_manifest_path=candidate_manifest_path,
        integrity_manifest_path=integrity_manifest_path,
    )

    manifest = json.loads((qualification_dir / "qualification_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == 7
    assert manifest["counts"]["qualified"] == 1
    assert manifest["counts"]["pending"] == 1
    assert manifest["counts"]["rejected_policy"] == 0
    assert manifest["counts"]["rejected_infrastructure"] == 0
    assert manifest["task_status"] == {"keep:0": "qualified", "new:0": "pending"}
    assert result["migration"]["api_calls"] == 0
    assert result["migration"]["retained_trial_records"] == 4
    assert result["migration"]["removed_trial_task_ids"] == ["drop:0"]
    archive = qualification_dir / result["migration"]["archive_subdir"]
    assert sha256_file(archive / "trials.jsonl") == old_trials_sha256
    assert (
        verify(
            qualification_dir / "awm_expert_qualified_all.parquet",
            qualification_dir / "qualification_manifest.json",
        )["tasks"]
        == 1
    )
