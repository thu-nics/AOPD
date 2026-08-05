import json
from pathlib import Path

import pandas as pd
import pytest

from agent_system.environments.env_package.awm.expert_screening import (
    EXPERT_SCREENING_PROTOCOL_VERSION,
)
from agent_system.environments.env_package.awm.integrity import (
    INTEGRITY_PROTOCOL_VERSION,
    TRAINING_POOL_FILENAME,
    _load_candidate_rows,
    _write_prefilter_artifacts,
    rebase_integrity,
)
from agent_system.environments.env_package.awm.native_rollout import sha256_file
from agent_system.environments.env_package.awm.qualification import (
    QUALIFICATION_PROTOCOL_VERSION,
    qualification_rollout_protocol,
)
from agent_system.environments.env_package.awm.qualification_migration import (
    migrate_qualification,
)
from agent_system.environments.env_package.awm.selection import SELECTION_PROTOCOL_VERSION
from agent_system.environments.env_package.awm.verification import (
    verify,
    verify_training_pool,
)


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
            "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
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


def _build_integrity(
    tmp_path: Path,
    specs: list[tuple[str, str]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    selection_dir = tmp_path / "selection"
    integrity_dir = tmp_path / "integrity"
    selection_dir.mkdir()
    integrity_dir.mkdir()
    specs = specs or [
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
    passing_rows = [row for row, (_, status) in zip(training_rows, specs, strict=True) if status == "pass"]
    pd.DataFrame(passing_rows).to_parquet(filtered_path, index=False)
    lists = {
        "quarantine_task_ids.json": [task_id for task_id, status in specs if status == "quarantine"],
        "needs_review_task_ids.json": [task_id for task_id, status in specs if status == "needs_review"],
        "infrastructure_pending_task_ids.json": [],
    }
    for name, values in lists.items():
        _write_json(integrity_dir / name, values)
    config = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_task_ids": [spec[0] for spec in specs],
    }
    _write_json(integrity_dir / "config.json", config)
    manifest = {
        **config,
        "kind": "awm_task_integrity_filter",
        "filtered_task_ids": [task_id for task_id, status in specs if status == "pass"],
        "counts": {status: sum(item_status == status for _, item_status in specs) for status in sorted({status for _, status in specs})},
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
        integrity_dir / TRAINING_POOL_FILENAME,
        filtered_path,
    )


def test_deterministic_training_pool_verifier_accepts_hash_bound_pool(tmp_path):
    _, integrity_manifest_path, training_pool, _ = _build_integrity(tmp_path)

    result = verify_training_pool(training_pool, integrity_manifest_path)

    assert result == {
        "tasks": 2,
        "data": str(training_pool),
        "kind": "deterministic_training_pool",
    }


def test_training_pool_verifier_accepts_hash_bound_expert_screened_pool(tmp_path):
    _, _, deterministic_pool, _ = _build_integrity(tmp_path)
    screening_dir = tmp_path / "screening"
    screening_dir.mkdir()
    screened_pool = screening_dir / "awm_expert_screened_pool.parquet"
    frame = pd.read_parquet(deterministic_pool).iloc[:1].copy()
    frame.to_parquet(screened_pool, index=False)
    task_ids = [str(item["task_id"]) for item in frame["extra_info"]]
    trials_path = screening_dir / "trials.jsonl"
    trials_path.write_text(
        json.dumps(
            {
                "task_id": task_ids[0],
                "seed": 300,
                "status": "policy_failure",
                "result": {"success": False},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = screening_dir / "config.json"
    config = {
        "protocol_version": EXPERT_SCREENING_PROTOCOL_VERSION,
        "candidate_task_ids": task_ids,
    }
    _write_json(config_path, config)
    manifest_path = screening_dir / "screening_manifest.json"
    _write_json(
        manifest_path,
        {
            **config,
            "kind": "awm_one_pass_expert_screening",
            "config_sha256": sha256_file(config_path),
            "training_pool_filename": screened_pool.name,
            "training_pool_data_sha256": sha256_file(screened_pool),
            "training_pool_task_ids": task_ids,
            "accepted_task_ids": task_ids,
            "task_status": {task_id: "accepted_policy_failure" for task_id in task_ids},
            "counts": {
                "accepted_success": 0,
                "accepted_policy_failure": 1,
                "rejected_environment": 0,
                "infrastructure_pending": 0,
                "pending": 0,
            },
            "trials_sha256": sha256_file(trials_path),
        },
    )

    assert verify_training_pool(screened_pool, manifest_path) == {
        "tasks": 1,
        "data": str(screened_pool),
        "kind": "expert_screened_training_pool",
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_status"][task_ids[0]] = "pending"
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="task-status derivation mismatch"):
        verify_training_pool(screened_pool, manifest_path)


def test_training_pool_verifier_rejects_legacy_qualification_manifest(tmp_path):
    data = tmp_path / "awm_expert_qualified_all.parquet"
    pd.DataFrame([{"extra_info": {"task_id": "scenario:0"}}]).to_parquet(data, index=False)
    manifest = tmp_path / "qualification_manifest.json"
    _write_json(manifest, {"protocol_version": QUALIFICATION_PROTOCOL_VERSION})

    with pytest.raises(RuntimeError, match="legacy qualification manifests are unsupported"):
        verify_training_pool(data, manifest)


def test_v6_to_current_migration_rejects_legacy_pass_only_pool(tmp_path):
    candidate_manifest_path, integrity_manifest_path, _, filtered_data = _build_integrity(tmp_path)

    with pytest.raises(RuntimeError, match="hash-bound prefilter pool"):
        migrate_qualification(
            qualification_dir=tmp_path / "qualification",
            data_path=filtered_data,
            candidate_manifest_path=candidate_manifest_path,
            integrity_manifest_path=integrity_manifest_path,
        )


def test_integrity_rejects_candidate_row_from_an_old_selection_protocol(tmp_path):
    candidate_manifest_path, _, _, _ = _build_integrity(tmp_path)
    candidate_data = candidate_manifest_path.parent / "candidates.parquet"
    frame = pd.read_parquet(candidate_data)
    extras = []
    for value in frame["extra_info"]:
        extra = dict(value)
        extra["selection_protocol_version"] = SELECTION_PROTOCOL_VERSION - 1
        extras.append(extra)
    frame["extra_info"] = extras
    frame.to_parquet(candidate_data, index=False)
    manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_data_sha256"] = sha256_file(candidate_data)
    _write_json(candidate_manifest_path, manifest)

    with pytest.raises(RuntimeError, match="candidate row selection protocol mismatch"):
        _load_candidate_rows(candidate_data, candidate_manifest_path)


def test_integrity_rebase_reuses_ordered_subset_without_api_calls(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    _build_integrity(source_root)
    target_root = tmp_path / "target"
    target_root.mkdir()
    target_candidate_manifest, _, _, _ = _build_integrity(target_root, specs=[("keep:0", "pass")])
    target_data = target_root / "selection" / "candidates.parquet"
    output_dir = tmp_path / "rebased_integrity"

    result = rebase_integrity(
        source_dir=source_root / "integrity",
        data_path=target_data,
        candidate_manifest_path=target_candidate_manifest,
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "integrity_manifest.json").read_text(encoding="utf-8"))
    assert result["api_calls"] == 0
    assert result["source_tasks"] == 3
    assert result["target_tasks"] == 1
    assert manifest["counts"] == {"pass": 1}
    assert manifest["prefilter_candidate_task_ids"] == ["keep:0"]


def test_v6_to_current_migration_reuses_compatible_trials_without_api_calls(tmp_path):
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

    result = migrate_qualification(
        qualification_dir=qualification_dir,
        data_path=prefilter_data,
        candidate_manifest_path=candidate_manifest_path,
        integrity_manifest_path=integrity_manifest_path,
        confirm_legacy_context=True,
    )

    manifest = json.loads((qualification_dir / "qualification_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == QUALIFICATION_PROTOCOL_VERSION
    assert manifest["counts"]["qualified"] == 1
    assert manifest["counts"]["pending"] == 1
    assert manifest["counts"]["rejected_policy"] == 0
    assert manifest["counts"]["rejected_infrastructure"] == 0
    assert manifest["task_status"] == {"keep:0": "qualified", "new:0": "pending"}
    for key, value in qualification_rollout_protocol().items():
        assert manifest[key] == value
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


def _write_v7_cache(
    qualification_dir: Path,
    prefilter_data: Path,
    candidate_manifest_path: Path,
    integrity_manifest_path: Path,
) -> str:
    qualification_dir.mkdir()
    frame = pd.read_parquet(prefilter_data)
    task_ids = [str(extra["task_id"]) for extra in frame["extra_info"].tolist()]
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
    trials_path = qualification_dir / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in trials),
        encoding="utf-8",
    )
    integrity_manifest = json.loads(integrity_manifest_path.read_text(encoding="utf-8"))
    config = {
        "protocol_version": 7,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "integrity_manifest_sha256": sha256_file(integrity_manifest_path),
        "candidate_data_sha256": sha256_file(prefilter_data),
        "candidate_task_ids": task_ids,
        "candidate_scope": "cheap_deterministic_prefilter_non_quarantine",
        "prefilter_protocol_version": integrity_manifest["prefilter_protocol_version"],
        "final_task_statuses": [
            "qualified",
            "rejected_policy",
            "rejected_infrastructure",
            "pending",
        ],
        "max_decisions": 20,
        "model": "deepseek-v4-flash",
    }
    _write_json(qualification_dir / "config.json", config)
    qualified = frame.loc[[str(extra["task_id"]) == "keep:0" for extra in frame["extra_info"].tolist()]].reset_index(drop=True)
    qualified_path = qualification_dir / "awm_expert_qualified_all.parquet"
    qualified.to_parquet(qualified_path, index=False)
    manifest = {
        **config,
        "provider_identity": {
            "model": "deepseek-v4-flash",
            "system_fingerprint": "revision",
        },
        "trials_sha256": sha256_file(trials_path),
        "qualified_all_sha256": sha256_file(qualified_path),
        "qualified_task_ids": ["keep:0"],
        "qualified_train_b8_sha256": None,
        "qualified_train_b8_task_ids": [],
    }
    manifest_path = qualification_dir / "qualification_manifest.json"
    _write_json(manifest_path, manifest)
    return sha256_file(manifest_path)


def test_v7_to_v8_migration_requires_confirmation_and_binds_context(tmp_path):
    candidate_manifest_path, integrity_manifest_path, prefilter_data, _ = _build_integrity(tmp_path)
    qualification_dir = tmp_path / "qualification"
    source_manifest_sha256 = _write_v7_cache(
        qualification_dir,
        prefilter_data,
        candidate_manifest_path,
        integrity_manifest_path,
    )

    with pytest.raises(RuntimeError, match="explicit confirmation"):
        migrate_qualification(
            qualification_dir=qualification_dir,
            data_path=prefilter_data,
            candidate_manifest_path=candidate_manifest_path,
            integrity_manifest_path=integrity_manifest_path,
        )

    result = migrate_qualification(
        qualification_dir=qualification_dir,
        data_path=prefilter_data,
        candidate_manifest_path=candidate_manifest_path,
        integrity_manifest_path=integrity_manifest_path,
        confirm_legacy_context=True,
    )

    config = json.loads((qualification_dir / "config.json").read_text(encoding="utf-8"))
    manifest_path = qualification_dir / "qualification_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert config["protocol_version"] == QUALIFICATION_PROTOCOL_VERSION
    assert manifest["protocol_version"] == QUALIFICATION_PROTOCOL_VERSION
    for key, value in qualification_rollout_protocol().items():
        assert config[key] == value
        assert manifest[key] == value
    migration = result["migration"]
    assert migration["from_qualification_protocol"] == 7
    assert migration["to_qualification_protocol"] == QUALIFICATION_PROTOCOL_VERSION
    assert migration["api_calls"] == 0
    assert migration["retained_trial_records"] == 4
    assert migration["removed_trial_records"] == 0
    assert migration["source_context_binding"]["status"] == "operator_confirmed"
    assert migration["source_context_binding"]["artifact_evidence"] == {
        "max_observed_decisions": 1,
        "max_observed_prompt_tokens": 10,
        "trajectory_records": 4,
        "prompt_usage_records": 4,
    }
    archive = qualification_dir / migration["archive_subdir"]
    assert sha256_file(archive / "qualification_manifest.json") == source_manifest_sha256
    assert (
        verify(
            qualification_dir / "awm_expert_qualified_all.parquet",
            manifest_path,
        )["tasks"]
        == 1
    )

    manifest["history_window"] = 10
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="rollout protocol mismatch"):
        verify(qualification_dir / "awm_expert_qualified_all.parquet", manifest_path)


def test_v8_candidate_rebase_retains_compatible_trials_without_api_calls(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    candidate_manifest_path, integrity_manifest_path, prefilter_data, _ = _build_integrity(source_root)
    qualification_dir = tmp_path / "qualification"
    _write_v7_cache(
        qualification_dir,
        prefilter_data,
        candidate_manifest_path,
        integrity_manifest_path,
    )
    migrate_qualification(
        qualification_dir=qualification_dir,
        data_path=prefilter_data,
        candidate_manifest_path=candidate_manifest_path,
        integrity_manifest_path=integrity_manifest_path,
        confirm_legacy_context=True,
    )

    reordered_root = tmp_path / "reordered"
    reordered_root.mkdir()
    reordered_candidate_manifest, reordered_integrity_manifest, reordered_data, _ = _build_integrity(
        reordered_root,
        specs=[("new:0", "needs_review"), ("keep:0", "pass")],
    )
    with pytest.raises(RuntimeError, match="ordered candidate subset"):
        migrate_qualification(
            qualification_dir=qualification_dir,
            data_path=reordered_data,
            candidate_manifest_path=reordered_candidate_manifest,
            integrity_manifest_path=reordered_integrity_manifest,
        )

    target_root = tmp_path / "target"
    target_root.mkdir()
    target_candidate_manifest, target_integrity_manifest, target_data, _ = _build_integrity(target_root, specs=[("keep:0", "pass")])
    result = migrate_qualification(
        qualification_dir=qualification_dir,
        data_path=target_data,
        candidate_manifest_path=target_candidate_manifest,
        integrity_manifest_path=target_integrity_manifest,
    )

    migration = result["migration"]
    manifest = json.loads((qualification_dir / "qualification_manifest.json").read_text(encoding="utf-8"))
    assert migration["from_qualification_protocol"] == QUALIFICATION_PROTOCOL_VERSION
    assert migration["to_qualification_protocol"] == QUALIFICATION_PROTOCOL_VERSION
    assert migration["api_calls"] == 0
    assert migration["source_context_binding"]["status"] == "manifest_bound"
    assert migration["source_trial_records"] == 4
    assert migration["retained_trial_records"] == 4
    assert migration["removed_trial_records"] == 0
    assert migration["removed_candidate_task_ids"] == ["new:0"]
    assert manifest["counts"]["qualified"] == 1
    assert manifest["counts"]["pending"] == 0
    assert manifest["qualified_task_ids"] == ["keep:0"]
    assert (
        verify(
            qualification_dir / "awm_expert_qualified_all.parquet",
            qualification_dir / "qualification_manifest.json",
        )["tasks"]
        == 1
    )
