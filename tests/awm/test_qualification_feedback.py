import json
from pathlib import Path

import pandas as pd

from agent_system.environments.env_package.awm.integrity import (
    INTEGRITY_PROTOCOL_VERSION,
    verify_integrity,
)
from agent_system.environments.env_package.awm.native_rollout import sha256_file
from agent_system.environments.env_package.awm.qualification import (
    QUALIFICATION_PROTOCOL_VERSION,
)
from agent_system.environments.env_package.awm.qualification_feedback import (
    apply_qualification_feedback,
    deterministic_environment_error,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _deterministic_trial(task_id: str = "scenario:0", *, confidence: int = 90) -> dict:
    return {
        "task_id": task_id,
        "trial_index": 0,
        "seed": 300,
        "status": "infrastructure_exhausted",
        "infrastructure_attempts": 3,
        "errors": [
            "verifier infrastructure error: 'server_error'",
            "verifier infrastructure error: 'server_error'",
            "TimeoutError: ",
        ],
        "last_result": {
            "reward_type": "server_error",
            "trajectory": [
                {
                    "decision": 2,
                    "parsed_action": '{"kind":"tool","name":"search"}',
                    "tool_response_is_error": True,
                    "tool_response": "Error calling search. Status code: 422. Response: bad route",
                }
            ],
            "verify_result": {
                "llm_judge": {
                    "classification": "server_error",
                    "confidence_score": [0, 5, confidence, 5],
                    "reasoning": "A documented tool call repeatedly reached a broken route.",
                    "evidence": {"error_signals": ["HTTP 422"]},
                }
            },
        },
    }


def test_deterministic_feedback_requires_repeated_judge_confirmed_http_error():
    evidence = deterministic_environment_error(_deterministic_trial())
    assert evidence is not None
    assert evidence["server_error_attempts"] == 2
    assert evidence["http_status_codes"] == [422]

    timeout = _deterministic_trial()
    timeout["errors"] = ["TimeoutError: "] * 3
    timeout["last_result"] = None
    assert deterministic_environment_error(timeout) is None

    single_server_error = _deterministic_trial()
    single_server_error["errors"] = [
        "verifier infrastructure error: 'server_error'",
        "TimeoutError: ",
        "TimeoutError: ",
    ]
    assert deterministic_environment_error(single_server_error) is None
    assert deterministic_environment_error(_deterministic_trial(confidence=79)) is None


def _integrity_fixture(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir()
    rows = [
        {
            "extra_info": {"task_id": f"scenario:{index}"},
            "env_kwargs": {"scenario": "scenario", "task_idx": index},
        }
        for index in range(2)
    ]
    filtered_path = output_dir / "awm_integrity_filtered.parquet"
    pd.DataFrame(rows).to_parquet(filtered_path, index=False)
    static_path = output_dir / "static_audit.jsonl"
    judge_path = output_dir / "judge_audit.jsonl"
    audit_path = output_dir / "integrity_audit.jsonl"
    static_path.write_text("", encoding="utf-8")
    judge_path.write_text("", encoding="utf-8")
    audit_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"scenario:{index}",
                    "status": "pass",
                    "status_reasons": [],
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    for name in (
        "quarantine_task_ids.json",
        "needs_review_task_ids.json",
        "infrastructure_pending_task_ids.json",
    ):
        _write_json(output_dir / name, [])
    manifest = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "candidate_task_ids": ["scenario:0", "scenario:1"],
        "filtered_task_ids": ["scenario:0", "scenario:1"],
        "counts": {"pass": 2},
        "static_audit_sha256": sha256_file(static_path),
        "judge_audit_sha256": sha256_file(judge_path),
        "integrity_audit_sha256": sha256_file(audit_path),
        "filtered_data_sha256": sha256_file(filtered_path),
        "quarantine_task_ids_sha256": sha256_file(output_dir / "quarantine_task_ids.json"),
        "needs_review_task_ids_sha256": sha256_file(output_dir / "needs_review_task_ids.json"),
        "infrastructure_pending_task_ids_sha256": sha256_file(output_dir / "infrastructure_pending_task_ids.json"),
    }
    manifest_path = output_dir / "integrity_manifest.json"
    _write_json(manifest_path, manifest)
    verify_integrity(output_dir)
    return manifest_path, filtered_path


def _qualification_fixture(
    output_dir: Path,
    integrity_manifest_path: Path,
    filtered_path: Path,
) -> None:
    output_dir.mkdir()
    config = {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "candidate_manifest_sha256": "selection-sha",
        "integrity_manifest_sha256": sha256_file(integrity_manifest_path),
        "candidate_data_sha256": sha256_file(filtered_path),
        "candidate_task_ids": ["scenario:0", "scenario:1"],
    }
    _write_json(output_dir / "config.json", config)
    trials = [
        _deterministic_trial(),
        {
            "task_id": "scenario:1",
            "trial_index": 0,
            "seed": 300,
            "status": "infrastructure_exhausted",
            "infrastructure_attempts": 3,
            "errors": ["TimeoutError: "] * 3,
            "last_result": None,
        },
    ]
    trials_path = output_dir / "trials.jsonl"
    trials_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in trials),
        encoding="utf-8",
    )
    manifest = {
        **config,
        "trials_sha256": sha256_file(trials_path),
    }
    _write_json(output_dir / "qualification_manifest.json", manifest)


def test_feedback_updates_quarantine_and_preserves_resume_snapshot(tmp_path):
    integrity_dir = tmp_path / "integrity"
    qualification_dir = tmp_path / "qualification"
    manifest_path, filtered_path = _integrity_fixture(integrity_dir)
    source_manifest_sha256 = sha256_file(manifest_path)
    source_data_sha256 = sha256_file(filtered_path)
    _qualification_fixture(qualification_dir, manifest_path, filtered_path)

    result = apply_qualification_feedback(integrity_dir, qualification_dir)

    assert result["applied_task_ids"] == ["scenario:0"]
    assert result["infrastructure_pending"] == 1
    assert json.loads((integrity_dir / "quarantine_task_ids.json").read_text()) == ["scenario:0"]
    frame = pd.read_parquet(integrity_dir / "awm_integrity_filtered.parquet")
    assert [extra["task_id"] for extra in frame["extra_info"].tolist()] == ["scenario:1"]
    updated_manifest = json.loads((integrity_dir / "integrity_manifest.json").read_text())
    assert updated_manifest["counts"] == {"pass": 1, "quarantine": 1}
    assert updated_manifest["qualification_feedback_quarantine_task_ids"] == ["scenario:0"]
    snapshot = qualification_dir / "source_integrity_snapshot"
    assert sha256_file(snapshot / "integrity_manifest.json") == source_manifest_sha256
    assert sha256_file(snapshot / "awm_integrity_filtered.parquet") == source_data_sha256
    verify_integrity(integrity_dir)

    repeated = apply_qualification_feedback(integrity_dir, qualification_dir)
    assert repeated["applied"] == 0
    repeated_manifest = json.loads((integrity_dir / "integrity_manifest.json").read_text())
    assert len(repeated_manifest["qualification_feedback_provenance"]) == 1

    trials_path = qualification_dir / "trials.jsonl"
    extended_trials = [_deterministic_trial(), _deterministic_trial("scenario:1")]
    trials_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in extended_trials),
        encoding="utf-8",
    )
    qualification_manifest_path = qualification_dir / "qualification_manifest.json"
    qualification_manifest = json.loads(qualification_manifest_path.read_text())
    qualification_manifest["trials_sha256"] = sha256_file(trials_path)
    _write_json(qualification_manifest_path, qualification_manifest)

    extended = apply_qualification_feedback(integrity_dir, qualification_dir)

    assert extended["applied_task_ids"] == ["scenario:1"]
    assert json.loads((integrity_dir / "quarantine_task_ids.json").read_text()) == [
        "scenario:0",
        "scenario:1",
    ]
    final_manifest = json.loads((integrity_dir / "integrity_manifest.json").read_text())
    assert final_manifest["counts"] == {"quarantine": 2}
    assert len(final_manifest["qualification_feedback_provenance"]) == 2
