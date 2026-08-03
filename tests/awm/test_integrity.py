import copy

import pandas as pd

from agent_system.environments.env_package.awm.integrity import (
    _add_usage,
    _parse_judge_json,
    _write_prefilter_artifacts,
    classify_static_record,
    judge_consensus,
    preserve_qualification_feedback,
    refresh_cached_static_record,
    select_judge_task_ids,
    static_source_audit,
)


def _verifier(mode: str, task: str) -> dict:
    name = "verify_task_completion" if mode == "code" else "verify_task"
    return {
        "scenario": "sample",
        "task_idx": 0,
        "task": task,
        "verification": {"code": f"def {name}(initial_db_path, final_db_path):\n    return {{}}\n"},
    }


def _runtime(**updates):
    value = {
        "status": "ok",
        "runtime_task_matches": True,
        "raw_tool_schema_matches": True,
        "canonical_tool_schema_matches": True,
        "no_op_reward_type": "incomplete",
    }
    value.update(updates)
    return value


def test_static_source_audit_accepts_identical_duplicates_and_flags_semantic_overlap():
    task = "Create a document titled 'Existing Plan'."
    row = {"task": task, "task_idx": 0}
    task_record = {"scenario": "sample", "tasks": [task]}
    code = _verifier("code", task)
    code_duplicate = copy.deepcopy(code)
    code_duplicate["verification"]["raw_response"] = "irrelevant generation metadata"
    sql = _verifier("sql", task)

    result, sources = static_source_audit(
        row,
        task_records=[task_record],
        code_records=[code, code_duplicate],
        sql_records=[sql],
        sample_data={
            "tables": [
                {
                    "table_name": "document",
                    "insert_statements": ["INSERT INTO document VALUES ('Existing Plan')"],
                }
            ]
        },
    )

    assert sources["code"]["entrypoint"] == "verify_task_completion"
    assert "requested_literal_present_in_initial_target_table" in result["semantic_warning_codes"]
    assert not [item for item in result["findings"] if item["severity"] == "quarantine"]
    assert any(item["code"] == "identical_duplicate_source_records" for item in result["findings"])


def test_static_source_audit_quarantines_conflicting_or_missing_verifier_sources():
    task = "Do the task."
    row = {"task": task, "task_idx": 0}
    code = _verifier("code", task)
    changed = copy.deepcopy(code)
    changed["verification"]["code"] += "# conflict\n"

    result, _ = static_source_audit(
        row,
        task_records=[{"scenario": "sample", "tasks": [task]}],
        code_records=[code, changed],
        sql_records=[],
        sample_data={},
    )

    codes = {item["code"] for item in result["findings"] if item["severity"] == "quarantine"}
    assert codes == {"conflicting_code_verifiers", "missing_sql_verifier"}


def test_static_classification_keeps_infrastructure_separate_from_task_defects():
    base = {"findings": [], "runtime": _runtime()}
    assert classify_static_record(base) == ("pass", [])
    infra = {"findings": [], "runtime": {"status": "infrastructure_exhausted"}}
    assert classify_static_record(infra) == (
        "infrastructure_pending",
        ["runtime_infrastructure_exhausted"],
    )
    deterministic = {
        "findings": [],
        "runtime": {
            "status": "deterministic_failure",
            "reason": "invalid_canonical_tool_schema",
        },
    }
    assert classify_static_record(deterministic) == (
        "quarantine",
        ["invalid_canonical_tool_schema"],
    )
    no_op = {"findings": [], "runtime": _runtime(no_op_reward_type="complete")}
    assert classify_static_record(no_op) == (
        "quarantine",
        ["no_op_code_verifier_complete"],
    )


def test_cached_schema_error_is_migrated_to_deterministic_quarantine():
    record = {
        "task": "Create a record.",
        "findings": [],
        "semantic_warning_codes": [],
        "runtime": {
            "status": "infrastructure_exhausted",
            "errors": ["SchemaError: duplicate required field"] * 3,
        },
    }

    refreshed = refresh_cached_static_record(record, sample_data={})

    assert refreshed["runtime"]["status"] == "deterministic_failure"
    assert refreshed["status"] == "quarantine"
    assert refreshed["status_reasons"] == ["invalid_canonical_tool_schema"]


def test_judge_usage_can_be_accumulated_across_resume():
    assert _add_usage(
        {"requests": 2, "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        {"requests": 1, "prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
    ) == {
        "requests": 3,
        "prompt_tokens": 16,
        "completion_tokens": 7,
        "total_tokens": 23,
    }


def test_conservative_double_judge_requires_matching_high_confidence_sql_defect():
    agreed = [
        {
            "status": "ok",
            "verdict": {
                "verdict": "infeasible",
                "defect_kind": "preexisting_target_state",
                "confidence": 0.95,
                "affected_protocols": ["code", "sql"],
            },
        },
        {
            "status": "ok",
            "verdict": {
                "verdict": "infeasible",
                "defect_kind": "preexisting_target_state",
                "confidence": 0.91,
                "affected_protocols": ["sql"],
            },
        },
    ]
    assert judge_consensus(agreed) == (
        "quarantine",
        ["judge:preexisting_target_state"],
    )
    disagreed = copy.deepcopy(agreed)
    disagreed[1]["verdict"]["defect_kind"] = "unreachable_mutation"
    assert judge_consensus(disagreed)[0] == "needs_review"
    low_confidence = copy.deepcopy(agreed)
    low_confidence[1]["verdict"]["confidence"] = 0.89
    assert judge_consensus(low_confidence)[0] == "needs_review"
    infra = [agreed[0], {"status": "infrastructure_exhausted"}]
    assert judge_consensus(infra)[0] == "infrastructure_pending"


def test_judge_json_validation_and_calibration_selection_are_deterministic():
    parsed = _parse_judge_json('```json\n{"verdict":"feasible","defect_kind":"none","confidence":0.99,"affected_protocols":["sql"],"evidence":["reachable"]}\n```')
    assert parsed["verdict"] == "feasible"
    records = []
    forced = "enterprise_software_1:6"
    records.append(
        {
            "task_id": forced,
            "status": "pass",
            "semantic_warning_codes": ["schema_repaired"],
            "native_prompt_tokens": 8000,
        }
    )
    for index in range(40):
        records.append(
            {
                "task_id": f"scenario_{index}:0",
                "status": "pass",
                "semantic_warning_codes": [],
                "native_prompt_tokens": 1000 + index * 250,
            }
        )
    selected = select_judge_task_ids(records, maximum=17, clean_controls=16)
    assert selected[0] == forced
    assert len(selected) == 17
    assert selected == select_judge_task_ids(records, maximum=17, clean_controls=16)


def test_qualification_feedback_survives_integrity_resume_reclassification():
    feedback = {
        "protocol_version": 1,
        "qualification_manifest_sha256": "manifest",
        "qualification_trials_sha256": "trials",
        "reason": "qualification:repeated_judge_confirmed_environment_error",
        "evidence": {"http_status_codes": [422]},
    }
    prior = {
        "task_id": "scenario:0",
        "status": "quarantine",
        "status_reasons": [feedback["reason"]],
        "qualification_feedback": [feedback],
    }

    preserved = preserve_qualification_feedback(
        {
            "task_id": "scenario:0",
            "status": "pass",
            "status_reasons": [],
        },
        prior,
    )

    assert preserved["status"] == "quarantine"
    assert preserved["status_reasons"] == [feedback["reason"]]
    assert preserved["qualification_feedback"] == [feedback]


def test_prefilter_rejects_only_quarantine_and_preserves_other_statuses(tmp_path):
    rows = [
        {
            "task_id": f"scenario:{index}",
            "training_row": {
                "extra_info": {"task_id": f"scenario:{index}"},
                "env_kwargs": {"scenario": "scenario", "task_idx": index},
            },
        }
        for index in range(4)
    ]
    records = [
        {"task_id": "scenario:0", "status": "pass"},
        {"task_id": "scenario:1", "status": "needs_review"},
        {"task_id": "scenario:2", "status": "infrastructure_pending"},
        {"task_id": "scenario:3", "status": "quarantine"},
    ]

    fields = _write_prefilter_artifacts(rows, records, tmp_path)

    assert fields["prefilter_candidate_task_ids"] == [
        "scenario:0",
        "scenario:1",
        "scenario:2",
    ]
    assert fields["training_pool_task_ids"] == fields["prefilter_candidate_task_ids"]
    assert fields["rejected_prefilter_task_ids"] == ["scenario:3"]
    frame = pd.read_parquet(tmp_path / "awm_training_pool.parquet")
    statuses = [item["awm_integrity_status"] for item in frame["extra_info"]]
    assert statuses == ["pass", "needs_review", "infrastructure_pending"]
    assert all(item["awm_prefilter_status"] == "candidate" for item in frame["extra_info"])
    assert all(item["awm_training_pool_status"] == "active" for item in frame["extra_info"])
