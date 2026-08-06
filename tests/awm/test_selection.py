import json

import pandas as pd

from agent_system.environments.env_package.awm.data.selection import (
    CANDIDATE_FILENAME,
    SELECTION_MODE_ALL_ELIGIBLE,
    SELECTION_PROTOCOL_VERSION,
    audit_counts,
    one_per_environment,
    selection_rounds,
    verify_selection,
)
from agent_system.environments.env_package.awm.runtime.rollout import sha256_file


def test_native_prompt_audit_counts_and_environment_round_robin():
    records = [
        {"task_id": "a:0", "scenario": "a", "native_prompt_tokens": 100},
        {"task_id": "a:1", "scenario": "a", "native_prompt_tokens": 200},
        {"task_id": "b:0", "scenario": "b", "native_prompt_tokens": 150},
        {"task_id": "b:1", "scenario": "b", "native_prompt_tokens": 17000},
        {"task_id": "c:0", "scenario": "c", "native_prompt_tokens": 18000},
    ]
    assert audit_counts(records, 16000) == {
        "tasks": 5,
        "eligible_tasks": 3,
        "eligible_environments": 2,
        "all_tasks_eligible_environments": 1,
    }
    rounds = selection_rounds(
        {
            "a": [{"task_id": "a:0"}, {"task_id": "a:1"}],
            "b": [{"task_id": "b:0"}],
        }
    )
    assert {scenario for scenario, rank, _ in rounds[:2] if rank == 0} == {"a", "b"}
    assert rounds[-1][1] == 1


def test_one_per_environment_drops_only_later_environment_rounds():
    records = [
        {"task_id": "a:0", "scenario": "a"},
        {"task_id": "b:0", "scenario": "b"},
        {"task_id": "a:1", "scenario": "a"},
    ]
    assert [record["task_id"] for record in one_per_environment(records)] == ["a:0", "b:0"]


def test_all_context_selection_verifies_without_per_task_preflight(tmp_path):
    task_ids = [f"scenario:{index}" for index in range(10)]
    candidate_path = tmp_path / CANDIDATE_FILENAME
    pd.DataFrame(
        [
            {
                "extra_info": {
                    "task_id": task_id,
                    "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
                }
            }
            for task_id in task_ids
        ]
    ).to_parquet(candidate_path, index=False)
    audit_path = tmp_path / "native_prompt_audit.jsonl"
    audit_path.write_text("\n".join(json.dumps({"task_id": item}) for item in task_ids) + "\n")
    summary_path = tmp_path / "audit_summary.json"
    summary_path.write_text("{}\n")
    preflight_path = tmp_path / "preflight.jsonl"
    preflight_path.write_text("")
    manifest = {
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "selection_mode": SELECTION_MODE_ALL_ELIGIBLE,
        "native_prompt_cutoff": 16000,
        "target_tasks": 10,
        "task_ids": task_ids,
        "records": [
            {
                "task_id": task_id,
                "scenario": "scenario",
                "task_idx": index,
                "native_prompt_tokens": 100,
                "preflight": None,
            }
            for index, task_id in enumerate(task_ids)
        ],
        "audit_counts": {"eligible_tasks": 10, "eligible_environments": 1},
        "selected_counts": {
            "tasks": 10,
            "environments": 1,
            "max_tasks_per_environment": 10,
        },
        "native_prompt_audit_sha256": sha256_file(audit_path),
        "audit_summary_sha256": sha256_file(summary_path),
        "preflight_sha256": sha256_file(preflight_path),
        "candidate_data_sha256": sha256_file(candidate_path),
    }
    (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest))

    verify_selection(tmp_path)
