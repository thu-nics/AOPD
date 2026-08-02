import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import agent_system.environments.env_package.awm.qualification as qualification_module
from agent_system.environments.env_package.awm.integrity import INTEGRITY_PROTOCOL_VERSION
from agent_system.environments.env_package.awm.native_rollout import sha256_file, summarize_results
from agent_system.environments.env_package.awm.qualification import (
    _load_jsonl,
    cumulative_usage_from_trials,
    environment_balanced,
    load_candidate_rows,
    provider_identity_from_trials,
    qualification_result_status,
    qualification_rollout_protocol,
    select_qwen_diagnostic,
    task_resolution,
    validate_trial_records,
)
from agent_system.environments.env_package.awm.selection import (
    SELECTION_PROTOCOL_VERSION,
    audit_counts,
    selection_rounds,
)


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
    eligible = {
        "a": [{"task_id": "a:0"}, {"task_id": "a:1"}],
        "b": [{"task_id": "b:0"}],
    }
    rounds = selection_rounds(eligible)
    first_round = rounds[:2]
    assert {scenario for scenario, rank, _ in first_round if rank == 0} == {"a", "b"}
    assert rounds[-1][1] == 1


def test_qualification_v8_binds_rollout_context_and_action_budget(monkeypatch):
    assert qualification_rollout_protocol() == {
        "history_window": 3,
        "history_unit": "complete_action_result_exchange",
        "history_prefix": "system_and_task_pinned",
        "model_context_tokens": 32000,
        "max_prompt_tokens": 29952,
        "context_response_reserve_tokens": 2048,
        "max_decisions": 20,
    }
    monkeypatch.setattr(qualification_module, "HISTORY_WINDOW", 10)
    with pytest.raises(RuntimeError, match="protocol-version bump"):
        qualification_rollout_protocol()


def test_qualification_resolution_requires_exactly_four_successes():
    records = {
        "qualified": [{"trial_index": index, "status": "success"} for index in range(4)],
        "failed": [
            {"trial_index": 0, "status": "success"},
            {"trial_index": 1, "status": "policy_failure"},
        ],
        "infra": [{"trial_index": 0, "status": "infrastructure_exhausted"}],
    }
    assert task_resolution("qualified", records) == "qualified"
    assert task_resolution("failed", records) == "rejected_policy"
    assert task_resolution("infra", records) == "rejected_infrastructure"
    assert task_resolution("missing", records) == "pending"


def test_sql_judge_infrastructure_is_not_a_policy_failure():
    assert qualification_result_status({"reward_type": "complete", "success": True}) == "success"
    for reward_type in ("incomplete", "agent_error"):
        assert qualification_result_status({"reward_type": reward_type, "success": False}) == "policy_failure"
    for reward_type in ("judge_error", "server_error", "no_verifier", "unexpected"):
        assert qualification_result_status({"reward_type": reward_type, "success": False}) == "infrastructure_error"


def test_qwen_diagnostic_is_distinct_environment_and_uses_32_tasks():
    rows = [
        {
            "task_id": f"scenario_{index}:0",
            "scenario": f"scenario_{index}",
            "task_idx": 0,
            "task": "task",
            "native_prompt_tokens": 5000 + index * 200,
            "training_row": {},
        }
        for index in range(40)
    ]
    trials = {
        row["task_id"]: [
            {
                "trial_index": trial,
                "status": "success",
                "result": {"decisions": 1 + (index % 7)},
            }
            for trial in range(4)
        ]
        for index, row in enumerate(rows)
    }

    selected = select_qwen_diagnostic(rows, trials)

    assert len(selected) == 32
    assert len({row["scenario"] for row in selected}) == 32
    assert {row["native_prompt_quartile"] for row in selected} == {0, 1, 2, 3}


def test_environment_balanced_orders_one_task_per_environment_first():
    rows = [
        {"task_id": "a:0", "scenario": "a"},
        {"task_id": "a:1", "scenario": "a"},
        {"task_id": "b:0", "scenario": "b"},
    ]
    balanced = environment_balanced(rows)
    assert len({row["scenario"] for row in balanced[:2]}) == 2


def test_native_summary_reports_parse_and_execution_failures():
    results = [
        {
            "success": True,
            "reward": 1,
            "trajectory": [
                {
                    "action_kind": "tool",
                    "tool_response_is_error": True,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                {
                    "action_kind": "invalid",
                    "parse_error": "multiple tool calls",
                    "usage": {},
                },
                {"action_kind": "message", "usage": {}},
            ],
        }
    ]

    summary = summarize_results(results)

    assert summary["success_rate"] == 1.0
    assert summary["valid_action_rate"] == 2 / 3
    assert summary["tool_execution_error_rate"] == 1.0
    assert summary["invalid_reasons"] == {"multiple tool calls": 1}
    assert json.dumps(summary)


def test_resumed_trial_identity_and_cumulative_usage():
    trials = [
        {
            "status": "success",
            "result": {
                "trajectory": [
                    {
                        "model": "deepseek-v4-flash",
                        "system_fingerprint": "revision-0731",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    },
                    {
                        "model": "deepseek-v4-flash",
                        "system_fingerprint": "revision-0731",
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 4,
                            "total_tokens": 24,
                        },
                    },
                ]
            },
        }
    ]
    assert provider_identity_from_trials(trials) == {
        "model": "deepseek-v4-flash",
        "system_fingerprint": "revision-0731",
    }
    assert cumulative_usage_from_trials(trials) == {
        "requests": 2,
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "total_tokens": 37,
    }


def test_small_qwen_diagnostic_keeps_metadata():
    rows = [
        {
            "task_id": f"scenario_{index}:0",
            "scenario": f"scenario_{index}",
            "native_prompt_tokens": 5000 + index * 1000,
        }
        for index in range(4)
    ]
    trials = {
        row["task_id"]: [
            {
                "trial_index": trial,
                "status": "success",
                "result": {"decisions": index + 1},
            }
            for trial in range(4)
        ]
        for index, row in enumerate(rows)
    }

    selected = select_qwen_diagnostic(rows, trials)

    assert len(selected) == 4
    assert all(row.get("native_prompt_quartile") is not None for row in selected)
    assert all(row.get("expert_max_decisions") is not None for row in selected)


def test_trial_validation_rejects_duplicates_and_load_ignores_torn_tail(tmp_path):
    record = {
        "task_id": "scenario:0",
        "trial_index": 0,
        "seed": 300,
        "status": "policy_failure",
        "result": {"success": False, "trajectory": []},
    }
    validate_trial_records([record], {"scenario:0"})
    with pytest.raises(RuntimeError, match="duplicate qualification trial"):
        validate_trial_records([record, record], {"scenario:0"})

    path = tmp_path / "trials.jsonl"
    path.write_text(json.dumps(record) + "\n{", encoding="utf-8")
    assert _load_jsonl(path) == [record]


def test_trial_validation_rejects_seed_and_status_result_mismatch():
    record = {
        "task_id": "scenario:0",
        "trial_index": 0,
        "seed": 301,
        "status": "success",
        "result": {"success": False, "trajectory": []},
    }
    with pytest.raises(RuntimeError, match="seed mismatch"):
        validate_trial_records([record], {"scenario:0"})


def test_qualification_accepts_only_hash_bound_integrity_filtered_rows(tmp_path):
    selection_data = tmp_path / "selection.parquet"
    filtered_data = tmp_path / "filtered.parquet"
    selection_manifest_path = tmp_path / "candidate_manifest.json"
    integrity_manifest_path = tmp_path / "integrity_manifest.json"
    rows = [
        {
            "extra_info": {
                "task_id": "scenario:0",
                "task": "Do it",
                "native_prompt_tokens": 100,
                "tool_schema_hash": "canonical",
                "raw_tool_schema_hash": "raw",
                "tool_schema_repair_count": 1,
            },
            "env_kwargs": {"scenario": "scenario", "task_idx": 0},
        },
        {
            "extra_info": {
                "task_id": "scenario:1",
                "task": "Do the other",
                "native_prompt_tokens": 110,
                "tool_schema_hash": "canonical",
                "raw_tool_schema_hash": "raw",
                "tool_schema_repair_count": 1,
            },
            "env_kwargs": {"scenario": "scenario", "task_idx": 1},
        },
    ]
    pd.DataFrame(rows).to_parquet(selection_data, index=False)
    pd.DataFrame(rows[:1]).to_parquet(filtered_data, index=False)
    selection_manifest = {
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "candidate_data_sha256": sha256_file(selection_data),
        "task_ids": ["scenario:0", "scenario:1"],
    }
    selection_manifest_path.write_text(json.dumps(selection_manifest))
    integrity_manifest = {
        "protocol_version": INTEGRITY_PROTOCOL_VERSION,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "filtered_data_sha256": sha256_file(filtered_data),
        "filtered_task_ids": ["scenario:0"],
    }
    integrity_manifest_path.write_text(json.dumps(integrity_manifest))

    loaded, manifest, integrity = load_candidate_rows(
        filtered_data,
        selection_manifest_path,
        integrity_manifest_path,
    )

    assert [row["task_id"] for row in loaded] == ["scenario:0"]
    assert manifest == selection_manifest
    assert integrity == integrity_manifest
    integrity_manifest["selection_manifest_sha256"] = "wrong"
    integrity_manifest_path.write_text(json.dumps(integrity_manifest))
    with pytest.raises(RuntimeError, match="selection-manifest mismatch"):
        load_candidate_rows(filtered_data, selection_manifest_path, integrity_manifest_path)


def test_qualification_v8_rejects_legacy_non_prefilter_pool(tmp_path, monkeypatch):
    data_path = tmp_path / "legacy_filtered.parquet"
    data_path.write_bytes(b"legacy")
    monkeypatch.setattr(
        qualification_module,
        "load_candidate_rows",
        lambda *_: ([], {}, {"prefilter_data_sha256": "different"}),
    )
    args = SimpleNamespace(
        data=data_path,
        candidate_manifest=tmp_path / "candidate_manifest.json",
        integrity_manifest=tmp_path / "integrity_manifest.json",
    )

    with pytest.raises(RuntimeError, match="requires the hash-bound"):
        asyncio.run(qualification_module.qualify(args))
