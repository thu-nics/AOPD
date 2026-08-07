import json

from agent_system.environments.env_package.awm.screening.expert import (
    _append_jsonl,
    _load_jsonl,
    _migrate_legacy_trial,
    new_task_limit,
    screening_result_status,
    screening_rollout_protocol,
    task_resolution,
    usage_from_trials,
    validate_trial_records,
)


def test_screening_accepts_only_success():
    assert screening_result_status({"reward_type": "complete", "success": True}) == "success"
    for reward_type in ("others", "incomplete", "agent_error"):
        assert screening_result_status({"reward_type": reward_type, "success": False}) == "failure"
    for reward_type in ("server_error", "no_verifier", "runtime_exception"):
        assert screening_result_status({"reward_type": reward_type, "success": False}) == "infrastructure_error"


def test_intermediate_infrastructure_error_is_retried_unless_trajectory_succeeds():
    failed = {
        "reward_type": "others",
        "success": False,
        "trajectory": [{"runtime_infrastructure_error": True}],
    }
    assert screening_result_status(failed) == "infrastructure_error"
    assert screening_result_status({**failed, "reward_type": "complete", "success": True}) == "success"


def test_task_resolution_keeps_only_success_passed():
    records = {
        "success:0": {"status": "success"},
        "failure:0": {"status": "failure"},
        "infra:0": {"status": "infrastructure_exhausted"},
    }
    assert task_resolution("success:0", records) == "passed"
    assert task_resolution("failure:0", records) == "failed"
    assert task_resolution("infra:0", records) == "infrastructure_failed"
    assert task_resolution("missing:0", records) == "pending"


def test_screening_uses_w6_and_reserves_the_configured_response_budget():
    assert screening_rollout_protocol(4096) == {
        "history_window": 6,
        "history_unit": "complete_action_result_exchange",
        "history_prefix": "system_and_task_pinned",
        "model_context_tokens": 32000,
        "max_prompt_tokens": 27904,
        "context_response_reserve_tokens": 4096,
        "max_decisions": 20,
    }


def test_five_percent_limit_is_ceil_and_does_not_change_candidate_scope():
    assert new_task_limit(6985, max_new_tasks=None, max_new_task_fraction=0.05) == 350
    assert new_task_limit(100, max_new_tasks=7, max_new_task_fraction=None) == 7
    assert new_task_limit(4, max_new_tasks=None, max_new_task_fraction=None) == 4


def test_resumable_jsonl_repairs_only_an_incomplete_tail(tmp_path):
    path = tmp_path / "trials.jsonl"
    first = {"task_id": "scenario:0", "seed": 300}
    path.write_text(json.dumps(first) + '\n{"task_id":')

    assert _load_jsonl(path, repair_torn_tail=True) == [first]
    _append_jsonl(path, {"task_id": "scenario:1", "seed": 300})
    assert [record["task_id"] for record in _load_jsonl(path)] == [
        "scenario:0",
        "scenario:1",
    ]


def test_usage_includes_deepseek_cache_tokens():
    records = [
        {
            "result": {
                "trajectory": [
                    {
                        "usage": {
                            "prompt_tokens": 100,
                            "prompt_cache_hit_tokens": 80,
                            "prompt_cache_miss_tokens": 20,
                            "completion_tokens": 10,
                            "total_tokens": 110,
                        }
                    }
                ]
            }
        }
    ]
    expected = {
        "requests": 1,
        "prompt_tokens": 100,
        "prompt_cache_hit_tokens": 80,
        "prompt_cache_miss_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 110,
    }
    assert usage_from_trials(records) == expected
    assert usage_from_trials([{"usage": expected, "result": {"trajectory": []}}]) == expected


def test_trial_validation_has_no_replay_contract():
    records = [
        {"task_id": "scenario:0", "seed": 300, "status": "success", "result": {"success": True}},
        {"task_id": "scenario:1", "seed": 300, "status": "failure", "result": {"success": False}},
        {"task_id": "scenario:2", "seed": 300, "status": "infrastructure_exhausted"},
    ]
    validate_trial_records(records, {"scenario:0", "scenario:1", "scenario:2"})


def test_legacy_migration_strips_replay_only_fields_and_keeps_evidence():
    migrated = _migrate_legacy_trial(
        {
            "task_id": "scenario:0",
            "seed": 300,
            "status": "environment_failure",
            "runtime_replay": {"status": "confirmed"},
            "result": {
                "success": True,
                "verify_observation_signature": "obsolete",
                "trajectory": [
                    {
                        "parsed_action": "action",
                        "tool_response": "observation",
                        "tool_observation_signature": "obsolete",
                    }
                ],
            },
        }
    )
    assert migrated["status"] == "failure"
    assert migrated["legacy_status"] == "environment_failure"
    assert "runtime_replay" not in migrated
    assert "verify_observation_signature" not in migrated["result"]
    assert "tool_observation_signature" not in migrated["result"]["trajectory"][0]
    assert migrated["result"]["trajectory"][0]["tool_response"] == "observation"
