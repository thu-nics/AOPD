import json

from agent_system.environments.env_package.awm.screening.expert import (
    new_task_limit,
    runtime_failure_candidate,
    screening_result_status,
    screening_rollout_protocol,
    task_resolution,
    usage_from_trials,
    validate_trial_records,
)


def test_screening_accepts_success_and_normal_code_verifier_failures():
    assert screening_result_status({"reward_type": "complete", "success": True}) == "success"
    for reward_type in ("others", "incomplete", "agent_error"):
        assert screening_result_status({"reward_type": reward_type, "success": False}) == "policy_failure"
    for reward_type in ("server_error", "no_verifier", "runtime_exception"):
        assert screening_result_status({"reward_type": reward_type, "success": False}) == "infrastructure_error"


def test_task_resolution_keeps_unattempted_tasks_pending():
    records = {
        "success:0": {"status": "success"},
        "failure:0": {"status": "policy_failure"},
        "environment:0": {"status": "environment_failure"},
        "infra:0": {"status": "infrastructure_exhausted"},
    }
    assert task_resolution("success:0", records) == "accepted_success"
    assert task_resolution("failure:0", records) == "accepted_policy_failure"
    assert task_resolution("environment:0", records) == "rejected_environment"
    assert task_resolution("infra:0", records) == "infrastructure_pending"
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
    assert new_task_limit(7845, max_new_tasks=None, max_new_task_fraction=0.05) == 393
    assert new_task_limit(100, max_new_tasks=7, max_new_task_fraction=None) == 7
    assert new_task_limit(4, max_new_tasks=None, max_new_task_fraction=None) == 4


def test_runtime_failure_candidate_prefers_first_strong_tool_error():
    result = {
        "trajectory": [
            {"runtime_infrastructure_error": False},
            {"runtime_infrastructure_error": True, "runtime_error_signature": "tool-signature"},
        ],
        "verify_infrastructure_error": True,
        "verify_error_signature": "verify-signature",
    }
    assert runtime_failure_candidate(result) == {
        "phase": "tool",
        "trajectory_index": 1,
        "expected_signature": "tool-signature",
    }
    assert runtime_failure_candidate(
        {
            "trajectory": [],
            "verify_infrastructure_error": True,
            "verify_error_signature": "verify-signature",
        }
    ) == {
        "phase": "verify",
        "trajectory_index": None,
        "expected_signature": "verify-signature",
    }


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
    assert usage_from_trials(records) == {
        "requests": 1,
        "prompt_tokens": 100,
        "prompt_cache_hit_tokens": 80,
        "prompt_cache_miss_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 110,
    }


def test_environment_rejection_requires_confirmed_replay():
    record = {
        "task_id": "scenario:0",
        "seed": 300,
        "status": "environment_failure",
        "result": {"success": False},
        "runtime_replay": {"status": "confirmed"},
    }
    validate_trial_records([record], {"scenario:0"})
    unconfirmed = json.loads(json.dumps(record))
    unconfirmed["runtime_replay"]["status"] = "pending"
    try:
        validate_trial_records([unconfirmed], {"scenario:0"})
    except RuntimeError as exc:
        assert "lacks confirmed replay" in str(exc)
    else:
        raise AssertionError("unconfirmed environment failure was accepted")
