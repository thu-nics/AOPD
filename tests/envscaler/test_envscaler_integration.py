import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from agent_system.environments.env_package.envscaler.data import (
    EnvironmentRoundRobin,
    materialize_mixed_schedule,
)
from agent_system.environments.env_package.envscaler.envs import interleave_families
from agent_system.environments.env_package.envscaler.filtering import audit_task
from agent_system.environments.env_package.envscaler.runtime import EnvScalerWorker
from agent_system.environments.env_package.envscaler.screening import (
    JUDGE_MAX_TOKENS,
    JUDGE_PROTOCOL_VERSION,
    SCREENING_PROTOCOL_VERSION,
    DeepSeekScreeningClient,
    _aggregate_api_usage,
    _is_completed_review,
    _legacy_review_needs_refresh,
    _load_existing_screening_records,
    _load_validated_deterministic_records,
    _resume_config_matches,
    _validate_resume_record,
    build_code_evidence,
    screen_one,
)
from agent_system.environments.env_package.envscaler.source import (
    ENVSCALER_COMMIT,
    EXPECTED_ENV_COUNT,
    EXPECTED_RL_ENV_COUNT,
    EXPECTED_RL_TASK_COUNT,
    EXPECTED_TASKS_PER_RL_ENV,
    load_envscaler_source,
    sha256_file,
)
from agent_system.environments.env_package.envscaler.user_simulator import (
    STOP,
    DeepSeekUserSimulator,
)


def _compose(config_name):
    config_dir = Path(__file__).parents[2] / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=config_name)


def _row(family, environment_id, task_id, task_index):
    extra = {
        "env_family": family,
        "task_id": task_id,
        "task_index": task_index,
    }
    if family == "awm":
        extra["scenario"] = environment_id
        env_kwargs = {
            "env_family": family,
            "scenario": environment_id,
            "task_idx": task_index,
        }
    else:
        extra["env_id"] = environment_id
        env_kwargs = {
            "env_family": family,
            "task_index": task_index,
            "task_id": task_id,
        }
    return {
        "data_source": family,
        "prompt": [{"role": "user", "content": task_id}],
        "ability": "agentic_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra,
        "env_kwargs": env_kwargs,
    }


def test_pinned_source_identity_and_rl_shape():
    source = load_envscaler_source()
    counts = {}
    for task in source.tasks:
        env_id = str(task["env_id"])
        counts[env_id] = counts.get(env_id, 0) + 1

    assert source.identity["commit"] == ENVSCALER_COMMIT
    assert len(source.environments) == EXPECTED_ENV_COUNT
    assert len(source.tasks) == EXPECTED_RL_TASK_COUNT
    assert len(counts) == EXPECTED_RL_ENV_COUNT
    assert set(counts.values()) == {EXPECTED_TASKS_PER_RL_ENV}


def test_deterministic_audit_accepts_known_good_and_rejects_exact_duplicate():
    assert audit_task(0)["status"] == "pass"
    duplicate = audit_task(64)
    assert duplicate["status"] == "quarantine"
    assert "exact_duplicate_checker" in duplicate["reasons"]
    assert len(duplicate["diagnostics"]["initial_state_sha256"]) == 64


def test_user_simulator_uses_plain_messages_and_disables_thinking():
    payloads = []
    replies = iter(["Hello, please help me.", "###STOP###"])

    def request(payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": next(replies)}}]}

    simulator = DeepSeekUserSimulator(request_fn=request)
    assert simulator.start("Update the record") == "Hello, please help me."
    assert simulator.reply("Done") == STOP
    assert all(item["thinking"] == {"type": "disabled"} for item in payloads)
    assert all(item["temperature"] == 1.0 for item in payloads)
    assert all("top_p" not in item and "max_tokens" not in item for item in payloads)


def test_user_simulator_rejects_sampling_protocol_drift():
    with pytest.raises(ValueError, match="temperature=1"):
        DeepSeekUserSimulator(temperature=0.0, request_fn=lambda payload: {})
    with pytest.raises(ValueError, match="disabled.*reasoning"):
        DeepSeekUserSimulator(reasoning_enabled=True, request_fn=lambda payload: {})


def test_user_simulator_accepts_legacy_reply_wrapper_only_for_compatibility():
    assert DeepSeekUserSimulator.parse_reply("# Reply: hello") == "hello"
    assert DeepSeekUserSimulator.parse_reply("analysis\n###STOP###") == STOP


def test_screening_usage_is_reconstructed_from_durable_records():
    records = {
        0: {
            "health_review_history": [
                {
                    "judge": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        }
                    }
                }
            ],
            "health_review": {
                "judge": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                    }
                }
            },
        }
    }
    assert _aggregate_api_usage(records) == {
        "successful_responses": 2,
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }


def test_screening_usage_counts_structured_http_responses():
    records = {
        0: {
            "health_review": {
                "judge": {
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 11,
                        "total_tokens": 51,
                    },
                    "structured_response_attempts": 2,
                }
            }
        },
        1: {
            "health_review": {
                "judge": None,
                "judge_attempt_usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 24,
                    "total_tokens": 84,
                },
                "judge_attempt_errors": ["empty", "empty", "truncated"],
            }
        },
    }

    assert _aggregate_api_usage(records) == {
        "successful_responses": 5,
        "prompt_tokens": 100,
        "completion_tokens": 35,
        "total_tokens": 135,
    }


def test_deterministic_audit_and_manifest_are_cross_validated(tmp_path):
    tasks = (
        {"task_id": "task-0", "env_id": "env-0"},
        {"task_id": "task-1", "env_id": "env-1"},
    )
    records = [
        {
            "protocol_version": 1,
            "task_index": 0,
            "task_id": "task-0",
            "env_id": "env-0",
            "status": "pass",
            "reasons": [],
            "diagnostics": {},
        },
        {
            "protocol_version": 1,
            "task_index": 1,
            "task_id": "task-1",
            "env_id": "env-1",
            "status": "quarantine",
            "reasons": ["missing_checkers"],
            "diagnostics": {},
        },
    ]
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "total_tasks": 2,
        "pass_tasks": 1,
        "quarantine_tasks": 1,
        "pass_task_indices": [0],
        "quarantine": [
            {
                "task_index": 1,
                "task_id": "task-1",
                "reasons": ["missing_checkers"],
            }
        ],
    }

    loaded = _load_validated_deterministic_records(audit_path, manifest, tasks)
    assert sorted(loaded) == [0, 1]

    corrupted = {**manifest, "pass_task_indices": [0, 1]}
    with pytest.raises(RuntimeError, match="manifest/audit mismatch"):
        _load_validated_deterministic_records(audit_path, corrupted, tasks)

    audit_path.write_text(
        "".join(json.dumps(record) + "\n" for record in [*records, records[0]]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate task_index"):
        _load_validated_deterministic_records(audit_path, manifest, tasks)


def test_screening_resume_records_are_cross_validated(tmp_path):
    deterministic = {
        "protocol_version": 1,
        "task_index": 0,
        "task_id": "task-0",
        "env_id": "env-0",
        "status": "pass",
        "reasons": [],
        "diagnostics": {},
    }
    record = {
        **deterministic,
        "health_review": {
            "task_index": 0,
            "accepted": True,
            "status": "healthy",
            "status_reason": "healthy",
            "judge": {
                "protocol_version": JUDGE_PROTOCOL_VERSION,
                "label": "healthy",
            },
        },
    }
    audit_path = tmp_path / "screening.jsonl"
    audit_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = _load_existing_screening_records(audit_path)
    _validate_resume_record(0, loaded[0], deterministic)

    audit_path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate task_index"):
        _load_existing_screening_records(audit_path)

    mismatched = {
        **record,
        "health_review": {
            **record["health_review"],
            "accepted": False,
        },
    }
    with pytest.raises(RuntimeError, match="quarantine verdict mismatch"):
        _validate_resume_record(0, mismatched, deterministic)


def test_code_augmented_evidence_has_no_expert_trajectory():
    deterministic = audit_task(0)
    evidence = build_code_evidence(0, deterministic_record=deterministic)

    assert evidence["task"]["task_id"] == deterministic["task_id"]
    assert evidence["environment"]["native_tools"]
    assert evidence["task"]["checkers"]
    assert evidence["no_action_checker_evidence"]["summary"]["state_complete"] is False
    initialization = evidence["runtime_initialization_protocol"]
    assert initialization["authoritative_state"] == "task.initial_state"
    assert initialization["native_helper"] == "init_env_instance"
    assert any("setattr" in step for step in initialization["steps"])
    encoded = json.dumps(evidence)
    assert "expert" not in encoded
    assert "trajectory" not in encoded


def test_code_augmented_judge_uses_awm_style_deepseek_protocol():
    payloads = []
    client = object.__new__(DeepSeekScreeningClient)
    client.max_retries = 3
    client.model = "deepseek-v4-flash"
    client.post = lambda payload: (
        payloads.append(payload)
        or {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "healthy",
                                "confidence": 1,
                                "rationale": "coherent",
                                "evidence": ["tools match checkers"],
                            }
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 10},
        }
    )

    result = client.judge({"environment": {}, "task": {}})

    assert result["protocol_version"] == JUDGE_PROTOCOL_VERSION
    assert result["label"] == "healthy"
    payload = payloads[0]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == JUDGE_MAX_TOKENS
    assert payload["response_format"] == {"type": "json_object"}
    assert "trajectory" not in payload["messages"][1]["content"]
    assert result["structured_response_attempts"] == 1
    assert result["structured_response_errors"] == []


def test_code_augmented_judge_retries_empty_content(monkeypatch):
    monkeypatch.setattr(
        "agent_system.environments.env_package.envscaler.screening.time.sleep",
        lambda _seconds: None,
    )
    valid = {
        "label": "healthy",
        "confidence": 90,
        "rationale": "coherent",
        "evidence": ["checker is faithful"],
    }
    responses = iter(
        [
            {
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
            {
                "choices": [{"message": {"content": json.dumps(valid)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            },
        ]
    )
    client = object.__new__(DeepSeekScreeningClient)
    client.max_retries = 3
    client.model = "deepseek-v4-flash"
    client.post = lambda _payload: next(responses)

    result = client.judge({"environment": {}, "task": {}})

    assert result["label"] == "healthy"
    assert result["structured_response_attempts"] == 2
    assert result["structured_response_errors"] == ["ValueError: screening judge returned empty content"]
    assert result["usage"]["total_tokens"] == 51


def test_code_augmented_judge_preserves_usage_before_later_request_failure(monkeypatch):
    monkeypatch.setattr(
        "agent_system.environments.env_package.envscaler.screening.time.sleep",
        lambda _seconds: None,
    )
    responses = iter(
        [
            {
                "choices": [{"message": {"content": ""}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            },
            RuntimeError("connection reset"),
        ]
    )
    client = object.__new__(DeepSeekScreeningClient)
    client.max_retries = 3
    client.model = "deepseek-v4-flash"

    def post(_payload):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    client.post = post
    with pytest.raises(RuntimeError, match="judge request failed") as caught:
        client.judge({"environment": {}, "task": {}})

    assert caught.value.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    }
    assert caught.value.structured_errors == ["ValueError: screening judge returned empty content"]


def test_code_augmented_judge_accepts_fenced_json_object():
    value = {
        "label": "uncertain",
        "confidence": 50,
        "rationale": "insufficient evidence",
        "evidence": [],
    }
    content = "Result follows:\n```json\n" + json.dumps(value) + "\n```"
    assert DeepSeekScreeningClient._parse_judge_content(content) == value


def _screening_record(label, rationale, protocol_version=2):
    status_reason = "healthy" if label == "healthy" else label
    return {
        "health_review": {
            "status_reason": status_reason,
            "judge": {
                "protocol_version": protocol_version,
                "label": label,
                "rationale": rationale,
                "evidence": [],
            },
        }
    }


def test_resume_selectively_refreshes_legacy_verdicts():
    infra = {"health_review": {"status_reason": "judge_infrastructure_exhausted"}}
    healthy = _screening_record("healthy", "coherent")
    clear_failure = _screening_record(
        "environment_or_verifier_failure",
        "Two checkers require mutually exclusive final values.",
    )
    initialization_error = _screening_record(
        "environment_or_verifier_failure",
        "The constructor never loads init_config, leaving runtime state empty.",
    )
    broad_contract_error = _screening_record(
        "environment_or_verifier_failure",
        "The task is satisfiable, but an unused tool has a misleading contract.",
    )
    current_failure = _screening_record(
        "environment_or_verifier_failure",
        "The task is satisfiable, but its checker is unsound.",
        protocol_version=JUDGE_PROTOCOL_VERSION,
    )

    assert _is_completed_review(infra) is False
    assert _is_completed_review(healthy) is True
    assert _is_completed_review(clear_failure) is True
    assert _legacy_review_needs_refresh(initialization_error["health_review"]) is True
    assert _is_completed_review(initialization_error) is False
    assert _is_completed_review(broad_contract_error) is False
    assert _is_completed_review(current_failure) is True


def test_resume_config_allows_reviewed_protocol_and_budget_migrations():
    current = {
        "protocol_version": 3,
        "judge_protocol_version": 3,
        "max_tokens": JUDGE_MAX_TOKENS,
        "model": "deepseek-v4-flash",
        "eligible_task_indices": [0, 1],
    }
    legacy = {
        **current,
        "protocol_version": 2,
        "judge_protocol_version": 2,
        "max_tokens": 8192,
    }
    v3_8k = {**current, "max_tokens": 8192}
    v3_16k = {**current, "max_tokens": 16_384}

    assert _resume_config_matches(current, current) is True
    assert _resume_config_matches(legacy, current) is True
    assert _resume_config_matches(v3_8k, current) is True
    assert _resume_config_matches(v3_16k, current) is True
    assert _resume_config_matches({**legacy, "model": "other"}, current) is False
    assert _resume_config_matches({**v3_8k, "max_tokens": 4096}, current) is False


class _StaticJudge:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def judge(self, evidence):
        assert "environment" in evidence and "task" in evidence
        return {
            "protocol_version": JUDGE_PROTOCOL_VERSION,
            "label": self.label,
            "confidence": self.confidence,
            "rationale": "test",
            "evidence": [],
            "usage": {},
        }


def test_healthy_membership_does_not_gate_on_confidence():
    deterministic = audit_task(0)
    source_root = load_envscaler_source().root

    healthy = screen_one(
        0,
        source_root=source_root,
        client=_StaticJudge("healthy", 0),
        deterministic_record=deterministic,
    )
    uncertain = screen_one(
        0,
        source_root=source_root,
        client=_StaticJudge("uncertain", 100),
        deterministic_record=deterministic,
    )

    assert healthy["accepted"] is True
    assert healthy["status"] == "healthy"
    assert uncertain["accepted"] is False
    assert uncertain["status_reason"] == "uncertain"


def test_mixed_family_interleave_has_exact_deterministic_quotas():
    labels = interleave_families({"awm": 48, "envscaler": 16})
    assert len(labels) == 64
    assert labels.count("awm") == 48
    assert labels.count("envscaler") == 16
    assert labels == interleave_families({"awm": 48, "envscaler": 16})


def test_preflight_failure_still_emits_terminal_checker_diagnostics():
    worker_class = EnvScalerWorker.__ray_metadata__.modified_class
    worker = worker_class(oracle_actor=object())
    worker._task = {
        "task_id": "task",
        "env_id": "env",
        "checklist_with_func": [],
    }
    worker._task_index = 0
    worker._runtime = type("Runtime", (), {})()
    worker._initial_state = {}
    worker._chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    worker._tools = []
    worker._reset_failure = "simulator unavailable"

    ready, info = asyncio.run(worker.prepare_state_group())

    assert ready is False
    assert info["terminal_outcome_valid"] is True
    assert info["terminal_success"] is False
    assert info["terminal_reward"] == 0.0


def test_round_robin_cycles_environments_before_reusing_one():
    scheduler = EnvironmentRoundRobin(
        [
            _row("envscaler", "env-b", "task-b0", 0),
            _row("envscaler", "env-a", "task-a1", 1),
            _row("envscaler", "env-a", "task-a0", 2),
        ],
        family="envscaler",
    )
    selected = [scheduler.next()["extra_info"]["task_id"] for _ in range(4)]
    assert selected == ["task-a0", "task-b0", "task-a1", "task-b0"]


def test_mixed_schedule_preserves_per_step_family_slots(tmp_path):
    awm_rows = [
        _row("awm", "scenario-b", "awm-b", 0),
        _row("awm", "scenario-a", "awm-a", 1),
    ]
    envscaler_rows = [
        _row("envscaler", "env-b", "es-b", 0),
        _row("envscaler", "env-a", "es-a", 1),
    ]
    awm_data = tmp_path / "awm.parquet"
    envscaler_data = tmp_path / "envscaler.parquet"
    health_path = tmp_path / "health.json"
    output_data = tmp_path / "mixed.parquet"
    output_manifest = tmp_path / "mixed.json"
    pd.DataFrame(awm_rows).to_parquet(awm_data, index=False)
    pd.DataFrame(envscaler_rows).to_parquet(envscaler_data, index=False)
    health_path.write_text(
        json.dumps(
            {
                "kind": "envscaler_healthy_task_pool",
                "protocol_version": SCREENING_PROTOCOL_VERSION,
                "expert_outcome_membership_gate": False,
                "accepted_task_ids": ["es-b", "es-a"],
                "artifacts": {
                    "training_pool": {
                        "sha256": sha256_file(envscaler_data),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize_mixed_schedule(
        awm_data=awm_data,
        envscaler_data=envscaler_data,
        envscaler_manifest=health_path,
        output_data=output_data,
        output_manifest=output_manifest,
        train_steps=2,
        awm_per_step=3,
        envscaler_per_step=1,
    )
    rows = pd.read_parquet(output_data).to_dict(orient="records")
    assert manifest["counts_per_step"] == {"awm": 3, "envscaler": 1}
    assert len(rows) == 8
    for step in range(2):
        step_rows = rows[step * 4 : (step + 1) * 4]
        assert [row["extra_info"]["env_family"] for row in step_rows].count("envscaler") == 1

    legacy = json.loads(health_path.read_text())
    legacy["protocol_version"] = 1
    health_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        materialize_mixed_schedule(
            awm_data=awm_data,
            envscaler_data=envscaler_data,
            envscaler_manifest=health_path,
            output_data=tmp_path / "legacy.parquet",
            output_manifest=tmp_path / "legacy.json",
            train_steps=1,
            awm_per_step=1,
            envscaler_per_step=1,
        )

    legacy["protocol_version"] = SCREENING_PROTOCOL_VERSION
    legacy["expert_outcome_membership_gate"] = True
    health_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not depend on expert outcome"):
        materialize_mixed_schedule(
            awm_data=awm_data,
            envscaler_data=envscaler_data,
            envscaler_manifest=health_path,
            output_data=tmp_path / "expert-gated.parquet",
            output_manifest=tmp_path / "expert-gated.json",
            train_steps=1,
            awm_per_step=1,
            envscaler_per_step=1,
        )


def test_mixed_hydra_config_matches_main_protocol():
    config = _compose("awm_envscaler_semantic")
    assert config.env.env_name == "awm_envscaler_semantic"
    assert config.env.context.history_policy == "token_budget"
    assert config.env.context.max_history_exchanges is None
    assert dict(config.env.agentic_mix.trajectory_counts) == {
        "awm": 48,
        "envscaler": 16,
    }
    assert config.env.envscaler.train_max_steps == 40
    assert config.env.envscaler.user_simulator.temperature == 1.0
    assert config.env.envscaler.user_simulator.reasoning_enabled is False
    assert config.env.rollout.n == 4
    rollout = config.actor_rollout_ref.rollout
    assert rollout.n == 1
    assert rollout.temperature == 0.6
    assert rollout.top_p == 0.95
    assert rollout.top_k == 20
    assert rollout.val_kwargs.do_sample is False
    assert rollout.val_kwargs.temperature == 0.0
    assert rollout.val_kwargs.top_p == 1.0
    assert rollout.val_kwargs.top_k == -1
    assert rollout.val_kwargs.min_p == 0.0
