import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_system.environments.env_package.awm.data import deterministic_health, health
from agent_system.environments.env_package.awm.runtime.actions import AWMAction
from agent_system.environments.env_package.awm.runtime.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime.manager import (
    AWMEnvironmentManager,
)
from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage


def test_strict_database_check_quarantines_any_seed_insert_failure():
    schema = {
        "db_schema": {
            "tables": [
                {
                    "name": "items",
                    "ddl": "CREATE TABLE items (id INTEGER PRIMARY KEY);",
                    "indexes": ["CREATE INDEX idx_items_id ON items(id);"],
                }
            ]
        }
    }
    healthy = {"sample_data": {"tables": [{"table_name": "items", "insert_statements": ["INSERT INTO items VALUES (1);"]}]}}
    broken = {
        "sample_data": {
            "tables": [
                {
                    "table_name": "items",
                    "insert_statements": [
                        "INSERT INTO items VALUES (1);",
                        "INSERT INTO items VALUES (1);",
                    ],
                }
            ]
        }
    }

    assert health._strict_database_check(schema, healthy) == []
    errors = health._strict_database_check(schema, broken)
    assert len(errors) == 1
    assert errors[0].startswith("seed_insert:items:IntegrityError:")


def test_sql_verifier_audit_requires_exact_task_and_entrypoint():
    row = {"task": "Update item"}
    valid = {
        "scenario": "example",
        "task_idx": 0,
        "task": "Update item",
        "verification": {"code": "def verify_task(initial_db, final_db):\n    return True\n"},
    }
    reasons, digest = health.audit_sql_verifier(row, [valid])
    assert reasons == []
    assert digest

    invalid = dict(valid)
    invalid["task"] = "Different task"
    invalid["verification"] = {"code": "def something_else():\n    return True\n"}
    reasons, _ = health.audit_sql_verifier(row, [invalid])
    assert reasons == [
        "sql_verifier_missing_entrypoint",
        "sql_verifier_task_mismatch",
    ]


def test_noop_judge_errors_retry_then_incomplete_is_healthy(monkeypatch):
    payloads = iter(
        [
            {"reward_type": "judge_error"},
            {"reward_type": "timeout"},
            {"reward_type": "incomplete"},
        ]
    )

    async def fake_noop(*args, **kwargs):
        return next(payloads)

    monkeypatch.setattr(health, "_noop_once", fake_noop)
    result = asyncio.run(
        health.audit_noop(
            {"scenario": "s", "task_idx": 0},
            base_url="unused",
            api_base="unused",
            api_key="unused",
            model="unused",
            semaphore=asyncio.Semaphore(1),
            attempts=3,
        )
    )

    assert result["status"] == "healthy"
    assert result["label"] == "incomplete"
    assert result["attempt"] == 3
    assert len(result["retry_errors"]) == 2


def test_one_off_expert_metadata_is_compact_and_non_gating(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "s:0",
                "status": "failure",
                "legacy_status": "policy_failure",
                "result": {
                    "success": False,
                    "reward_type": "incomplete",
                    "trajectory": [{"model": "deepseek-v4-flash", "large": "ignored"}],
                },
            }
        )
        + "\n"
    )

    assert health.load_expert_metadata(path) == {
        "s:0": {
            "available": True,
            "status": "failure",
            "legacy_status": "policy_failure",
            "success": False,
            "reward_type": "incomplete",
            "model": "deepseek-v4-flash",
        }
    }


def test_outcome_execution_uses_fixed_reward_mapping_not_transport_reward():
    worker_class = AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        max_history_exchanges=6,
        verifier_mode="sql",
        reward_mode="outcome",
    )
    worker._scenario = "scenario"
    worker._task_idx = 0
    worker._task = "Finish the task"
    worker._chat = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": worker._task},
    ]

    async def fake_verify(_final_answer):
        return 99.0, {"reward_type": "incomplete"}, {"status": "normal"}

    worker._verify_and_done = fake_verify
    protocol_reward, done = asyncio.run(
        worker._execute(
            "I could not finish.",
            AWMAction(kind="message", content="I could not finish."),
        )
    )

    assert done is True
    assert protocol_reward == 0.1
    assert worker._last_info["terminal_reward"] == 0.1


def test_terminal_reward_mapping_is_official_and_infrastructure_is_invalid():
    assert AWMWorker._terminal_metadata({"reward_type": "complete"})["terminal_reward"] == 1.0
    assert AWMWorker._terminal_metadata({"reward_type": "incomplete"})["terminal_reward"] == 0.1
    assert AWMWorker._terminal_metadata({"reward_type": "agent_error"})["terminal_reward"] == 0.0
    server = AWMWorker._terminal_metadata({"reward_type": "server_error"})
    assert server["terminal_outcome_valid"] is False
    assert server["outcome_train_mask"] is False
    assert server["terminal_success"] is None
    judge_error = AWMWorker._terminal_metadata(
        {
            "reward_type": "judge_error",
            "verify_result": {"llm_judge": {"error": "bad response"}},
        }
    )
    assert judge_error["terminal_judge_error"] == "bad response"


def test_terminal_metrics_report_coverage_and_both_success_denominators():
    manager = AWMEnvironmentManager(None, None, SimpleNamespace())
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "terminal_label": "complete",
                    "terminal_outcome_valid": True,
                    "terminal_reward": 1.0,
                }
            ],
            [
                {
                    "terminal_label": "server_error",
                    "terminal_outcome_valid": False,
                    "terminal_reward": None,
                }
            ],
            [
                {
                    "terminal_label": "incomplete",
                    "terminal_outcome_valid": True,
                    "terminal_reward": 0.1,
                }
            ],
        ]
    )

    np.testing.assert_allclose(metrics["env/success_rate"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(metrics["env/success_rate_all"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(metrics["env/terminal_judge_coverage"], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(metrics["env/terminal_reward_mean"], [0.55] * 3)
    np.testing.assert_allclose(metrics["env/terminal_server_error_rate"], [0.0, 1.0, 0.0])


def test_outcome_grpo_excludes_invalid_terminal_trajectory():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.1], [0.0, 0.0]])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["task"] * 4, dtype=object),
            "traj_uid": np.asarray(["a", "b", "c", "d"], dtype=object),
            "outcome_train_mask": np.asarray([True, False, True, True]),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.GRPO)

    assert result.non_tensor_batch["outcome_skip_loss"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert result.batch["response_mask"][1].sum().item() == 0
    assert result.batch["advantages"][1].sum().item() == 0
    assert result.meta_info["outcome/terminal_judge_train_coverage"] == 0.75


def test_outcome_grpo_handles_fully_invalid_terminal_group():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.1]])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["task"] * 2, dtype=object),
            "traj_uid": np.asarray(["a", "b"], dtype=object),
            "outcome_train_mask": np.asarray([False, False]),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.GRPO)

    assert result.non_tensor_batch["outcome_skip_loss"].tolist() == [True, True]
    assert result.batch["response_mask"].sum().item() == 0
    assert result.batch["advantages"].sum().item() == 0
    assert result.meta_info["outcome/terminal_judge_train_coverage"] == 0.0


def test_outcome_terminal_judge_coverage_is_trajectory_weighted():
    rewards = torch.zeros((3, 2))
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.ones_like(rewards),
        },
        non_tensors={
            "uid": np.asarray(["task"] * 3, dtype=object),
            "traj_uid": np.asarray(["long", "long", "short"], dtype=object),
            "outcome_train_mask": np.asarray([True, True, False]),
        },
    )

    result = compute_advantage(data, AdvantageEstimator.GRPO)

    assert result.meta_info["outcome/terminal_judge_train_coverage"] == 0.5


def test_deterministic_health_audit_is_hash_bound_and_binary(tmp_path, monkeypatch):
    rows = [
        {"task_id": "s:0", "scenario": "s", "task_idx": 0, "task": "ok"},
        {"task_id": "s:1", "scenario": "s", "task_idx": 1, "task": "bad"},
    ]
    selection = {"selected_counts": {"all_context_eligible": 2}}
    source_hashes = {"source": "digest"}
    candidate_manifest = tmp_path / "candidate_manifest.json"
    data = tmp_path / "candidates.parquet"
    candidate_manifest.write_text("manifest")
    data.write_text("data")
    scenario_records = [{"scenario": "s", "status": "healthy", "status_reasons": [], "database_errors": []}]

    monkeypatch.setattr(deterministic_health, "audit_scenarios", lambda *_: scenario_records)
    monkeypatch.setattr(
        deterministic_health,
        "_load_multimap",
        lambda *_args, **_kwargs: {("s", 0): [{}], ("s", 1): [{}]},
    )
    monkeypatch.setattr(
        deterministic_health,
        "audit_sql_verifier",
        lambda row, _records: ([], "digest") if row["task_idx"] == 0 else (["broken"], None),
    )

    manifest = deterministic_health.build(
        rows=rows,
        selection=selection,
        source_hashes=source_hashes,
        data_dir=tmp_path,
        candidate_manifest=candidate_manifest,
        data=data,
        output_dir=tmp_path / "02_deterministic_audit",
    )

    assert manifest["counts"] == {
        "context_eligible": 2,
        "healthy": 1,
        "quarantine": 1,
        "healthy_environments": 1,
    }
    assert deterministic_health.verify(tmp_path / "02_deterministic_audit")["tasks"] == 1
    audit_path = tmp_path / "02_deterministic_audit" / "task_audit.jsonl"
    audit_path.write_text(audit_path.read_text() + "{}\n")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        deterministic_health.verify(tmp_path / "02_deterministic_audit")


def test_deterministic_health_audit_rebuilds_matching_partial_output(tmp_path, monkeypatch):
    rows = [{"task_id": "s:0", "scenario": "s", "task_idx": 0, "task": "ok"}]
    selection = {"selected_counts": {"all_context_eligible": 1}}
    candidate_manifest = tmp_path / "candidate_manifest.json"
    data = tmp_path / "candidates.parquet"
    candidate_manifest.write_text("manifest")
    data.write_text("data")
    output_dir = tmp_path / "custom_deterministic"
    output_dir.mkdir()
    identity = deterministic_health._identity(
        rows=rows,
        selection=selection,
        source_hashes={"source": "digest"},
        candidate_manifest=candidate_manifest,
        data=data,
    )
    (output_dir / "config.json").write_text(json.dumps(identity))
    (output_dir / "task_audit.jsonl").write_text("torn partial output")
    monkeypatch.setattr(
        deterministic_health,
        "audit_scenarios",
        lambda *_: [{"scenario": "s", "status": "healthy", "status_reasons": [], "database_errors": []}],
    )
    monkeypatch.setattr(
        deterministic_health,
        "_load_multimap",
        lambda *_args, **_kwargs: {("s", 0): [{}]},
    )
    monkeypatch.setattr(deterministic_health, "audit_sql_verifier", lambda *_: ([], "digest"))

    manifest = deterministic_health.build(
        rows=rows,
        selection=selection,
        source_hashes={"source": "digest"},
        data_dir=tmp_path,
        candidate_manifest=candidate_manifest,
        data=data,
        output_dir=output_dir,
    )

    assert manifest["counts"]["healthy"] == 1
    assert deterministic_health.verify(output_dir)["tasks"] == 1


def test_health_pool_fresh_output_allows_launcher_logs_only(tmp_path):
    output_dir = tmp_path / "03_code_augmented_screening"
    output_dir.mkdir()
    (output_dir / "server.log").write_text("server startup")
    config_path = health._initialize_output_config(output_dir, {"protocol_version": 2})

    assert json.loads(config_path.read_text()) == {"protocol_version": 2}

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "unexpected.json").write_text("{}")
    with pytest.raises(FileExistsError, match="unexpected.json"):
        health._initialize_output_config(blocked, {"protocol_version": 2})


def test_task_health_validator_rejects_inconsistent_noop_evidence():
    deterministic = {
        "status_reasons": [],
        "sql_verifier_sha256": "digest",
    }
    valid = {
        "status": "healthy",
        "status_reasons": [],
        "sql_verifier_sha256": "digest",
        "noop": {
            "status": "healthy",
            "label": "incomplete",
            "status_reason": None,
        },
    }
    health._validate_task_record_against_deterministic(valid, deterministic)

    invalid = dict(valid)
    invalid["noop"] = dict(valid["noop"], label="complete")
    with pytest.raises(RuntimeError, match="healthy no-action evidence"):
        health._validate_task_record_against_deterministic(invalid, deterministic)
