import json

import pytest

from agent_system.environments.env_package.awm.data import feasibility
from agent_system.environments.static_feasibility import (
    STATIC_FEASIBILITY_DECISION_RULES,
)


def test_parser_accepts_json_fence_and_validates_labels():
    parsed = feasibility.DeepSeekFeasibilityClient._parse_content('prefix\n```json\n{"label":"healthy","confidence":91,"rationale":"reachable","evidence":["tool exists"]}\n```')
    assert feasibility.DeepSeekFeasibilityClient._validate(parsed) == {
        "label": "healthy",
        "confidence": 91,
        "rationale": "reachable",
        "evidence": ["tool exists"],
    }
    with pytest.raises(ValueError, match="invalid static feasibility label"):
        feasibility.DeepSeekFeasibilityClient._validate(
            {
                "label": "policy_failure",
                "confidence": 99,
                "rationale": "wrong scope",
                "evidence": [],
            }
        )


def test_judge_uses_shared_generation_contract():
    payloads = []
    client = object.__new__(feasibility.DeepSeekFeasibilityClient)
    client.max_retries = 3
    client.max_tokens = feasibility.JUDGE_MAX_TOKENS
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
                                "confidence": 90,
                                "rationale": "coherent",
                                "evidence": [],
                            }
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 10},
        }
    )

    result = client.judge({"environment": {}, "task": {}})

    assert result["protocol_version"] == feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION
    payload = payloads[0]
    shared_settings = feasibility.static_feasibility_generation_settings(max_tokens=feasibility.JUDGE_MAX_TOKENS)
    assert {key: payload[key] for key in shared_settings} == shared_settings
    assert payload["response_format"] == {"type": "json_object"}
    assert feasibility.JUDGE_INSTRUCTION.endswith(STATIC_FEASIBILITY_DECISION_RULES)


def test_resume_refreshes_infrastructure_and_stale_protocol_records():
    infrastructure = {"status_reason": "judge_infrastructure_exhausted"}
    stale = {"status_reason": "healthy", "judge": {"protocol_version": 0}}
    current = {
        "static_feasibility_protocol_version": (feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION),
        "status_reason": "healthy",
        "judge": {"protocol_version": feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION},
    }
    static_failure = {
        "static_feasibility_protocol_version": feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION,
        "status_reason": "static_evidence_failure",
        "judge": None,
    }
    assert feasibility._is_completed_review(infrastructure) is False
    assert feasibility._is_completed_review(stale) is False
    assert feasibility._is_completed_review(current) is True
    assert feasibility._is_completed_review(static_failure) is True


def test_membership_uses_label_not_confidence():
    class Store:
        @staticmethod
        def build_static_evidence(_row, *, deterministic_record):
            assert deterministic_record["status"] == "healthy"
            return {
                "environment": {
                    "scenario": "s",
                    "native_tools": [{"type": "function", "function": {"name": "act"}}],
                },
                "task": {"task_id": "s:0"},
                "sql_verifier": {"no_action_execution": {"execution_status": "ok"}},
            }

    class Client:
        @staticmethod
        def judge(_evidence):
            return {
                "protocol_version": feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION,
                "label": "healthy",
                "confidence": 1,
                "rationale": "confidence is diagnostic",
                "evidence": [],
                "usage": {},
            }

    result = feasibility.screen_one(
        {"task_id": "s:0", "scenario": "s", "task_idx": 0},
        store=Store(),
        client=Client(),
        deterministic_record={"status": "healthy"},
    )
    assert result["accepted"] is True
    assert result["static_feasibility_protocol_version"] == feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION
    assert result["status"] == "healthy"


def test_judge_infrastructure_failure_is_pending():
    class Store:
        def build_static_evidence(self, _row, *, deterministic_record):
            return {
                "environment": {
                    "scenario": "s",
                    "native_tools": [{"type": "function", "function": {"name": "act"}}],
                },
                "task": {"task_id": "s:0"},
                "sql_verifier": {"no_action_execution": {"execution_status": "ok"}},
            }

    class Client:
        def judge(self, _evidence):
            raise RuntimeError("timeout")

    row = {"task_id": "s:0", "scenario": "s", "task_idx": 0}
    result = feasibility.screen_one(
        row,
        store=Store(),
        client=Client(),
        deterministic_record={"status": "healthy"},
    )

    assert result["accepted"] is None
    assert result["status"] == "pending"
    assert result["status_reason"] == "judge_infrastructure_exhausted"
    assert feasibility._is_completed_review(result) is False
    feasibility._validate_review(result, row=row)


def test_resume_record_is_strict():
    row = {"task_id": "s:0", "scenario": "s", "task_idx": 0}
    valid = {
        **row,
        "static_feasibility_protocol_version": (feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION),
        "accepted": False,
        "status": "quarantine",
        "status_reason": "uncertain",
        "judge": {
            "protocol_version": feasibility.STATIC_FEASIBILITY_PROTOCOL_VERSION,
            "label": "uncertain",
            "confidence": 70,
            "rationale": "insufficient evidence",
            "evidence": [],
        },
    }
    feasibility._validate_review(valid, row=row)
    invalid = dict(valid, accepted=True)
    with pytest.raises(RuntimeError, match="healthy verdict mismatch"):
        feasibility._validate_review(invalid, row=row)

    stale = dict(valid, static_feasibility_protocol_version=0)
    feasibility._validate_review(stale, row=row)
    assert feasibility._is_completed_review(stale) is False


def test_offset_index_defers_duplicate_rejection_to_selected_record(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"scenario": "duplicate", "task_idx": 0},
                {"scenario": "duplicate", "task_idx": 0},
                {"scenario": "healthy", "task_idx": 0},
            )
        )
    )
    offsets = feasibility.StaticEvidenceStore._offsets(source, with_task=True)
    assert len(offsets[("duplicate", 0)]) == 2
    assert len(offsets[("healthy", 0)]) == 1
