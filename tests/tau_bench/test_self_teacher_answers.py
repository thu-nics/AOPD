"""Reference projection contracts; no model requests or environment writes."""

import copy
import json
import os
from pathlib import Path

import pytest

from agent_system.environments.env_package.tau_bench.self_teacher import SelfTeacherRollout, build_self_teacher_messages
from agent_system.environments.env_package.tau_bench.self_teacher_answers import USER_ACTIONS, customer_step, reference_guidance, reference_source_fingerprint, target_state
from agent_system.environments.env_package.tau_bench.self_teacher_privilege import privileged_context

TOOLS = [{"type": "function", "function": {"name": "cancel_order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}}]


def task():
    from test_self_teacher_context import TASK

    value = copy.deepcopy(TASK)
    value["evaluation_criteria"] = {
        "actions": [{"name": "cancel_order", "arguments": {"order_id": "W2378156"}, "requestor": "assistant", "compare_args": ["order_id"], "info": "EVALUATOR_INTERNAL"}],
        "nl_assertions": ["Do not promise an unconfirmed refund."],
        "reward_basis": ["DB"],
    }
    return value


def test_answer_parameters_preserved_without_evaluator_or_db_fields(tmp_path):
    from test_self_teacher_context import brief_file

    value = task()
    before = copy.deepcopy(value)
    result = privileged_context("retail", value, [], briefs_path=brief_file(tmp_path), mode="answer_conditioned", tools=TOOLS)
    assert result["reference"]["reference_steps"][0]["arguments"] == {"order_id": "W2378156"}
    assert "mia@example.com" in json.dumps(result)
    assert result["reference"]["available"]
    assert not any(s in json.dumps(result) for s in ["compare_args", "EVALUATOR_INTERNAL", "reward_basis", "FORBIDDEN"])
    result["reference"]["reference_steps"][0]["arguments"]["order_id"] = "CHANGED"
    assert value == before


def test_customer_and_assertion_apis_become_plain_language():
    criteria = {
        "actions": [{"requestor": "user", "name": name, "arguments": {}} for name in USER_ACTIONS],
        "env_assertions": [{"env_type": "user", "func_name": "assert_mobile_data_status", "arguments": {"expected_status": True}, "assert_value": True}],
    }
    result = reference_guidance({"evaluation_criteria": criteria}, [])
    text = json.dumps(result)
    assert all(name not in text for name in USER_ACTIONS)
    assert "assert_mobile_data_status" not in text
    assert all(s["actor"] == "customer" and "tool" not in s for s in result["reference_steps"])
    assert "Mobile data should work." in text
    assert "4G/5G" in customer_step("set_network_mode_preference", {"mode": "4g_5g_preferred"})
    assert "SMS" in customer_step("grant_app_permission", {"app_name": "messaging", "permission": "sms"})
    assert "check" in customer_step("toggle_roaming", {})  # do not invent a toggle direction


@pytest.mark.parametrize("criteria,available", [({}, False), ({"actions": []}, False), ({"communicate_info": ["cotton"]}, True), ({"nl_assertions": ["Do not cancel."]}, True)])
def test_no_reference_actions_never_implies_noop_or_success(criteria, available):
    result = reference_guidance({"evaluation_criteria": criteria}, [])
    assert result["available"] == available
    assert result["reference_steps"] == []
    assert "success" not in result


@pytest.mark.parametrize("action", [{"requestor": "user", "name": "UNKNOWN", "arguments": {}}, {"requestor": "assistant", "name": "UNKNOWN", "arguments": {}}, {"requestor": "user", "name": "toggle_roaming", "arguments": {"target": True}}])
def test_unmapped_actions_fail_loudly(action):
    with pytest.raises(ValueError):
        reference_guidance({"evaluation_criteria": {"actions": [action]}}, TOOLS)


def test_exact_target_semantics_and_unknown_assertions():
    result = target_state({"env_type": "assistant", "func_name": "assert_data_refueling_amount", "arguments": {"customer_id": "C1001", "line_id": "L1002", "expected_amount": 2.0}, "assert_value": True})
    assert "total" in result and "not an additional amount" in result
    result = target_state({"env_type": "user", "func_name": "assert_service_status", "arguments": {"expected_status": "no_service"}, "assert_value": True})
    assert "remain unavailable" in result
    with pytest.raises(ValueError, match="unmapped"):
        target_state({"env_type": "user", "func_name": "UNKNOWN", "arguments": {}, "assert_value": True})
    with pytest.raises(ValueError, match="unmapped"):
        target_state({"env_type": "user", "func_name": "assert_mobile_data_status", "arguments": {"expected_status": True}, "assert_value": False})


def test_answer_worker_preserves_student_history_and_uses_shared_projection(tmp_path):
    from test_protocol import _tau_preflight_worker
    from test_self_teacher_context import brief_file

    worker = _tau_preflight_worker(lambda **kw: pytest.fail("must not call API"))
    worker.teacher_source = "self"
    worker.use_privileged_teacher_context = True
    worker.self_privilege_mode = "answer_conditioned"
    worker.self_customer_briefs_path = brief_file(tmp_path)
    worker.domain = "retail"
    worker._task = task
    worker._tools = lambda: TOOLS
    chat = [{"role": "system", "content": "Policy"}, {"role": "user", "content": "May I cancel?"}]
    worker._student_chat = lambda: chat
    original = copy.deepcopy(chat)
    request = worker.describe_self_teacher_state(chat)
    assert request["messages"][1:] == request["public_chat"][1:] == chat[1:]
    assert "W2378156" in request["messages"][0]["content"]
    assert "W2378156" not in json.dumps(request["public_chat"])
    assert chat == original and request["tools"] == TOOLS
    public = build_self_teacher_messages(chat)
    assert "W2378156" not in json.dumps(public)
    assert "required confirmation" in public[0]["content"]


def test_reference_fingerprint_and_resume_identity(tmp_path):
    from test_self_teacher import Tokenizer, config

    cfg = config(tmp_path)
    cfg.env.tau.source_root = str(tmp_path / "tau")
    cfg.env.tau.oracle.self_privilege_mode = "answer_conditioned"
    files = []
    for domain in ("airline", "retail", "telecom"):
        root = tmp_path / "tau/data/tau2/domains" / domain
        root.mkdir(parents=True)
        (root / "split_tasks.json").write_text(json.dumps({"train": ["train"]}))
        f = root / "tasks.json"
        f.write_text(json.dumps([{"id": "train", "evaluation_criteria": {"communicate_info": ["old"]}}, {"id": "test", "evaluation_criteria": {"communicate_info": ["secret"]}}]))
        files.append(f)
    first = reference_source_fingerprint(cfg.env.tau.source_root)
    SelfTeacherRollout(cfg, Tokenizer())
    values = json.loads(files[0].read_text())
    values[1]["evaluation_criteria"]["communicate_info"] = ["test changed"]
    files[0].write_text(json.dumps(values))
    assert reference_source_fingerprint(cfg.env.tau.source_root) == first
    values[0]["evaluation_criteria"]["communicate_info"] = ["train changed"]
    files[0].write_text(json.dumps(values))
    assert reference_source_fingerprint(cfg.env.tau.source_root) != first
    with pytest.raises(ValueError, match="resume protocol changed"):
        SelfTeacherRollout(cfg, Tokenizer())


def test_native_train_reference_coverage_read_only():
    source = Path(os.environ.get("TAU2_ROOT", Path(__file__).resolve().parents[3] / "tau2-bench"))
    if not (source / "data/tau2/domains/telecom/tasks.json").exists():
        pytest.skip("optional pinned Tau source is not installed")
    from tau2.runner.build import build_environment
    from tau2.runner.helpers import load_tasks

    missing = []
    for domain, count in [("airline", 30), ("retail", 74), ("telecom", 74)]:
        env = build_environment(domain)
        before = (env.get_db_hash(), env.get_user_db_hash())
        tools = [t.openai_schema for t in env.get_tools()]
        tasks = load_tasks(domain, "train")
        assert len(tasks) == count
        for native in tasks:
            result = reference_guidance(native.model_dump(mode="json"), tools)
            if not result["available"]:
                missing.append((domain, str(native.id)))
            text = json.dumps(result)
            if domain == "telecom":
                assert all(name not in text for name in USER_ACTIONS)
                assert "assert_" not in text
        assert before == (env.get_db_hash(), env.get_user_db_hash())
    assert missing == [("retail", "57")]
