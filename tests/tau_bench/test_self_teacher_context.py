"""Contracts for reviewed customer briefs and live, read-only private state."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.tau_bench import self_teacher_context as context
from agent_system.environments.env_package.tau_bench.customer_briefs import BRIEF_PROTOCOL, BriefStore, digest
from agent_system.environments.env_package.tau_bench.self_teacher import build_self_teacher_messages
from agent_system.environments.env_package.tau_bench.self_teacher_privilege import bind_customer, privileged_context, redact_private

SCENARIO = {"instructions": {"known_info": "You are Mia Garcia with email mia@example.com.", "reason_for_call": "Keep the keyboard; refund everything else only if PayPal is possible."}}
TASK = {"id": "test", "user_scenario": SCENARIO, "evaluation_criteria": {"actions": [{"name": "BAD_TOOL"}]}, "initial_state": {"hidden": "FORBIDDEN"}}


def brief_file(tmp_path):
    path = tmp_path / "briefs.json"
    facts = [
        {"category": "known", "text": "The customer can provide an email address: mia@example.com.", "source_field": "instructions.known_info", "quote": "email mia@example.com"},
        {"category": "constraint", "text": "Keep the keyboard; refund everything else only if PayPal is possible.", "source_field": "instructions.reason_for_call", "quote": SCENARIO["instructions"]["reason_for_call"]},
    ]
    path.write_text(json.dumps({"protocol": BRIEF_PROTOCOL, "status": "frozen", "records": [{"domain": "retail", "task_id": "test", "user_scenario": SCENARIO, "scenario_sha256": digest(SCENARIO), "facts": facts, "review_status": "approved"}]}))
    return str(path)


def test_frozen_briefs_and_source_integrity(tmp_path):
    path = brief_file(tmp_path)
    assert BriefStore(path).get("retail", TASK)
    with pytest.raises(ValueError, match="source drift"):
        BriefStore(path).get("retail", dict(TASK, user_scenario={}))
    value = json.loads(Path(path).read_text())
    value["status"] = "draft"
    Path(path).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reviewed/frozen"):
        BriefStore(path)


def test_customer_preserves_availability_and_constraints_not_answers(tmp_path):
    original = copy.deepcopy(TASK)
    result = privileged_context("retail", TASK, [], briefs_path=brief_file(tmp_path), mode="customer")
    text = json.dumps(result)
    assert "can provide an email address" in text and "mia@example.com" not in text
    assert "only if PayPal" in text and "keyboard" in text
    assert "BAD_TOOL" not in text and "FORBIDDEN" not in text
    assert TASK == original


@pytest.mark.parametrize("role", ["system", "assistant", "user", "tool"])
def test_identifiers_must_be_publicly_grounded(role):
    value = {"known": "mia@example.com", "amount": 125000, "constraint": "Budget $10000; 2.0 GB."}
    result = redact_private(value, SCENARIO, [{"role": role, "content": "mia@example.com"}])
    assert (result["known"] == "mia@example.com") == (role in {"user", "tool"})
    assert result["amount"] == 125000
    assert "$10000" in result["constraint"] and "2.0 GB" in result["constraint"]


def test_alias_stable_when_other_identifiers_appear():
    one = redact_private({"id": "W2378156"}, SCENARIO, [])
    two = redact_private({"id": "W2378156", "other": "W2378157"}, SCENARIO, [])
    assert one["id"] == two["id"]
    assert redact_private({"id": "W2378156"}, SCENARIO, [{"role": "assistant", "tool_calls": [{"arguments": "W2378156"}]}]) == one


def environment():
    user = {"user_id": "mia_garcia_1", "name": {"first_name": "Mia", "last_name": "Garcia"}, "email": "mia@example.com", "payment_methods": {}}
    db = {
        "users": {"mia_garcia_1": user},
        "orders": {
            "W2378156": {"order_id": "W2378156", "user_id": "mia_garcia_1", "status": "pending", "items": [{"item_id": "1234567890", "name": "Keyboard", "price": 125000, "options": {"color": "red"}}], "address": {"city": "Boston"}, "payment_history": []},
            "OTHER": {"order_id": "OTHER", "user_id": "someone_else", "status": "pending"},
        },
    }
    return SimpleNamespace(tools=SimpleNamespace(db=db), user_tools=None, sync_tools=lambda: pytest.fail("projection must not sync or call tools"))


def test_live_projection_read_only_updated_and_customer_scoped(tmp_path):
    env = environment()
    before = copy.deepcopy(env.tools.db)
    path = brief_file(tmp_path)
    p1 = privileged_context("retail", TASK, [], briefs_path=path, mode="customer_and_state", environment=env)
    assert env.tools.db == before and "OTHER" not in json.dumps(p1)
    assert p1["current_state"]["orders"][0]["status"] == "pending"
    assert p1["current_state"]["orders"][0]["items"][0]["price"] == 125000
    env.tools.db["orders"]["W2378156"]["status"] = "cancelled"
    p2 = privileged_context("retail", TASK, [], briefs_path=path, mode="customer_and_state", environment=env)
    assert p2["current_state"]["orders"][0]["status"] == "cancelled" and p1 != p2


def test_ambiguous_binding_fails_not_fallback_to_task_id():
    env = environment()
    with pytest.raises(ValueError, match="uniquely bind"):
        bind_customer("retail", {"id": "mia_garcia_1", "evaluation_criteria": {"user_id": "mia_garcia_1"}}, env.tools.db)


def test_missing_live_env_is_not_silently_static(tmp_path):
    with pytest.raises(ValueError, match="live environment"):
        privileged_context("retail", TASK, [], briefs_path=brief_file(tmp_path), mode="customer_and_state")


def test_worker_uses_same_shared_projection_and_preserves_history(tmp_path):
    from test_protocol import _tau_preflight_worker

    worker = _tau_preflight_worker(lambda **kw: pytest.fail("no API expected"))
    worker.teacher_source = "self"
    worker.use_privileged_teacher_context = True
    worker.self_privilege_mode = "customer"
    worker.self_customer_briefs_path = brief_file(tmp_path)
    worker.domain = "retail"
    worker._task = lambda: TASK
    chat = [{"role": "system", "content": "Policy"}, {"role": "user", "content": "Help"}]
    worker._student_chat = lambda: chat
    r = worker.describe_self_teacher_state(chat)
    assert r["messages"][1:] == chat[1:] == r["public_chat"][1:]
    assert "can provide an email" in r["messages"][0]["content"]
    assert "mia@example.com" not in r["messages"][0]["content"]


def test_other_domain_public_policy_unchanged():
    for text in ("Airline policy", "Retail policy", ""):
        assert context.teacher_public_policy(text) == text


@pytest.mark.parametrize("text", ["<tech_support_policy>drift</tech_support_policy>", "<tech_support_policy>broken", "</tech_support_policy>", "<tech_support_policy>x</tech_support_policy><tech_support_policy>y</tech_support_policy>"])
def test_changed_telecom_manual_fails_closed(text):
    with pytest.raises(ValueError):
        context.teacher_public_policy(text)


def test_teacher_policy_projection_preserves_main_policy_and_actual_history(monkeypatch):
    manual = "Device API check_status_bar() and reset_apn_settings()."
    monkeypatch.setattr(context, "NATIVE_MANUAL_SHA256", hashlib.sha256(manual.encode()).hexdigest())
    context.teacher_public_policy.cache_clear()
    policy = "MAIN POLICY: authenticate first.\n<tech_support_policy>\n" + manual + "\n</tech_support_policy>\nTAIL POLICY"
    chat = [
        {"role": "system", "content": policy},
        {"role": "user", "content": "Help me."},
        {"role": "assistant", "content": "I incorrectly mentioned reset_apn_settings()."},
        {"role": "tool", "content": "Real output", "tool_call_id": "call-1"},
    ]
    original = copy.deepcopy(chat)
    s1 = build_self_teacher_messages(chat)
    s2 = build_self_teacher_messages(chat, {"customer": {"goal": ["Restore mobile data."]}})
    assert chat == original
    assert s1[1:] == s2[1:] == chat[1:]
    assert s2[0]["content"].startswith(s1[0]["content"])
    assert s1[0]["content"].startswith("MAIN POLICY: authenticate first.\n")
    assert "</tech_support_policy>\nTAIL POLICY" in s1[0]["content"]
    assert "check_status_bar" not in s1[0]["content"] and "reset_apn_settings" not in s2[0]["content"]
    context.teacher_public_policy.cache_clear()


def test_pinned_real_manual_removes_all_user_api_names():
    source = Path(os.environ.get("TAU2_ROOT", Path(__file__).resolve().parents[3] / "tau2-bench"))
    path = source / "data/tau2/domains/telecom/tech_support_manual.md"
    if not path.exists():
        pytest.skip("optional pinned Tau source is not installed")
    manual = path.read_text()
    main = (path.parent / "main_policy.md").read_text()
    result = context.teacher_public_policy("<main_policy>\n" + main + "\n</main_policy><tech_support_policy>\n" + manual + "\n</tech_support_policy>")
    names = set(re.findall(r"\*\*([a-z]+_[a-z_]+)\*\*", manual))
    assert len(names) == 28
    assert not any(name in result for name in names)
    assert "check_payment_request" not in result and "make_payment" not in result
    assert "Always check that the bill status is updated to PAID" in result
    assert "ask them to complete the payment" in result
    for fact in ("PIN/PUK", "restart", "account-side roaming", "Data Saver", "storage", "SMS", "2G", "3G", "4G", "5G", "per-GB", "main policy"):
        assert fact in result


def test_real_telecom_snapshot_is_live_read_only_and_excludes_credentials():
    source = Path(os.environ.get("TAU2_ROOT", Path(__file__).resolve().parents[3] / "tau2-bench"))
    if not (source / "data/tau2/domains/telecom/tasks.json").exists():
        pytest.skip("optional pinned Tau source is not installed")
    from tau2.runner.build import build_environment
    from tau2.runner.helpers import load_tasks

    from agent_system.environments.env_package.tau_bench.self_teacher_privilege import live_customer_state

    task = load_tasks("telecom", "train")[0]
    env = build_environment("telecom")
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data, initialization_actions=initial.initialization_actions, message_history=initial.message_history or [])
    scenario = task.user_scenario.model_dump(mode="json")
    before = (env.get_db_hash(), env.get_user_db_hash())
    first, _ = live_customer_state("telecom", scenario, env)
    assert before == (env.get_db_hash(), env.get_user_db_hash())
    assert "pin" not in json.dumps(first).lower() and "puk" not in json.dumps(first).lower()
    assert "toggle_data" not in json.dumps(first) and "reset_apn_settings" not in json.dumps(first)
    env.user_tools.db.device.airplane_mode = not env.user_tools.db.device.airplane_mode
    second, _ = live_customer_state("telecom", scenario, env)
    assert second["device"]["airplane_mode"] != first["device"]["airplane_mode"]


def test_brief_digest_is_part_of_self_teacher_resume_identity(tmp_path):
    from test_self_teacher import Tokenizer, config

    from agent_system.environments.env_package.tau_bench.customer_briefs import load_briefs
    from agent_system.environments.env_package.tau_bench.self_teacher import SelfTeacherRollout

    cfg = config(tmp_path)
    SelfTeacherRollout(cfg, Tokenizer())
    path = Path(cfg.env.tau.oracle.self_customer_briefs_path)
    bundle = json.loads(path.read_text())
    bundle["records"][0]["facts"][0]["text"] = "Revised customer knowledge"
    path.write_text(json.dumps(bundle))
    load_briefs.cache_clear()
    with pytest.raises(ValueError, match="resume protocol changed"):
        SelfTeacherRollout(cfg, Tokenizer())
