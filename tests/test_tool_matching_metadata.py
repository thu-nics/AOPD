"""Equivalence contracts plus isolated counterfactuals using native tool code."""

import ast
import asyncio
import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_system.environments.action_matching import build_tool_match_plan, finish_tool_match_plan, match_candidate_tools, tool_pair_fingerprint
from agent_system.environments.env_package.awm.runtime.actions import AWMAction
from agent_system.environments.tool_matching_metadata import callable_tool_matching_metadata, comparison_arguments, source_tool_matching_metadata


def action(name, **arguments):
    return AWMAction(kind="tool", name=name, arguments=arguments)


def test_missing_evidence_is_failure_only_for_unresolved_pair():
    a, b = action("write", note="A"), action("write", note="B")
    assert asyncio.run(match_candidate_tools(None, [a] * 3, [a], [], []))["counts"] == {0: 3}
    with pytest.raises(ValueError, match="source unavailable"):
        build_tool_match_plan([a], [b], [{"name": "write", "inputSchema": {}}], [])
    plan = build_tool_match_plan([a], [b], [], [], semantic_enabled=False)
    assert finish_tool_match_plan(plan, [])["counts"] == {0: 0}


def test_source_evidence_excludes_initialization_and_checker_and_has_dependencies():
    source = """
class Params(BaseModel):
    status: str
def helper(value):
    return value.lower()
class Env:
    def __init__(self):
        self.secret = "HIDDEN_TASK_ANSWER"
    def check_task_complete(self):
        return "VERIFIER_TARGET"
    def update(self, body: Params):
        return helper(body.status)
"""
    tools = [{"name": "update", "inputSchema": {}}]
    metadata = source_tool_matching_metadata(source, tools, family="test", environment="test", class_name="Env")
    evidence = metadata["update"]["source"]
    assert "class Params" in evidence and "def helper" in evidence and "def update" in evidence
    assert "HIDDEN_TASK_ANSWER" not in evidence and "VERIFIER_TARGET" not in evidence
    bad = source.replace("return helper(body.status)", "return self.check_task_complete()")
    assert "error" in source_tool_matching_metadata(bad, tools, family="test", environment="test", class_name="Env")["update"]


def test_changed_source_schema_context_and_rules_change_pair_fingerprint():
    def fingerprint(evidence):
        return tool_pair_fingerprint(provider="test", model="model", endpoint="url", decoding_config={}, evidence=evidence)

    tools = [{"name": "write", "inputSchema": {"type": "object"}}]
    metadata = source_tool_matching_metadata("def write(note):\n return note", tools, family="test", environment="test")
    evidence = build_tool_match_plan([action("write", note="A")], [action("write", note="B")], tools, [], tool_matching_metadata=metadata)["pairs"][0]
    for key in ("source_hash", "rules_version", "environment"):
        changed = copy.deepcopy(evidence)
        changed["tool_matching_metadata"][key] = "changed"
        assert fingerprint(evidence) != fingerprint(changed)
    for key in ("tool", "public_context"):
        assert fingerprint(evidence) != fingerprint({**evidence, key: "changed"})


def test_operation_id_wins_over_private_helper_and_duplicate_routes_fail():
    source = "def fetch():\n return 1\n@app.get('/me', operation_id='fetch')\ndef endpoint():\n return fetch()\n"
    tools = [{"name": "fetch"}]
    entry = source_tool_matching_metadata(source, tools, family="awm", environment="test")["fetch"]
    assert "error" not in entry
    assert "def endpoint" in entry["source"] and "def fetch" in entry["source"]
    ambiguous = source + "\n@app.get('/other', operation_id='fetch')\ndef other():\n return 2\n"
    assert "error" in source_tool_matching_metadata(ambiguous, tools, family="awm", environment="test")["fetch"]


def test_overflow_is_not_silently_truncated(monkeypatch):
    import agent_system.environments.tool_matching_metadata as metadata_module

    monkeypatch.setattr(metadata_module, "MAX_SOURCE_CHARS", 10)
    entry = source_tool_matching_metadata("def write(note):\n return note", [{"name": "write"}], family="test", environment="test")["write"]
    assert "budget" in entry["error"] and "source" not in entry


def test_local_cls_dictionary_is_not_a_missing_class_method():
    source = "class Env:\n def read(self, id):\n  cls = self.classes[id]\n  return cls.get('name')\n"
    assert "error" not in source_tool_matching_metadata(source, [{"name": "read"}], family="test", environment="test", class_name="Env")["read"]


@pytest.fixture(scope="module")
def retail():
    pytest.importorskip("tau2")
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    if not Path(RETAIL_DB_PATH).is_file():
        pytest.skip("native Tau retail DB unavailable")
    db = RetailDB.load(RETAIL_DB_PATH)
    native = RetailTools(db.model_copy(deep=True)).get_tools()
    metadata = callable_tool_matching_metadata(list(native.values()), family="tau", environment="retail")
    return db, RetailTools, native, metadata


@pytest.mark.parametrize("exchange", [False, True])
def test_native_tau_reordering_changes_neither_db_nor_observation(retail, exchange):
    db, Tools, native, metadata = retail
    name = "exchange_delivered_order_items" if exchange else "return_delivered_order_items"
    args = dict(order_id="#W4817420", item_ids=["6777246137", "4900661478"], payment_method_id="gift_card_8168843")
    if exchange:
        args["new_item_ids"] = ["5758737025", "8479046075"]
    permuted = copy.deepcopy(args)
    permuted["item_ids"].reverse()
    if exchange:
        permuted["new_item_ids"].reverse()
    left, right = Tools(db.model_copy(deep=True)), Tools(db.model_copy(deep=True))
    assert getattr(left, name)(**args).model_dump() == getattr(right, name)(**permuted).model_dump()
    assert left.db.model_dump() == right.db.model_dump()
    a, b = action(name, **args), action(name, **permuted)
    plan = build_tool_match_plan([a, a, a], [b], [native[name].openai_schema], [], tool_matching_metadata=metadata)
    assert not plan["pairs"]
    result = finish_tool_match_plan(plan, [])
    assert result["counts"] == {0: 3} and result["normalized_counts"] == [3]
    assert b.arguments == permuted
    wrong = copy.deepcopy(args)
    if exchange:
        wrong["new_item_ids"].reverse()  # Change mapping, not pair order.
    else:
        wrong["item_ids"].append(wrong["item_ids"][0])  # Multiplicity matters.
    negative = build_tool_match_plan([a], [action(name, **wrong)], [native[name].openai_schema], [], tool_matching_metadata=metadata)
    assert negative["matrix"] == [[False]] and len(negative["pairs"]) == 1


def test_native_tau_runtime_literal_constraints_are_in_evidence(retail):
    _, _, native, metadata = retail
    name = "cancel_pending_order"
    a = action(name, order_id="#W0000000", reason="no longer needed")
    b = action(name, order_id="#W0000000", reason="not needed anymore")
    plan = build_tool_match_plan([a], [b], [native[name].openai_schema], [], tool_matching_metadata=metadata)
    evidence = plan["pairs"][0]
    assert "ordered by mistake" in evidence["tool_matching_metadata"]["source"]
    assert finish_tool_match_plan(plan, [False])["counts"] == {0: 0}


@pytest.fixture(scope="module")
def envscaler():
    from agent_system.environments.env_package.envscaler.source import DEFAULT_SOURCE_ROOT, load_envscaler_source

    if not DEFAULT_SOURCE_ROOT.is_dir():
        pytest.skip("native EnvScaler checkout unavailable")
    return load_envscaler_source()


def _native_function(source, name, namespace):
    # Tests only: remove decorators/annotations, retain the actual function body.
    node = copy.deepcopy(next(n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    node.args.defaults = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<native-test-tool>", "exec"), namespace)
    return namespace[name]


def test_native_envscaler_lowercase_equivalence_is_source_scoped(envscaler):
    env = envscaler.environments["env_142_rl"]
    name = "update_appointment_status"
    function = _native_function(env["env_class_code"], name, {})
    a = SimpleNamespace(appointments={"a": {"appointment_status": "scheduled"}})
    b = copy.deepcopy(a)
    assert function(a, "a", "completed") == function(b, "a", "COMPLETED")
    assert vars(a) == vars(b)
    metadata = source_tool_matching_metadata(env["env_class_code"], env["tools"], family="envscaler", environment="env_142_rl", class_name=env["env_class_name"])
    entry = metadata[name]
    args = {"appointment_id": "a", "new_status": "COMPLETED"}
    assert comparison_arguments(args, entry)["new_status"] == "completed"
    assert args["new_status"] == "COMPLETED"
    changed = source_tool_matching_metadata(env["env_class_code"].replace(".lower()", ".strip()"), env["tools"], family="envscaler", environment="env_142_rl", class_name=env["env_class_name"])
    assert not changed[name]["rules"]
    assert comparison_arguments(args, changed[name]) == args
    changed_helper = source_tool_matching_metadata(env["env_class_code"] + "\n# changed module\n", env["tools"], family="envscaler", environment="env_142_rl", class_name=env["env_class_name"])
    assert not changed_helper[name]["rules"]


def test_native_envscaler_ingredient_set_rule(envscaler):
    env = envscaler.environments["env_160_rl"]
    name = "update_allowed_ingredients_for_user"
    function = _native_function(env["env_class_code"], name, {})
    initial = SimpleNamespace(user_profiles={"u": {"allowed_ingredients": ["c"], "allergies": [], "disallowed_ingredients": []}}, ingredients={"a": {}, "b": {}, "c": {}})
    left, right = copy.deepcopy(initial), copy.deepcopy(initial)
    assert function(left, "u", ["a", "b", "a"], ["c", "c"]) == function(right, "u", ["b", "a"], ["c"])
    assert vars(left) == vars(right)
    assert function(copy.deepcopy(initial), "u", None, []) == function(copy.deepcopy(initial), "u", [], None)
    entry = source_tool_matching_metadata(env["env_class_code"], env["tools"], family="envscaler", environment="env_160_rl", class_name=env["env_class_name"])[name]
    assert comparison_arguments({"user_id": "u", "add_ingredient_ids": ["a", "b", "a"], "remove_ingredient_ids": ["c", "c"]}, entry) == comparison_arguments({"user_id": "u", "add_ingredient_ids": ["b", "a"], "remove_ingredient_ids": ["c"]}, entry)


def test_native_awm_set_equivalence_with_isolated_session():
    path = Path(__file__).resolve().parents[2] / "openenv-awm-cache" / "gen_envs.jsonl"
    if not path.is_file():
        pytest.skip("native AWM source cache unavailable")
    with path.open() as stream:
        env = next(r for r in map(json.loads, stream) if r["scenario"] == "social_media_4")
    now = datetime(2026, 1, 1)

    def execute(add, remove):
        prefs = SimpleNamespace(user_id=1, hide_subreddit_ids="1,3", nsfw_blur_enabled=1, updated_at=now)
        session = SimpleNamespace()
        session.query = session.filter = lambda *a: session
        session.first = lambda: prefs
        session.commit = session.close = lambda: None
        namespace = {"SessionLocal": lambda: session, "current_user_id": lambda: 1, "now_utc": lambda: now, "UserContentPreference": SimpleNamespace(user_id=1), "PatchHiddenSubredditsResponse": lambda **kw: kw}
        function = _native_function(env["full_code"], "patch_hidden_subreddits", namespace)
        return asyncio.run(function(SimpleNamespace(add_subreddit_ids=add, remove_subreddit_ids=remove))), vars(prefs)

    assert execute([2, 4, 2], [3, 3]) == execute([4, 2], [3])
    assert execute(None, []) == execute([], None)
    tools = [{"name": "patch_hidden_subreddits", "inputSchema": {}}]
    metadata = source_tool_matching_metadata(env["full_code"], tools, family="awm", environment="social_media_4")
    entry = metadata[tools[0]["name"]]
    assert entry["rules"] and "PatchHiddenSubredditsRequest" in entry["source"]
    assert comparison_arguments({"add_subreddit_ids": [2, 4, 2], "remove_subreddit_ids": [3, 3]}, entry) == comparison_arguments({"add_subreddit_ids": [4, 2], "remove_subreddit_ids": [3]}, entry)


def test_normalization_count_ignores_padding():
    from agent_system.environments.teacher_reward import teacher_selection_diagnostics

    rows = [{"tool_argument_normalized_match_count": 3}, {"tool_argument_normalized_match_count": 3, "is_padding": True}]
    assert teacher_selection_diagnostics([rows], [])["tool_argument_normalized_match_count"] == 3
