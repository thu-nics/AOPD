import asyncio
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent_system.environments.action_matching import build_tool_match_plan as _build_tool_match_plan
from agent_system.environments.action_matching import finish_tool_match_plan, match_candidate_tools
from agent_system.environments.env_package.awm.runtime.actions import AWMAction, canonical_action, score_candidates, state_fingerprint, validate_action
from agent_system.environments.env_package.awm.runtime.envs import AWMWorker
from agent_system.environments.env_package.awm.runtime.oracle import DeepSeekAWMOracleClient
from agent_system.environments.env_package.envscaler.runtime import EnvScalerWorker
from agent_system.environments.tool_matching_metadata import source_tool_matching_metadata

TOOLS = [
    {
        "name": "log_interaction",
        "description": "Store a support interaction note.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["id", "note"],
            "additionalProperties": False,
        },
    }
]
CHAT = [{"role": "system", "content": "Use the available tools."}, {"role": "user", "content": "Record the customer's payment."}]
METADATA = source_tool_matching_metadata("def log_interaction(id, note):\n    return {'id': id, 'note': note}\n", TOOLS, family="test", environment="test")


def build_tool_match_plan(*args, **kwargs):
    kwargs.setdefault("tool_matching_metadata", METADATA)
    return _build_tool_match_plan(*args, **kwargs)


def action(note="Payment completed.", **kwargs):
    return AWMAction(kind="tool", name="log_interaction", arguments={"id": 1, "note": note, **kwargs})


def response(value):
    return {"model": "deepseek-v4-flash", "system_fingerprint": "test", "choices": [{"message": {"content": json.dumps({"equivalent": value})}}]}


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_duplicate_votes_are_not_collapsed_and_identity_is_unchanged():
    teacher = action()
    candidate = action("The payment was made.")
    original = canonical_action(candidate)
    calls = []

    def request(payload):
        calls.append(payload)
        evidence = json.loads(payload["messages"][1]["content"])
        return response(evidence["teacher_arguments"]["id"] == evidence["candidate_arguments"]["id"])

    client = DeepSeekAWMOracleClient(request_fn=request)
    plan = build_tool_match_plan([teacher, teacher, candidate], [candidate, action(id=2), AWMAction(kind="invalid")], TOOLS, CHAT)
    result = finish_tool_match_plan(plan, client.match_tool_argument_pairs(plan["pairs"]))
    assert result["matrix"] == [[True, True, True], [False] * 3, [False] * 3]
    assert result["added_counts"] == [2, 0, 0]
    scores = score_candidates([candidate, action(id=2), AWMAction(kind="invalid")], [teacher, teacher, candidate], tool_match_counts=result["counts"], teacher_sample_count=3, frequency_bonus_scale=0.5)
    assert [s.reward for s in scores] == [1.5, 0.0, -1.0]
    assert len(calls) == 3  # One prose pair and two different-ID negatives.
    assert canonical_action(candidate) == original != canonical_action(teacher)
    assert client.stats()["tool_matcher_positive_pairs"] == 2


@pytest.mark.parametrize(
    "candidate,chat,schema",
    [
        (action(id=2), CHAT, {}),
        (AWMAction(kind="tool", name="other", arguments=action().arguments), CHAT, {}),
        (action("paid"), CHAT, {"enum": ["Payment completed.", "paid"]}),
        (action("paid"), CHAT, {"pattern": ".*"}),
        (action("paid"), CHAT, {"description": "Exact template string"}),
        (action("paid"), [{"role": "user", "content": 'Set note to "Payment completed.".'}], {}),
        (action(""), CHAT, {}),
        (action('{"paid":true}'), CHAT, {}),
        (action("Paid.", extra=True), CHAT, {}),
    ],
)
def test_unknown_differences_reach_source_matcher_but_invalid_and_other_tools_do_not(candidate, chat, schema):
    tools = copy.deepcopy(TOOLS)
    tools[0]["inputSchema"]["properties"]["note"].update(schema)
    candidate = validate_action(candidate, tools)
    plan = build_tool_match_plan([action()], [candidate], tools, chat)
    assert bool(plan["pairs"]) == (candidate.kind == "tool" and candidate.name == "log_interaction")
    assert finish_tool_match_plan(plan, [False] * len(plan["pairs"]))["counts"] == {0: 0}
    if plan["pairs"]:
        assert plan["pairs"][0]["tool_matching_metadata"]["source"] == METADATA["log_interaction"]["source"]


def test_nested_nullable_ref_text_and_strict_arrays():
    schema = {"type": "object", "$defs": {"Text": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "properties": {"body": {"type": "object", "properties": {"note": {"$ref": "#/$defs/Text"}, "ids": {"type": "array", "items": {"type": "integer"}}}}}}
    tools = [{"name": "log_interaction", "inputSchema": schema}]
    teacher = AWMAction(kind="tool", name="log_interaction", arguments={"body": {"note": "Paid.", "ids": [1, 2]}})
    candidate = AWMAction(kind="tool", name="log_interaction", arguments={"body": {"note": "Payment completed.", "ids": [1, 2]}})
    plan = build_tool_match_plan([teacher], [candidate], tools, CHAT)
    assert plan["pairs"][0]["differing_paths"] == [["body", "note"]]
    candidate.arguments["body"]["ids"] = [2, 1]
    assert ["body", "ids", 0] in build_tool_match_plan([teacher], [candidate], tools, CHAT)["pairs"][0]["differing_paths"]


@pytest.mark.parametrize("decision", [[1], ["true"], [], {"equivalent": True}, [True, True]])
def test_bad_matcher_result_is_not_a_false_label(decision):
    plan = build_tool_match_plan([action()], [action("Paid.")], TOOLS, CHAT)
    with pytest.raises(ValueError, match="Booleans"):
        finish_tool_match_plan(plan, decision)


def test_cache_is_persistent_context_sensitive_and_separate_from_messages(tmp_path):
    path = str(tmp_path / "matcher.jsonl")
    client = DeepSeekAWMOracleClient(matcher_cache_path=path, request_fn=lambda p: response(False))
    evidence = build_tool_match_plan([action()], [action("Paid.")], TOOLS, CHAT)["pairs"][0]
    assert client.match_tool_argument_pairs([evidence]) == [False]
    client.match_message_pairs(["teacher"], ["candidate"])
    calls = []
    resumed = DeepSeekAWMOracleClient(matcher_cache_path=path, request_fn=lambda p: calls.append(p) or response(True))
    assert resumed.match_tool_argument_pairs([evidence]) == [False]
    assert resumed.match_message_pairs(["teacher"], ["candidate"])["counts"] == [0]
    assert not calls
    changed = copy.deepcopy(evidence)
    changed["public_context"].append({"role": "user", "content": "Actually do not pay."})
    assert resumed.match_tool_argument_pairs([changed]) == [True]
    assert len(calls) == 1


def test_concurrent_duplicate_pair_single_flight():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def request(payload):
        calls.append(payload)
        entered.set()
        assert release.wait(10)
        return response(True)

    client = DeepSeekAWMOracleClient(request_fn=request)
    pair = build_tool_match_plan([action()], [action("Paid.")], TOOLS, CHAT)["pairs"][0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client._match_tool_pair, pair)
        assert entered.wait(10)
        second = pool.submit(client._match_tool_pair, pair)
        release.set()
        assert first.result() is second.result() is True
    assert len(calls) == 1


def test_old_tool_verdict_protocol_is_not_reused(tmp_path):
    path = tmp_path / "matcher.jsonl"
    client = DeepSeekAWMOracleClient(matcher_cache_path=str(path), request_fn=lambda p: response(True))
    pair = build_tool_match_plan([action()], [action("Paid.")], TOOLS, CHAT)["pairs"][0]
    assert client.match_tool_argument_pairs([pair]) == [True]
    record = json.loads(path.read_text())
    record["protocol_version"] = 2
    path.write_text(json.dumps(record) + "\n")
    calls = []
    resumed = DeepSeekAWMOracleClient(matcher_cache_path=str(path), request_fn=lambda p: calls.append(p) or response(False))
    assert resumed.match_tool_argument_pairs([pair]) == [False]
    assert len(calls) == 1


@pytest.mark.parametrize("family", ["awm", "envscaler"])
@pytest.mark.parametrize("failure", [False, True])
def test_worker_scores_before_one_commit_or_masks_only_current_group(family, failure):
    def match(pairs):
        if failure:
            raise RuntimeError("matcher unavailable")
        return [True] * len(pairs)

    oracle = SimpleNamespace(match_tool_argument_pairs=Remote(match))
    if family == "awm":
        worker = AWMWorker.__ray_metadata__.modified_class(base_url="unused", max_steps=20, verifier_mode="sql", reward_mode="semantic", oracle_actor=oracle, tool_argument_matcher_enabled=True)
        worker._scenario, worker._task_idx, worker._task = "test", 0, "test"
        scenario = "test"
    else:
        worker = EnvScalerWorker.__ray_metadata__.modified_class(oracle_actor=oracle, tool_argument_matcher_enabled=True)
        worker._task = {"env_id": "test", "task_id": "test", "checklist_with_func": []}
        worker._task_index = 0
        worker._runtime = SimpleNamespace()
        worker._initial_state = {}
        scenario = "envscaler:test"
    worker._chat, worker._tools = copy.deepcopy(CHAT), copy.deepcopy(TOOLS)
    worker._tool_matching_metadata = METADATA
    worker._step = 2  # A previous valid group must not be retroactively removed.
    teacher = action()
    samples = [{"sample_index": i, "action": teacher.to_dict()} for i in range(3)]
    worker._prepared_teacher_supervision = {
        "state_fingerprint": state_fingerprint(scenario, 0, worker._chat, worker._tools),
        "teacher_samples": samples,
        "teacher_actions": [teacher] * 3,
        "teacher_multiset": [teacher.to_dict()] * 3,
        "teacher_invalid_sample_count": 0,
        "teacher_action_kind_disagreement": False,
    }
    executed = []

    async def execute(raw, parsed):
        executed.append(parsed)
        worker._last_info = {"runtime_train_mask": True}
        return (0.0, False) if family == "awm" else False

    worker._execute = execute
    original_chat = copy.deepcopy(worker._chat)
    raw = '<tool_call>{"name":"log_interaction","arguments":{"id":1,"note":"Paid."}}</tool_call>'
    results, selected, _, _, done, info = asyncio.run(worker.step_candidate_group([raw] * 4))
    if failure:
        assert selected == -1 and done and info["matcher_failure"]
        assert not executed and worker._chat == original_chat and worker._step == 2
        assert all(not r[3]["semantic_train_mask"] for r in results)
    else:
        assert len(executed) == 1 and selected in range(4)
        assert all(r[3]["teacher_frequency"] == 3 for r in results)
        assert all(r[3]["tool_argument_semantic_match_count"] == 3 for r in results)


def test_exact_only_group_never_calls_matcher_and_invalid_stays_invalid():
    invalid = validate_action(action(None), TOOLS)
    assert invalid.kind == "invalid"
    result = asyncio.run(match_candidate_tools(None, [action()] * 3, [action(), invalid], TOOLS, CHAT))
    assert result["counts"] == {0: 3, 1: 0}


def test_object_enum_and_boolean_numeric_differences_are_not_program_equated():
    tools = copy.deepcopy(TOOLS)
    tools[0]["inputSchema"]["enum"] = [action().arguments, action("Paid.").arguments]
    plan = build_tool_match_plan([action()], [action("Paid.")], tools, CHAT)
    assert plan["matrix"] == [[False]] and len(plan["pairs"]) == 1
    tools[0]["inputSchema"].pop("enum")
    tools[0]["inputSchema"]["properties"]["id"] = {"type": ["boolean", "integer"]}
    plan = build_tool_match_plan([action()], [action("Paid.", id=True)], tools, CHAT)
    assert plan["matrix"] == [[False]] and len(plan["pairs"]) == 1


def test_failed_tool_match_is_not_cached_and_can_retry():
    calls = []

    def request(payload):
        calls.append(payload)
        return response("invalid") if len(calls) == 1 else response(True)

    client = DeepSeekAWMOracleClient(request_fn=request)
    pairs = build_tool_match_plan([action()], [action("Paid.")], TOOLS, CHAT)["pairs"]
    with pytest.raises(ValueError, match="boolean"):
        client.match_tool_argument_pairs(pairs)
    assert client.match_tool_argument_pairs(pairs) == [True]
    assert client.stats()["matcher_failures"] == 1
    assert not client._tool_matcher_flights
