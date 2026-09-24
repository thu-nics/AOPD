"""Regression contracts for lossless action matching, independent of rollouts."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent_system.environments.action_matching import (
    MESSAGE_MATCHER_INSTRUCTION,
    callable_defaults,
    finish_tool_match_plan,
    source_defaults,
)
from agent_system.environments.action_matching import build_tool_match_plan as _build_tool_match_plan
from agent_system.environments.env_package.awm.runtime.actions import AWMAction, parse_action, validate_action
from agent_system.environments.env_package.awm.runtime.oracle import ORACLE_PROTOCOL_VERSION, DeepSeekAWMOracleClient
from agent_system.environments.env_package.tau_bench.actions import ParsedAction, validate_tau_action
from agent_system.environments.env_package.tau_bench.actions import parse_action as tau_parse
from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient
from agent_system.environments.tool_matching_metadata import source_tool_matching_metadata

TOOLS = [{"name": "write", "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}, "note": {"type": "string"}, "enabled": {"type": "boolean", "default": True}, "parent_id": {"type": ["integer", "null"]}}, "required": ["id"], "additionalProperties": False}}]
NATIVE = [{"type": "function", "function": {"name": "write", "parameters": TOOLS[0]["inputSchema"]}}]
CHAT = [{"role": "user", "content": "Record the refund for customer 1."}]
METADATA = source_tool_matching_metadata("def write(id, note=None):\n    return {'id': id, 'note': note}\n", TOOLS, family="test", environment="test")


def test_message_instruction_is_shared_and_checks_constraints_before_paraphrases():
    from agent_system.environments.env_package.awm.runtime import oracle as awm_oracle
    from agent_system.environments.env_package.tau_bench import oracle as tau_oracle

    assert awm_oracle.MATCHER_INSTRUCTION == MESSAGE_MATCHER_INSTRUCTION
    assert tau_oracle.MATCHER_SEMANTICS.startswith(MESSAGE_MATCHER_INSTRUCTION.split("Return only", 1)[0])
    assert "constraints BEFORE allowing paraphrases" in MESSAGE_MATCHER_INSTRUCTION
    assert "extra relevant clarification" in MESSAGE_MATCHER_INSTRUCTION
    assert "A completion or handoff announcement cannot replace a question, offer, or conditional plan" in MESSAGE_MATCHER_INSTRUCTION
    assert "text describing a tool call is not an executed tool call" in MESSAGE_MATCHER_INSTRUCTION
    assert "Do not drop a prerequisite or add an unsupported claim" in MESSAGE_MATCHER_INSTRUCTION
    assert "requires literal text, preserve that text rather than paraphrasing it" in MESSAGE_MATCHER_INSTRUCTION
    assert "another pair's match cannot justify this pair" in MESSAGE_MATCHER_INSTRUCTION
    assert MESSAGE_MATCHER_INSTRUCTION.index("If either constraint is violated, return false") < MESSAGE_MATCHER_INSTRUCTION.index("Only within those constraints, accept paraphrases")
    assert "not an action-quality judge" in MESSAGE_MATCHER_INSTRUCTION


def test_message_optional_detail_does_not_relax_material_constraints():
    assert "An explicitly optional field with a known lookup fallback" in MESSAGE_MATCHER_INSTRUCTION
    assert "an extra mandatory question is" in MESSAGE_MATCHER_INSTRUCTION
    assert "Successful linked tool results" in MESSAGE_MATCHER_INSTRUCTION
    assert "subject to any literal-text requirement above" in MESSAGE_MATCHER_INSTRUCTION
    assert "payment/refund direction, eligibility conditions or execution status" in MESSAGE_MATCHER_INSTRUCTION


@pytest.mark.parametrize("family", ["awm", "tau"])
def test_message_prompt_hash_change_preserves_teacher_and_tool_cache(family, tmp_path, monkeypatch):
    from agent_system.environments.env_package.awm.runtime import oracle as awm_oracle
    from agent_system.environments.env_package.tau_bench import oracle as tau_oracle

    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    cls = DeepSeekAWMOracleClient if family == "awm" else TauTeacherClient
    module = awm_oracle if family == "awm" else tau_oracle
    hash_name = "MATCHER_PROMPT_HASH" if family == "awm" else "MATCHER_SEMANTICS_HASH"
    teacher_path = tmp_path / "teacher.jsonl"
    kwargs = {"cache_path": str(teacher_path), "matcher_cache_path": str(tmp_path / "matcher.jsonl")}
    if family == "awm":
        kwargs["request_fn"] = lambda payload: pytest.fail("transport must be mocked")
    first = cls(**kwargs)

    def initial_response(payload, **kwargs):
        if "tools" in payload:
            return {"model": "deepseek-v4-flash", "choices": [{"message": {"tool_calls": [{"function": {"name": "write", "arguments": '{"id":1}'}}]}}]}
        return _response(True)

    monkeypatch.setattr(first, "_post", initial_response)
    if family == "tau":
        monkeypatch.setattr(first, "_sample_once", lambda **kwargs: ParsedAction(kind="tool", name="write", arguments={"id": 1}))
    votes = first.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    assert len(votes) == 3
    assert first.match_message_pairs(["Refund sent."], ["Refund issued."], CHAT, NATIVE)["counts"] == [1]
    pair = build_tool_match_plan([AWMAction(kind="tool", name="write", arguments={"id": 1, "note": "Refund sent."})], [AWMAction(kind="tool", name="write", arguments={"id": 1, "note": "Refund issued."})], NATIVE, CHAT)["pairs"]
    assert first.match_tool_argument_pairs(pair) == [True]
    before = teacher_path.read_bytes()

    monkeypatch.setattr(module, hash_name, "changed-message-prompt-only")
    second = cls(**kwargs)
    calls = []
    monkeypatch.setattr(second, "_post", lambda payload, **kwargs: calls.append(payload) or _response(False))
    assert second.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE) == votes
    assert second.match_tool_argument_pairs(pair) == [True]
    assert calls == []
    assert teacher_path.read_bytes() == before
    assert second.match_message_pairs(["Refund sent."], ["Refund issued."], CHAT, NATIVE)["counts"] == [0]
    assert len(calls) == 1


def build_tool_match_plan(*args, **kwargs):
    kwargs.setdefault("tool_matching_metadata", METADATA)
    return _build_tool_match_plan(*args, **kwargs)


@pytest.mark.parametrize("parser", [parse_action, tau_parse])
def test_tool_examples_in_reasoning_are_not_actions(parser):
    example = '<tool_call>{"name":"example","arguments":{}}</tool_call>'
    actual = '<tool_call>{"name":"write","arguments":{"id":1}}</tool_call>'
    assert parser(f"<think>{example}</think>{actual}").name == "write"
    assert parser(f"<think>{example}{actual}").kind == "invalid"


@pytest.mark.parametrize("value", [[], False, 0, None])
def test_non_object_arguments_never_become_empty_objects(value):
    action = AWMAction(kind="tool", name="write", arguments=value)
    assert validate_action(action, TOOLS).kind == "invalid"


def test_null_and_missing_are_distinct_in_native_pydantic_execution():
    class Params(BaseModel):
        parent_id: int | None = None

    tool = SimpleNamespace(name="write", params=Params)
    for arguments in [{}, {"parent_id": None}]:
        awm = validate_action(AWMAction(kind="tool", name="write", arguments={"id": 1, **arguments}), TOOLS)
        tau = validate_tau_action(ParsedAction(kind="tool", name="write", arguments=arguments), [tool])
        assert ("parent_id" in awm.arguments) == ("parent_id" in arguments)
        assert Params(**tau.arguments).model_fields_set == set(arguments)


def test_literal_arguments_are_not_stripped_or_casefolded():
    action = validate_action(AWMAction(kind="tool", name="write", arguments={"id": 1, "note": "  AbC\n  X  "}), TOOLS)
    assert action.arguments["note"] == "  AbC\n  X  "
    tools = copy.deepcopy(TOOLS)
    tools[0]["inputSchema"]["properties"]["note"]["enum"] = ["OPEN"]
    assert validate_action(AWMAction(kind="tool", name="write", arguments={"id": 1, "note": "open"}), tools).kind == "invalid"


def test_only_execution_proven_defaults_are_equivalent():
    teacher = AWMAction(kind="tool", name="write", arguments={"id": 1, "enabled": True})
    candidate = AWMAction(kind="tool", name="write", arguments={"id": 1})

    def write(id, enabled=True):
        pass

    metadata = copy.deepcopy(METADATA)
    metadata["write"]["defaults"] = callable_defaults(write)
    assert finish_tool_match_plan(build_tool_match_plan([teacher], [candidate], TOOLS, CHAT), [False])["counts"] == {0: 0}
    assert finish_tool_match_plan(build_tool_match_plan([teacher] * 3, [candidate], TOOLS, CHAT, tool_matching_metadata=metadata), [])["counts"] == {0: 3}
    assert candidate.arguments == {"id": 1}
    explicit_null = AWMAction(kind="tool", name="write", arguments={"id": 1, "parent_id": None})
    assert finish_tool_match_plan(build_tool_match_plan([explicit_null], [candidate], TOOLS, CHAT), [False])["counts"] == {0: 0}


def test_source_defaults_require_plain_native_declarations():
    code = """
class Request(BaseModel):
    id: int
    enabled: bool = Field(True)
@app.post('/write', operation_id='write')
def endpoint(body: Request):
    return body.enabled
"""
    assert source_defaults(code, TOOLS) == {"write": {"enabled": True}}
    assert source_defaults(code + "\n# model_fields_set is inspected\n", TOOLS) == {}


def test_source_defaults_follow_public_operation_id_not_later_private_helper():
    source = "@app.get('/fetch', operation_id='fetch')\ndef endpoint(limit=7):\n return limit\ndef fetch(limit=9):\n return limit\n"
    tools = [{"name": "fetch", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}}]
    assert source_defaults(source, tools) == {"fetch": {"limit": 7}}


def test_real_mcp_preserves_clear_null_and_keeps_validation(tmp_path):
    from agent_world_model_env.server.scenario_manager import ScenarioProcess

    from agent_system.environments.env_package.awm.runtime.tool_schema import install_action_schema_patch

    install_action_schema_patch()
    code = """
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()
class Request(BaseModel):
    id: int
    parent_id: int | None = None
@app.post('/clear', operation_id='clear_parent')
def clear(body: Request):
    return {'cleared': 'parent_id' in body.model_fields_set and body.parent_id is None}
if __name__ == '__main__':
    uvicorn.run(app)
"""
    process = ScenarioProcess()
    try:
        process.start(code, str(tmp_path / "unused.db"), str(tmp_path))
        result = process.call_tool("clear_parent", {"id": 1, "parent_id": None})
        assert result["success"], result
        assert json.loads(result["result"])["cleared"]
        missing = process.call_tool("clear_parent", {"id": 1})
        assert missing["success"] and not json.loads(missing["result"])["cleared"]
        assert not process.call_tool("clear_parent", {"id": 1, "parent_id": "not-an-integer"})["success"]
    finally:
        process.stop()


def test_unknown_prose_arrays_and_ids_require_source_matcher():
    schema = {"type": "object", "properties": {"utterances": {"type": "array", "items": {"type": "string"}}, "customer_id": {"type": "string"}}}
    tools = [{"name": "write", "inputSchema": schema}]
    teacher = AWMAction(kind="tool", name="write", arguments={"utterances": ["Refund sent."], "customer_id": "ABC"})
    candidate = AWMAction(kind="tool", name="write", arguments={"utterances": ["The refund was issued."], "customer_id": "ABC"})
    assert build_tool_match_plan([teacher], [candidate], tools, CHAT)["pairs"][0]["differing_paths"] == [["utterances", 0]]
    candidate.arguments["customer_id"] = "abc"
    assert ["customer_id"] in build_tool_match_plan([teacher], [candidate], tools, CHAT)["pairs"][0]["differing_paths"]


def _response(value):
    return {"model": "deepseek-v4-flash", "system_fingerprint": "test", "choices": [{"message": {"content": json.dumps({"equivalent": value})}}]}


def test_message_cache_includes_public_context_and_literals(tmp_path):
    calls = []
    path = str(tmp_path / "matcher.jsonl")
    client = DeepSeekAWMOracleClient(matcher_cache_path=path, request_fn=lambda payload: calls.append(payload) or _response(False))
    assert client.match_message_pairs(["ID ABC"], ["ID abc"], CHAT, NATIVE)["counts"] == [0]
    assert calls[0]["messages"][0]["role"] == "system"
    assert json.loads(calls[0]["messages"][1]["content"])["public_context"] == CHAT
    resumed = DeepSeekAWMOracleClient(matcher_cache_path=path, request_fn=lambda payload: calls.append(payload) or _response(True))
    assert resumed.match_message_pairs(["ID ABC"], ["ID abc"], CHAT, NATIVE)["counts"] == [0]
    assert len(calls) == 1
    assert resumed.match_message_pairs(["ID ABC"], ["ID abc"], [*CHAT, {"role": "user", "content": "Different context"}], NATIVE)["counts"] == [1]
    assert len(calls) == 2


def test_import_revalidates_raw_nulls_and_preserves_source(tmp_path):
    source, target = tmp_path / "old.jsonl", tmp_path / "new.jsonl"

    def request(payload):
        return {"model": "deepseek-v4-flash", "choices": [{"message": {"tool_calls": [{"function": {"name": "write", "arguments": '{"id":1,"parent_id":null}'}}]}}]}

    old = DeepSeekAWMOracleClient(cache_path=str(source), request_fn=request)
    old.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    record = json.loads(source.read_text())
    record["protocol_version"] = 14
    for sample in record["teacher_samples"]:
        sample["action"]["arguments"].pop("parent_id")
    source.write_text(json.dumps(record) + "\n")
    before = source.read_bytes()
    fresh = DeepSeekAWMOracleClient(cache_path=str(target), teacher_cache_import_paths=[str(source)], request_fn=lambda p: pytest.fail("import should not query the teacher"))
    votes = fresh.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    assert len(votes) == 3 and all(vote["action"]["arguments"]["parent_id"] is None for vote in votes)
    assert source.read_bytes() == before
    assert json.loads(target.read_text())["protocol_version"] == ORACLE_PROTOCOL_VERSION
    assert fresh.stats()["teacher_cache_imported_votes"] == 3
    assert fresh.stats()["teacher_requests"] == 0


def test_tau_tool_matcher_cache_and_transfer_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    client = TauTeacherClient(matcher_cache_path=str(tmp_path / "matcher.jsonl"))
    calls = []
    client._post = lambda payload, **kwargs: calls.append(payload) or _response(True)
    teacher = ParsedAction(kind="tool", name="write", arguments={"id": 1, "note": "Refund sent."})
    candidate = ParsedAction(kind="tool", name="write", arguments={"id": 1, "note": "Refund issued."})
    plan = build_tool_match_plan([teacher] * 3, [candidate], NATIVE, CHAT)
    assert finish_tool_match_plan(plan, client.match_tool_argument_pairs(plan["pairs"]))["counts"] == {0: 3}
    assert len(calls) == 1
    resumed = TauTeacherClient(matcher_cache_path=str(tmp_path / "matcher.jsonl"))
    resumed._post = lambda payload: pytest.fail("persistent match should hit")
    assert resumed.match_tool_argument_pairs(plan["pairs"]) == [True] * 3


@pytest.mark.parametrize("field,value", [("api_base", "https://unrelated.invalid/v1"), ("teacher_prompt_hash", "changed")])
def test_cache_import_rejects_changed_endpoint_or_prompt(tmp_path, field, value):
    source = tmp_path / "source.jsonl"

    def request(payload):
        return {"model": "deepseek-v4-flash", "choices": [{"message": {"content": "Complete."}}]}

    old = DeepSeekAWMOracleClient(cache_path=str(source), request_fn=request)
    old.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    record = json.loads(source.read_text())
    if field == "teacher_prompt_hash":
        record["teacher_protocol_config"][field] = value
    else:
        record[field] = value
    source.write_text(json.dumps(record) + "\n")
    calls = []
    new = DeepSeekAWMOracleClient(teacher_cache_import_paths=[str(source)], request_fn=lambda p: calls.append(p) or request(p))
    new.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    assert len(calls) == 3


@pytest.mark.parametrize('import_version,expected_calls,imported_votes', [(10, 1, 2), (8, 3, 0), (9, 3, 0)])
def test_tau_cache_import_revalidates_and_refills_only_missing_votes(tmp_path, monkeypatch, import_version, expected_calls, imported_votes):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    source, target = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    old = TauTeacherClient(cache_path=str(source))
    old._sample_once = lambda **kwargs: ParsedAction(kind="tool", name="write", arguments={"id": 1})
    old.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)
    record = json.loads(source.read_text())
    record["protocol_version"] = import_version
    record["teacher_samples"] = record["teacher_samples"][:2]
    record["valid_samples"] = 2
    source.write_text(json.dumps(record) + "\n")
    before = source.read_bytes()
    new = TauTeacherClient(cache_path=str(target), teacher_cache_import_paths=[str(source)])
    calls = []
    new._sample_once = lambda **kwargs: calls.append(kwargs) or ParsedAction(kind="tool", name="write", arguments={"id": 1})
    assert len(new.sample_multiset(state_fingerprint="state", messages=CHAT, tools=NATIVE)) == 3
    assert len(calls) == expected_calls and source.read_bytes() == before
    assert new.stats().get("cache_imported_votes", 0) == imported_votes
