from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.tau_bench.actions import (
    ParsedAction,
    canonical_action,
    deduplicate_actions,
    parse_action,
    tau_messages_to_openai,
    teacher_action_match_key,
    to_tau_action,
    validate_tau_action,
)


def test_parse_qwen_tool_call_and_canonicalize_arguments():
    action = parse_action('<think>check</think><tool_call>{"name":"lookup","arguments":{"b":2.0,"a":1}}</tool_call>')
    assert action.kind == "tool"
    assert action.name == "lookup"
    assert canonical_action(action) == ('{"arguments":{"a":1,"b":2},"kind":"tool","name":"lookup"}')


def test_rejects_tool_call_mixed_with_user_message():
    action = parse_action("""<think>check</think><tool_call>{"name":"a","arguments":{}}</tool_call>I did it.""")
    assert action.kind == "invalid"
    assert action.error == "tool call mixed with user-facing message"


def test_teacher_matching_ignores_transfer_summary_without_mutating_raw_action():
    teacher = ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": "Please assist with a refund."})
    student = ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": "The customer needs a human."})
    raw_teacher = teacher.to_dict()
    assert teacher_action_match_key(raw_teacher) == teacher_action_match_key(student)
    assert raw_teacher == teacher.to_dict()
    assert canonical_action(teacher) != canonical_action(student)
    assert len(deduplicate_actions([teacher, student])) == 2
    assert '"summary": "The customer needs a human."' in to_tau_action(student)


@pytest.mark.parametrize("name", ["send_message", "transfer_funds", "lookup"])
def test_teacher_matching_keeps_other_tools_arguments_exact(name):
    first = ParsedAction(kind="tool", name=name, arguments={"summary": "first", "id": 1})
    different_text = ParsedAction(kind="tool", name=name, arguments={"summary": "second", "id": 1})
    different_id = ParsedAction(kind="tool", name=name, arguments={"summary": "first", "id": 2})
    assert teacher_action_match_key(first) == canonical_action(first)
    assert teacher_action_match_key(first) != teacher_action_match_key(different_text)
    assert teacher_action_match_key(first) != teacher_action_match_key(different_id)


@pytest.mark.parametrize("arguments", [{}, {"summary": 42}, {"summary": "help", "extra": True}])
def test_transfer_reward_matching_does_not_relax_schema_validation(arguments):
    from pydantic import BaseModel

    class Params(BaseModel):
        summary: str

    tool = SimpleNamespace(name="transfer_to_human_agents", params=Params)
    invalid = validate_tau_action(ParsedAction(kind="tool", name=tool.name, arguments=arguments), [tool])
    valid = validate_tau_action(ParsedAction(kind="tool", name=tool.name, arguments={"summary": "help"}), [tool])
    assert invalid.kind == "invalid"
    assert teacher_action_match_key(invalid) != teacher_action_match_key(valid)


def test_rejects_unclosed_reasoning_as_an_action():
    action = parse_action("<think>still reasoning")
    assert action.kind == "invalid"
    assert action.error == "unclosed reasoning tag"


def test_rejects_multiple_tool_calls():
    action = parse_action('<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert action.kind == "invalid"
    assert action.error == "multiple tool calls"


def test_deduplicates_three_independent_samples_into_empirical_set():
    actions = [
        ParsedAction(kind="tool", name="a", arguments={"x": 1}),
        ParsedAction(kind="tool", name="a", arguments={"x": 1.0}),
        ParsedAction(kind="message", content="  hello   there "),
    ]
    assert len(deduplicate_actions(actions)) == 2


def test_tool_validation_is_strict_and_rejects_unknown_arguments():
    from pydantic import BaseModel

    class Params(BaseModel):
        item_id: int

    tool = SimpleNamespace(name="lookup", params=Params)
    valid = validate_tau_action(
        ParsedAction(kind="tool", name="lookup", arguments={"item_id": 1}),
        [tool],
    )
    wrong_type = validate_tau_action(
        ParsedAction(kind="tool", name="lookup", arguments={"item_id": "1"}),
        [tool],
    )
    extra = validate_tau_action(
        ParsedAction(kind="tool", name="lookup", arguments={"item_id": 1, "extra": 2}),
        [tool],
    )
    assert valid.kind == "tool"
    assert wrong_type.kind == "invalid"
    assert extra.kind == "invalid"


def test_tau_history_links_generated_tool_call_id_to_result():
    tool_call = SimpleNamespace(id="", name="lookup", arguments={"id": "1"})
    messages = [
        SimpleNamespace(role="user", content="help", tool_calls=None),
        SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call]),
        SimpleNamespace(role="tool", content="found", id="", requestor="assistant", tool_calls=None),
    ]
    converted = tau_messages_to_openai(messages)
    call_id = converted[1]["tool_calls"][0]["id"]
    assert call_id
    assert converted[2]["tool_call_id"] == call_id
