from types import SimpleNamespace

from agent_system.environments.env_package.awm.actions import (
    AWMAction,
    append_exchange,
    build_scaffold_chat,
    canonical_action,
    format_tools_for_response,
    parse_action,
    score_candidates,
    validate_action,
)
from agent_system.environments.env_package.awm.envs import (
    validate_teacher_multiset,
)
from agent_system.environments.env_package.awm.manager import AWMEnvironmentManager

TOOLS = [
    SimpleNamespace(
        name="lookup",
        description="Look up one item.",
        input_schema={
            "type": "object",
            "properties": {"item_id": {"type": "integer"}},
            "required": ["item_id"],
            "additionalProperties": False,
        },
    )
]


def test_parses_native_wrapper_and_canonicalizes_arguments():
    action = validate_action(
        parse_action('<think>check</think><tool_call>{"name":"call_tool","arguments":{"tool_name":"mcp_tool_lookup","arguments":"{\\"item_id\\": 7}"}}</tool_call>'),
        TOOLS,
    )
    assert action.kind == "tool"
    assert canonical_action(action) == ('{"arguments":{"item_id":7},"kind":"tool","name":"lookup"}')


def test_parses_deepseek_dsml_tool_call_closing_tag():
    raw = '<tool_call>{"name":"call_tool","arguments":{"tool_name":"mcp_tool_lookup","arguments":{"item_id":7}}}</｜｜DSML｜｜tool_call>'
    action = validate_action(parse_action(raw), TOOLS)
    assert action == AWMAction(kind="tool", name="lookup", arguments={"item_id": 7})


def test_parses_deepseek_nested_dsml_end_marker():
    raw = '<tool_call>{"name":"call_tool","arguments":{"tool_name":"mcp_tool_lookup","arguments":{"item_id":7}}}</｜｜DSML｜｜></tool_call>'
    action = validate_action(parse_action(raw), TOOLS)
    assert action == AWMAction(kind="tool", name="lookup", arguments={"item_id": 7})


def test_complete_schema_render_and_nested_canonicalization():
    tools = [
        {
            "name": "batch_update",
            "description": "Update structured records.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_id": {
                                    "anyOf": [
                                        {"type": "integer"},
                                        {"type": "null"},
                                    ]
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["Open", "Closed"],
                                },
                                "at": {
                                    "type": "string",
                                    "format": "date-time",
                                },
                                "timezone": {"type": "string"},
                            },
                            "required": ["item_id", "status", "at", "timezone"],
                        },
                    }
                },
                "required": ["records"],
            },
        }
    ]
    rendered = format_tools_for_response(tools)
    assert '"items"' in rendered
    assert '"item_id"' in rendered
    assert '"format":"date-time"' in rendered

    action = validate_action(
        AWMAction(
            kind="tool",
            name="batch_update",
            arguments={
                "records": [
                    {
                        "item_id": "7",
                        "status": "open",
                        "at": "2026-08-01T10:00:00+08:00",
                        "timezone": "america/new_york",
                    }
                ]
            },
        ),
        tools,
    )
    assert action.kind == "tool"
    assert action.arguments == {
        "records": [
            {
                "item_id": 7,
                "status": "Open",
                "at": "2026-08-01T02:00:00Z",
                "timezone": "America/New_York",
            }
        ]
    }


def test_final_response_is_a_message_without_action_kind_gate():
    action = validate_action(parse_action("  The update is complete.  "), TOOLS)
    assert action == AWMAction(kind="message", content="The update is complete.")


def test_json_final_response_is_not_misclassified_as_a_tool_call():
    action = validate_action(parse_action('{"status":"complete"}'), TOOLS)
    assert action == AWMAction(kind="message", content='{"status":"complete"}')


def test_bare_tool_json_is_a_malformed_action_not_an_executable_call():
    raw = '{"name":"call_tool","arguments":{"tool_name":"mcp_tool_lookup","arguments":{"item_id":1}}}'
    action = parse_action(raw)
    assert action.kind == "invalid"
    assert action.error == "tool call missing <tool_call> wrapper"


def test_repeated_list_tools_is_masked_meta_action():
    action = parse_action('<tool_call>{"name":"list_tools","arguments":null}</tool_call>')
    scored = score_candidates([action], [])
    assert action.kind == "meta_list_tools"
    assert scored[0].reward is None
    assert scored[0].selection_score == -1.0
    assert scored[0].semantic_train_mask is False


def test_rejects_tool_call_mixed_with_final_response():
    action = parse_action('<tool_call>{"name":"mcp_tool_lookup","arguments":{"item_id":1}}</tool_call>Done.')
    assert action.kind == "invalid"


def test_frequency_reward_preserves_teacher_multiset_duplicates():
    a = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    b = AWMAction(kind="tool", name="lookup", arguments={"item_id": 2})
    message = AWMAction(kind="message", content="Done")
    invalid = AWMAction(kind="invalid", error="bad")
    rewards = score_candidates(
        [a, b, message, invalid],
        [a, a, b],
        message_match_counts={2: 2},
    )
    assert [item.reward for item in rewards] == [2.0, 1.0, 2.0, -1.0]
    assert [item.teacher_frequency for item in rewards] == [2, 1, 2, 0]


def test_teacher_multiset_keeps_only_valid_actions_without_deduplication():
    valid = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    meta = AWMAction(kind="meta_list_tools", name="list_tools", arguments={})
    invalid = AWMAction(kind="invalid", error="bad")
    samples = [
        {"action": valid.to_dict()},
        {"action": valid.to_dict()},
        {"action": meta.to_dict()},
    ]
    assert validate_teacher_multiset(samples, TOOLS) == [valid, valid]
    assert (
        validate_teacher_multiset(
            [{"action": meta.to_dict()}, {"action": invalid.to_dict()}],
            TOOLS,
        )
        == []
    )


def test_scaffold_is_pinned_and_history_window_keeps_complete_exchanges():
    chat = build_scaffold_chat("Do the task", TOOLS)
    pinned = list(chat)
    for index in range(5):
        chat = append_exchange(
            chat,
            assistant_content=f"action-{index}",
            tool_response=f"result-{index}",
            history_window=3,
        )
    assert chat[:4] == pinned
    assert [message["content"] for message in chat[4:] if message["role"] == "assistant"] == [
        "action-2",
        "action-3",
        "action-4",
    ]
    assert len(chat) == 10


def test_manager_reports_teacher_and_repeated_list_tools_rates():
    manager = AWMEnvironmentManager(None, None, None)
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "teacher_failure": False,
                    "teacher_invalid_sample_count": 1,
                    "teacher_sample_count": 3,
                    "action_kind": "tool",
                },
                {
                    "teacher_failure": True,
                    "teacher_invalid_sample_count": 2,
                    "teacher_sample_count": 3,
                    "action_kind": "teacher_failure",
                },
            ]
        ],
        total_batch_list=[
            [
                {
                    "is_action_valid": True,
                    "teacher_frequency": 2,
                    "semantic_train_mask": True,
                    "action_kind": "tool",
                },
                {
                    "is_action_valid": True,
                    "teacher_frequency": 0,
                    "semantic_train_mask": False,
                    "action_kind": "meta_list_tools",
                },
                {
                    "is_action_valid": False,
                    "teacher_frequency": 0,
                    "semantic_train_mask": True,
                    "action_kind": "invalid",
                },
                {
                    "is_action_valid": True,
                    "teacher_frequency": 1,
                    "semantic_train_mask": True,
                    "action_kind": "message",
                },
            ]
        ],
    )
    assert metrics["env/teacher_failure_rate"].tolist() == [0.5]
    assert metrics["env/teacher_invalid_sample_rate"].tolist() == [0.5]
    assert metrics["env/repeated_list_tools_rate"].tolist() == [0.25]
    assert metrics["env/semantic_masked_rate"].tolist() == [0.25]
    assert metrics["env/valid_action_rate"].tolist() == [0.75]
    assert metrics["env/teacher_frequency"].tolist() == [0.75]


def test_manager_exports_oracle_actor_usage_metrics(monkeypatch):
    class RemoteStats:
        @staticmethod
        def remote():
            return {"teacher_total_tokens": 123, "matcher_requests": 4}

    actor = SimpleNamespace(get_stats=RemoteStats())
    monkeypatch.setattr(
        "agent_system.environments.env_package.awm.manager.ray.get",
        lambda value: value,
    )
    manager = AWMEnvironmentManager(None, None, None, oracle_actor=actor)

    metrics = manager.success_evaluator(total_infos=[[]], total_batch_list=[[]])

    assert metrics["env/oracle_teacher_total_tokens"].tolist() == [123.0]
    assert metrics["env/oracle_matcher_requests"].tolist() == [4.0]
