from types import SimpleNamespace

from jsonschema import Draft202012Validator

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    append_exchange,
    build_native_chat,
    canonical_action,
    openai_tools,
    parse_action,
    parse_native_action,
    score_candidates,
    tool_schema_audit,
    validate_action,
)
from agent_system.environments.env_package.awm.runtime.envs import (
    validate_teacher_multiset,
)
from agent_system.environments.env_package.awm.runtime.manager import AWMEnvironmentManager

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


def test_nullable_schema_conflict_is_repaired_without_mutating_raw_schema():
    raw_schema = {
        "type": "object",
        "properties": {
            "pharmacy_id": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "type": "integer",
                "description": "Optional pharmacy filter.",
            }
        },
    }
    tools = [{"name": "list_inventory_items", "inputSchema": raw_schema}]

    audit = tool_schema_audit(tools)
    canonical = audit["canonical_tools"][0]["inputSchema"]
    repaired = canonical["properties"]["pharmacy_id"]

    assert raw_schema["properties"]["pharmacy_id"]["type"] == "integer"
    assert "type" not in repaired
    assert audit["raw_tool_schema_hash"] != audit["canonical_tool_schema_hash"]
    assert audit["schema_repairs"] == [
        {
            "tool_name": "list_inventory_items",
            "json_pointer": "/properties/pharmacy_id",
            "repair": "remove_redundant_nullable_sibling_type",
            "removed_type": "integer",
        }
    ]
    assert tool_schema_audit(audit["canonical_tools"])["schema_repairs"] == []
    action = validate_action(
        AWMAction(kind="tool", name="list_inventory_items", arguments={"pharmacy_id": None}),
        tools,
    )
    assert action.kind == "tool"
    assert action.arguments == {}


def test_duplicate_required_fields_are_losslessly_deduplicated():
    raw_schema = {
        "type": "object",
        "properties": {
            "company_id": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["company_id", "company_id", "email", "company_id"],
    }
    tools = [{"name": "create_employee", "inputSchema": raw_schema}]

    audit = tool_schema_audit(tools)
    canonical = audit["canonical_tools"][0]["inputSchema"]

    assert raw_schema["required"] == [
        "company_id",
        "company_id",
        "email",
        "company_id",
    ]
    assert canonical["required"] == ["company_id", "email"]
    assert audit["schema_repairs"] == [
        {
            "tool_name": "create_employee",
            "json_pointer": "/required",
            "repair": "deduplicate_required_fields",
            "removed_duplicates": ["company_id", "company_id"],
        }
    ]
    assert tool_schema_audit(audit["canonical_tools"])["schema_repairs"] == []
    Draft202012Validator.check_schema(canonical)


def test_required_field_inside_schema_default_is_not_rewritten():
    raw_schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "default": {"required": ["keep", "keep"]},
            }
        },
    }

    audit = tool_schema_audit([{"name": "submit", "inputSchema": raw_schema}])

    canonical = audit["canonical_tools"][0]["inputSchema"]
    assert canonical["properties"]["payload"]["default"]["required"] == ["keep", "keep"]
    assert audit["schema_repairs"] == []


def test_nullable_optional_argument_through_local_ref_is_omitted():
    tools = [
        {
            "name": "list_items",
            "inputSchema": {
                "type": "object",
                "$defs": {
                    "optional_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "type": "integer",
                    }
                },
                "properties": {
                    "item_id": {"$ref": "#/$defs/optional_id"},
                },
            },
        }
    ]

    action = validate_action(
        AWMAction(kind="tool", name="list_items", arguments={"item_id": None}),
        tools,
    )
    assert action.kind == "tool"
    assert action.arguments == {}


def test_parses_native_wrapper_and_canonicalizes_arguments():
    action = validate_action(
        parse_action('<think>check</think><tool_call>{"name":"lookup","arguments":"{\\"item_id\\": 7}"}</tool_call>'),
        TOOLS,
    )
    assert action.kind == "tool"
    assert canonical_action(action) == ('{"arguments":{"item_id":7},"kind":"tool","name":"lookup"}')


def test_parses_deepseek_dsml_tool_call_closing_tag():
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":7}}</｜｜DSML｜｜tool_call>'
    action = validate_action(parse_action(raw), TOOLS)
    assert action == AWMAction(kind="tool", name="lookup", arguments={"item_id": 7})


def test_parses_deepseek_nested_dsml_end_marker():
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":7}}</｜｜DSML｜｜></tool_call>'
    action = validate_action(parse_action(raw), TOOLS)
    assert action == AWMAction(kind="tool", name="lookup", arguments={"item_id": 7})


def test_parses_deepseek_dsml_tool_call_opening_and_closing_tags():
    raw = '<｜｜DSML｜｜tool_call>{"name":"lookup","arguments":{"item_id":7}}</｜｜DSML｜｜tool_call>'
    action = validate_action(parse_action(raw), TOOLS)
    assert action == AWMAction(kind="tool", name="lookup", arguments={"item_id": 7})


def test_rejects_unclosed_deepseek_dsml_tool_call():
    raw = '<｜｜DSML｜｜tool_call>{"name":"lookup","arguments":{"item_id":7}}'
    action = parse_action(raw)
    assert action.kind == "invalid"
    assert action.error == "unclosed tool-call tag"


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
    native = openai_tools(tools)
    assert native[0]["function"]["name"] == "batch_update"
    assert native[0]["function"]["parameters"] == tools[0]["inputSchema"]
    assert "strict" not in native[0]["function"]

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
    raw = '{"name":"lookup","arguments":{"item_id":1}}'
    action = parse_action(raw)
    assert action.kind == "invalid"
    assert action.error == "tool call missing <tool_call> wrapper"


def test_native_multiple_calls_differ_for_expert_and_student():
    calls = [
        {"id": "call-1", "function": {"name": "lookup", "arguments": '{"item_id":1}'}},
        {"id": "call-2", "function": {"name": "lookup", "arguments": '{"item_id":2}'}},
    ]
    expert, skipped = parse_native_action(None, calls, take_first=True)
    student, student_skipped = parse_native_action(None, calls, take_first=False)
    assert validate_action(expert, TOOLS) == AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    assert skipped == 1
    assert student.kind == "invalid"
    assert student.error == "multiple tool calls"
    assert student_skipped == 0


def test_native_student_rejects_tool_call_mixed_with_message():
    calls = [{"id": "call-1", "function": {"name": "lookup", "arguments": '{"item_id":1}'}}]
    student, skipped = parse_native_action("Done.", calls, take_first=False)
    expert, _ = parse_native_action("I will look it up.", calls, take_first=True)
    reasoning_only, _ = parse_native_action("<think>I should look it up.</think>", calls, take_first=False)
    assert student.kind == "invalid"
    assert student.error == "tool call mixed with communicative message"
    assert skipped == 0
    assert validate_action(expert, TOOLS).kind == "tool"
    assert validate_action(reasoning_only, TOOLS).kind == "tool"


def test_retired_meta_tool_names_are_not_environment_tools():
    for name in ("list_tools", "call_tool", "mcp_tool_lookup"):
        action = parse_action(f'<tool_call>{{"name":"{name}","arguments":{{}}}}</tool_call>')
        assert validate_action(action, TOOLS).kind == "invalid"


def test_rejects_tool_call_mixed_with_final_response():
    action = parse_action('<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>Done.')
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
    unknown = AWMAction(kind="tool", name="list_tools", arguments={})
    invalid = AWMAction(kind="invalid", error="bad")
    samples = [
        {"action": valid.to_dict()},
        {"action": valid.to_dict()},
        {"action": unknown.to_dict()},
    ]
    assert validate_teacher_multiset(samples, TOOLS) == [valid, valid]
    assert (
        validate_teacher_multiset(
            [{"action": unknown.to_dict()}, {"action": invalid.to_dict()}],
            TOOLS,
        )
        == []
    )


def test_native_prefix_is_pinned_and_history_keeps_linked_exchanges():
    chat = build_native_chat("Do the task")
    pinned = list(chat)
    for index in range(5):
        action = AWMAction(kind="tool", name="lookup", arguments={"item_id": index})
        chat = append_exchange(
            chat,
            action=action,
            raw_action="",
            tool_response=f"result-{index}",
            history_window=3,
            tool_call_id=f"call-{index}",
        )
    assert chat[:2] == pinned
    assistants = [message for message in chat[2:] if message["role"] == "assistant"]
    assert [message["tool_calls"][0]["id"] for message in assistants] == ["call-2", "call-3", "call-4"]
    assert all("reasoning_content" not in message for message in assistants)
    assert [message["tool_call_id"] for message in chat[2:] if message["role"] == "tool"] == ["call-2", "call-3", "call-4"]
    assert len(chat) == 8


def test_append_exchange_can_preserve_provider_reasoning_when_requested():
    chat = append_exchange(
        build_native_chat("Do the task"),
        action=AWMAction(kind="tool", name="lookup", arguments={"item_id": 1}),
        raw_action="",
        tool_response="result",
        history_window=3,
        tool_call_id="call-1",
        assistant_reasoning_content="provider reasoning",
    )

    assistant = next(message for message in chat if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "provider reasoning"
    assert assistant["tool_calls"][0]["id"] == "call-1"


def test_manager_reports_teacher_and_semantic_mask_rates():
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
                    "action_kind": "tool",
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
        "agent_system.environments.env_package.awm.runtime.manager.ray.get",
        lambda value: value,
    )
    manager = AWMEnvironmentManager(None, None, None, oracle_actor=actor)

    metrics = manager.success_evaluator(total_infos=[[]], total_batch_list=[[]])

    assert metrics["env/oracle_teacher_total_tokens"].tolist() == [123.0]
    assert metrics["env/oracle_matcher_requests"].tolist() == [4.0]


def test_manager_reports_context_overflow_separately_from_runtime_failure():
    manager = AWMEnvironmentManager(None, None, None)
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "action_kind": "context_overflow",
                    "context_overflow": True,
                    "context_prompt_tokens": 28050,
                    "context_excess_tokens": 146,
                    "runtime_failure": False,
                }
            ],
            [],
        ],
        total_batch_list=[[], []],
    )

    assert metrics["env/context_overflow_rate"].tolist() == [1.0, 0.0]
    assert metrics["env/context_overflow_prompt_tokens_mean"].tolist() == [
        28050.0,
        28050.0,
    ]
    assert metrics["env/context_overflow_excess_tokens_mean"].tolist() == [
        146.0,
        146.0,
    ]
    assert metrics["env/runtime_failure_rate"].tolist() == [0.0, 0.0]
    assert metrics["env/valid_action_rate"].tolist() == [0.0, 0.0]
