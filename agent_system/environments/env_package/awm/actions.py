"""AWM action parsing, schema validation, canonicalization, and scaffolding."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from zoneinfo import available_timezones

_TOOL_CALL_CLOSE = r"(?:(?:</｜｜DSML｜｜>\s*)?</tool_call>|</｜｜DSML｜｜tool_call>)"
_TOOL_CALL_RE = re.compile(
    rf"<tool_call>\s*(.*?)\s*{_TOOL_CALL_CLOSE}",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROTOCOL_VERSION = 4


@dataclass(frozen=True)
class AWMAction:
    """One semantic action in the unified AWM action space."""

    kind: str
    name: str | None = None
    arguments: dict[str, Any] | None = None
    content: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredCandidate:
    action: AWMAction
    reward: float | None
    selection_score: float
    semantic_train_mask: bool
    teacher_frequency: int


def normalize_message(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value


def _parse_tool_payload(value: Any) -> AWMAction:
    if not isinstance(value, dict):
        raise ValueError("tool call must be a JSON object")
    if "function" in value:
        value = value["function"]
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call name must be a non-empty string")
    name = name.strip()
    arguments = _json_object(arguments or {})

    if name == "list_tools":
        return AWMAction(kind="meta_list_tools", name="list_tools", arguments={})
    if name == "call_tool":
        tool_name = arguments.get("tool_name", "")
        inner_arguments = arguments.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("call_tool.tool_name must be a non-empty string")
        inner_arguments = _json_object(inner_arguments or {})
        tool_name = tool_name.strip()
        if tool_name.startswith("mcp_tool_"):
            tool_name = tool_name[len("mcp_tool_") :]
        return AWMAction(kind="tool", name=tool_name, arguments=inner_arguments)
    if name.startswith("mcp_tool_"):
        return AWMAction(
            kind="tool",
            name=name[len("mcp_tool_") :],
            arguments=arguments,
        )
    return AWMAction(kind="invalid", error=f"unknown meta-tool: {name}")


def parse_action(text: str | None) -> AWMAction:
    """Parse one AWM XML action or an ordinary communicative message."""
    if text is None or not str(text).strip():
        return AWMAction(kind="invalid", error="empty action")
    raw = str(text).strip()
    matches = _TOOL_CALL_RE.findall(raw)
    if len(matches) > 1:
        return AWMAction(kind="invalid", error="multiple tool calls")
    if matches:
        remainder = _THINK_RE.sub("", _TOOL_CALL_RE.sub("", raw)).strip()
        if remainder:
            return AWMAction(kind="invalid", error="tool call mixed with communicative message")
        try:
            return _parse_tool_payload(json.loads(matches[0]))
        except Exception as exc:
            return AWMAction(kind="invalid", error=f"invalid tool call: {exc}")
    if "<tool_call>" in raw or "</tool_call>" in raw:
        return AWMAction(kind="invalid", error="unclosed tool-call tag")

    candidate = _THINK_RE.sub("", raw).strip()
    if "<think>" in candidate or "</think>" in candidate:
        return AWMAction(kind="invalid", error="unclosed reasoning tag")
    if not candidate:
        return AWMAction(kind="invalid", error="empty action after reasoning")
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return AWMAction(kind="message", content=candidate)
    if not isinstance(payload, dict) or not ({"name", "function"} & payload.keys()):
        return AWMAction(kind="message", content=candidate)
    return AWMAction(kind="invalid", error="tool call missing <tool_call> wrapper")


def _tool_fields(tool: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(tool, Mapping):
        name = tool.get("name") or tool.get("function", {}).get("name")
        description = tool.get("description", "")
        schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters") or tool.get("function", {}).get("parameters") or {}
    else:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", "")
        schema = getattr(tool, "input_schema", None) or {}
    if not isinstance(name, str) or not name:
        raise ValueError("AWM tool is missing a name")
    if not isinstance(schema, dict):
        raise ValueError(f"AWM tool {name!r} has a non-object schema")
    return name, schema, str(description or "")


def normalize_tools(tools: Iterable[Any]) -> list[dict[str, Any]]:
    normalized = []
    for tool in tools:
        name, schema, description = _tool_fields(tool)
        normalized.append(
            {
                "name": name,
                "description": description,
                "inputSchema": schema,
            }
        )
    return normalized


@lru_cache(maxsize=1)
def _timezone_names() -> dict[str, str]:
    return {name.casefold(): name for name in available_timezones()}


def _normalize_temporal_string(
    value: str,
    *,
    schema_format: str | None,
    field_name: str | None,
) -> str:
    stripped = value.strip()
    try:
        if schema_format == "date":
            return date.fromisoformat(stripped).isoformat()
        if schema_format == "date-time":
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc)
                return parsed.isoformat().replace("+00:00", "Z")
            return parsed.isoformat()
        if schema_format == "time":
            return time.fromisoformat(stripped.replace("Z", "+00:00")).isoformat()
    except ValueError:
        # Let jsonschema's format checker emit the controlled validation error.
        return stripped

    normalized_name = (field_name or "").casefold().replace("-", "_")
    if normalized_name in {"timezone", "time_zone", "tz"}:
        if stripped.casefold() in {"utc", "z", "gmt", "etc/utc"}:
            return "UTC"
        return _timezone_names().get(stripped.casefold(), stripped)
    return stripped


def _coerce_scalar(
    value: Any,
    schema_type: str | None,
    *,
    schema: Mapping[str, Any],
    field_name: str | None,
) -> Any:
    if schema_type == "integer" and isinstance(value, str):
        return int(value)
    if schema_type == "number" and isinstance(value, str):
        return float(value)
    if schema_type == "boolean" and isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    if schema_type == "string" and value is not None:
        value = value if isinstance(value, str) else str(value)
        value = _normalize_temporal_string(
            value,
            schema_format=(str(schema["format"]) if schema.get("format") else None),
            field_name=field_name,
        )
        enum = schema.get("enum") or []
        for choice in enum:
            if isinstance(choice, str) and choice.casefold() == value.casefold():
                return choice
    return value


def _resolve_local_ref(
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    resolved: Any = root_schema
    for component in reference[2:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, Mapping) or key not in resolved:
            return schema
        resolved = resolved[key]
    if not isinstance(resolved, Mapping):
        return schema
    return {**resolved, **{key: item for key, item in schema.items() if key != "$ref"}}


def _coerce_to_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any] | None = None,
    field_name: str | None = None,
) -> Any:
    root_schema = root_schema or schema
    schema = _resolve_local_ref(schema, root_schema)

    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        siblings = {key: item for key, item in schema.items() if key != keyword}
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            combined = {**siblings, **variant}
            candidate = _coerce_to_schema(
                value,
                combined,
                root_schema=root_schema,
                field_name=field_name,
            )
            if Draft202012Validator(combined, format_checker=FormatChecker()).is_valid(candidate):
                return candidate

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = non_null[0] if len(non_null) == 1 else None
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties") or {}
        additional = schema.get("additionalProperties", {})
        return {
            str(key): _coerce_to_schema(
                item,
                properties.get(key, additional if isinstance(additional, Mapping) else {}),
                root_schema=root_schema,
                field_name=str(key),
            )
            for key, item in value.items()
        }
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items") or {}
        return [
            _coerce_to_schema(
                item,
                item_schema,
                root_schema=root_schema,
                field_name=field_name,
            )
            for item in value
        ]
    return _coerce_scalar(
        value,
        schema_type if isinstance(schema_type, str) else None,
        schema=schema,
        field_name=field_name,
    )


def validate_action(action: AWMAction, tools: Iterable[Any]) -> AWMAction:
    """Validate and schema-normalize one AWM action without executing it."""
    if action.kind in {"invalid", "meta_list_tools"}:
        return action
    if action.kind == "message":
        content = (action.content or "").strip()
        if not content:
            return AWMAction(kind="invalid", error="empty message")
        return AWMAction(kind="message", content=content)
    by_name = {item["name"]: item for item in normalize_tools(tools)}
    tool = by_name.get(action.name or "")
    if tool is None:
        return AWMAction(kind="invalid", error=f"unknown tool: {action.name}")
    schema = tool["inputSchema"] or {"type": "object"}
    try:
        arguments = _coerce_to_schema(action.arguments or {}, schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(arguments)
    except Exception as exc:
        return AWMAction(kind="invalid", error=f"invalid tool arguments: {exc}")
    return AWMAction(kind="tool", name=action.name, arguments=arguments)


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_action(action: AWMAction | Mapping[str, Any]) -> str:
    if isinstance(action, Mapping):
        action = AWMAction(**dict(action))
    if action.kind == "tool":
        payload = {
            "kind": "tool",
            "name": action.name,
            "arguments": normalize_json(action.arguments or {}),
        }
    elif action.kind == "message":
        payload = {"kind": "message", "content": normalize_message(action.content or "")}
    elif action.kind == "meta_list_tools":
        payload = {"kind": "meta_list_tools"}
    else:
        payload = {"kind": "invalid", "error": action.error or "invalid"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def action_multiset_frequencies(actions: Iterable[AWMAction]) -> Counter[str]:
    return Counter(canonical_action(action) for action in actions if action.kind == "tool")


def score_candidates(
    candidates: Sequence[AWMAction],
    teacher_actions: Sequence[AWMAction],
    *,
    message_match_counts: Mapping[int, int] | None = None,
) -> list[ScoredCandidate]:
    """Score candidates against an ordered teacher multiset without deduplication."""
    tool_counts = action_multiset_frequencies(teacher_actions)
    message_match_counts = message_match_counts or {}
    output = []
    for index, action in enumerate(candidates):
        if action.kind == "meta_list_tools":
            output.append(ScoredCandidate(action, None, -1.0, False, 0))
            continue
        if action.kind == "invalid":
            output.append(ScoredCandidate(action, -1.0, -1.0, True, 0))
            continue
        if action.kind == "tool":
            frequency = int(tool_counts.get(canonical_action(action), 0))
        else:
            frequency = int(message_match_counts.get(index, 0))
        reward = float(frequency) if frequency > 0 else 0.0
        output.append(ScoredCandidate(action, reward, reward, True, frequency))
    return output


def state_fingerprint(
    scenario: str,
    task_idx: int,
    chat: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "protocol_version": _PROTOCOL_VERSION,
        "scenario": scenario,
        "task_idx": int(task_idx),
        "chat": list(chat),
        "tools": list(tools),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def tool_schema_hash(tools: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(tools), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _format_schema(schema: Mapping[str, Any], indent: int = 6) -> str:
    encoded = json.dumps(
        normalize_json(schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{' ' * indent}{encoded}"


def format_tools_for_response(tools: Iterable[Any]) -> str:
    """Render the native AWM list_tools response used by the model scaffold."""
    normalized = normalize_tools(tools)
    lines = [f"Available MCP Tools ({len(normalized)} tools):", "=" * 80, ""]
    for index, tool in enumerate(normalized, 1):
        name = tool["name"]
        display_name = name if name.startswith("mcp_tool_") else f"mcp_tool_{name}"
        description_lines = (tool.get("description") or "No description").splitlines()
        lines.append(f"{index}. {display_name}")
        lines.append(f"   Description: {description_lines[0].strip()}")
        lines.extend(f"   {line.strip()}" for line in description_lines[1:] if line.strip())
        schema = tool.get("inputSchema") or {}
        if schema:
            lines.append("   Input JSON Schema:")
            lines.append(_format_schema(schema))
        else:
            lines.append("   Input JSON Schema: {}")
        lines.append("")
    return "\n".join(lines).strip()


def scaffold_system_prompt() -> str:
    return """# MCP Tools

You are in an MCP environment. Use the available environment tools to complete the user's task. At each decision, produce exactly one environment action: either one call_tool function call or one ordinary message to the user. You are already logged in, and your user id is 1 if required.

The scaffold has already called list_tools exactly once. The complete tool documentation is present in the conversation. Do not call list_tools again.

To call an environment tool, return one JSON object inside <tool_call></tool_call>:
<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "mcp_tool_<name>", "arguments": "<JSON object>"}}
</tool_call>

At the final step, directly output the answer or summary without a tool call."""


def build_scaffold_chat(task: str, tools: Iterable[Any]) -> list[dict[str, Any]]:
    response = format_tools_for_response(tools)
    return [
        {"role": "system", "content": scaffold_system_prompt()},
        {"role": "user", "content": str(task)},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name":"list_tools","arguments":null}\n</tool_call>',
        },
        {"role": "user", "content": f"Tool response:\n{response}"},
    ]


def append_exchange(
    chat: Sequence[Mapping[str, Any]],
    *,
    assistant_content: str,
    tool_response: str | None,
    history_window: int,
) -> list[dict[str, Any]]:
    """Append one exchange while pinning the four-message scaffold prefix."""
    if len(chat) < 4:
        raise ValueError("AWM scaffold chat must contain four pinned messages")
    pinned = [dict(item) for item in chat[:4]]
    tail = [dict(item) for item in chat[4:]]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in tail:
        if message.get("role") == "assistant" and current:
            chunks.append(current)
            current = []
        current.append(message)
    if current:
        chunks.append(current)
    new_chunk = [{"role": "assistant", "content": assistant_content}]
    if tool_response is not None:
        new_chunk.append({"role": "user", "content": f"Tool response:\n{tool_response}"})
    chunks.append(new_chunk)
    if history_window < 0:
        raise ValueError("history_window must be non-negative")
    chunks = chunks[-history_window:] if history_window else []
    return [*pinned, *(message for chunk in chunks for message in chunk)]
