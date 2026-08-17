"""AWM action parsing, schema validation, and native tool-call history."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from zoneinfo import available_timezones

_TOOL_CALL_OPEN = r"(?:<tool_call>|<｜｜DSML｜｜tool_call>)"
_TOOL_CALL_CLOSE = r"(?:(?:</｜｜DSML｜｜>\s*)?</tool_call>|</｜｜DSML｜｜tool_call>)"
_TOOL_CALL_RE = re.compile(
    rf"{_TOOL_CALL_OPEN}\s*(.*?)\s*{_TOOL_CALL_CLOSE}",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROTOCOL_VERSION = 9
DEFAULT_FREQUENCY_BONUS_SCALE = 0.5


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

    return AWMAction(kind="tool", name=name, arguments=arguments)


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
    if "<tool_call>" in raw or "</tool_call>" in raw or "<｜｜DSML｜｜tool_call>" in raw or "</｜｜DSML｜｜tool_call>" in raw:
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


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise ValueError("tool call must be an object")


def parse_native_action(
    content: str | None,
    tool_calls: Sequence[Any] | None,
    *,
    take_first: bool,
) -> tuple[AWMAction, int]:
    """Parse provider-native calls, optionally truncating expert parallel calls."""
    calls = list(tool_calls or [])
    if not calls:
        return parse_action(content), 0
    if len(calls) > 1 and not take_first:
        return AWMAction(kind="invalid", error="multiple tool calls"), 0
    if _THINK_RE.sub("", str(content or "")).strip() and not take_first:
        return AWMAction(kind="invalid", error="tool call mixed with communicative message"), 0
    try:
        call = _mapping(calls[0])
        action = _parse_tool_payload(call)
    except Exception as exc:
        action = AWMAction(kind="invalid", error=f"invalid tool call: {exc}")
    return action, max(0, len(calls) - 1)


def _tool_fields(tool: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(tool, Mapping):
        name = tool.get("name") or tool.get("function", {}).get("name")
        description = tool.get("description") or tool.get("function", {}).get("description", "")
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


def _json_pointer_part(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _canonicalize_schema_node(
    value: Any,
    *,
    path: str,
    repairs: list[dict[str, Any]],
) -> Any:
    if isinstance(value, list):
        return [
            _canonicalize_schema_node(
                item,
                path=f"{path}/{index}",
                repairs=repairs,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value

    output = {
        str(key): _canonicalize_schema_node(
            item,
            path=f"{path}/{_json_pointer_part(key)}",
            repairs=repairs,
        )
        for key, item in value.items()
    }
    required = output.get("required")
    properties = output.get("properties")
    if isinstance(properties, Mapping) and isinstance(required, list) and all(isinstance(item, str) for item in required):
        unique_required = list(dict.fromkeys(required))
        if len(unique_required) != len(required):
            removed_duplicates = [item for index, item in enumerate(required) if item in required[:index]]
            output["required"] = unique_required
            repairs.append(
                {
                    "json_pointer": f"{path}/required" if path else "/required",
                    "repair": "deduplicate_required_fields",
                    "removed_duplicates": removed_duplicates,
                }
            )
    variants = output.get("anyOf")
    sibling_type = output.get("type")
    if isinstance(variants, list) and isinstance(sibling_type, str):
        branch_types = {branch.get("type") for branch in variants if isinstance(branch, Mapping) and isinstance(branch.get("type"), str)}
        if "null" in branch_types and sibling_type in branch_types:
            output.pop("type")
            repairs.append(
                {
                    "json_pointer": path or "/",
                    "repair": "remove_redundant_nullable_sibling_type",
                    "removed_type": sibling_type,
                }
            )
    return output


def canonicalize_tool_schema(
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply lossless repairs to malformed schemas emitted by pinned AWM."""
    repairs: list[dict[str, Any]] = []
    canonical = _canonicalize_schema_node(schema, path="", repairs=repairs)
    if not isinstance(canonical, dict):
        raise ValueError("tool schema must be an object")
    return canonical, repairs


def normalize_tools(
    tools: Iterable[Any],
    *,
    canonicalize_schemas: bool = True,
) -> list[dict[str, Any]]:
    normalized = []
    for tool in tools:
        name, schema, description = _tool_fields(tool)
        if canonicalize_schemas:
            schema, _ = canonicalize_tool_schema(schema)
        normalized.append(
            {
                "name": name,
                "description": description,
                "inputSchema": schema,
            }
        )
    return normalized


def tool_schema_audit(tools: Iterable[Any]) -> dict[str, Any]:
    """Return immutable raw/canonical schemas, hashes, and exact repairs."""
    raw_tools = normalize_tools(tools, canonicalize_schemas=False)
    canonical_tools = []
    repairs = []
    for tool in raw_tools:
        schema, schema_repairs = canonicalize_tool_schema(tool["inputSchema"])
        canonical_tools.append({**tool, "inputSchema": schema})
        repairs.extend({"tool_name": tool["name"], **repair} for repair in schema_repairs)
    return {
        "raw_tools": raw_tools,
        "canonical_tools": canonical_tools,
        "raw_tool_schema_hash": tool_schema_hash(raw_tools),
        "canonical_tool_schema_hash": tool_schema_hash(canonical_tools),
        "schema_repairs": repairs,
    }


_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def openai_tools(tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert canonical AWM MCP schemas to provider-native function tools."""
    output = []
    names: set[str] = set()
    for tool in normalize_tools(tools):
        name = tool["name"]
        if not _FUNCTION_NAME_RE.fullmatch(name):
            raise ValueError(f"AWM tool name is not provider-compatible: {name!r}")
        if name in names:
            raise ValueError(f"duplicate AWM tool name: {name!r}")
        names.add(name)
        output.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
        )
    return output


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


def _schema_declares_nullable(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    return any(isinstance(variant, Mapping) and _schema_declares_nullable(variant) for keyword in ("oneOf", "anyOf") for variant in (schema.get(keyword) or []))


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
        required = set(schema.get("required") or [])
        output = {}
        for key, item in value.items():
            field_schema = properties.get(
                key,
                additional if isinstance(additional, Mapping) else {},
            )
            nullable_schema = _resolve_local_ref(field_schema, root_schema) if isinstance(field_schema, Mapping) else {}
            candidate = _coerce_to_schema(
                item,
                field_schema,
                root_schema=root_schema,
                field_name=str(key),
            )
            if candidate is None and key not in required and isinstance(field_schema, Mapping) and _schema_declares_nullable(nullable_schema):
                continue
            output[str(key)] = candidate
        return output
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
    if action.kind == "invalid":
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
    else:
        payload = {"kind": "invalid", "error": action.error or "invalid"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def action_multiset_frequencies(actions: Iterable[AWMAction]) -> Counter[str]:
    return Counter(canonical_action(action) for action in actions if action.kind == "tool")


def semantic_match_reward(
    frequency: int,
    *,
    teacher_sample_count: int,
    frequency_bonus_scale: float = DEFAULT_FREQUENCY_BONUS_SCALE,
) -> float:
    """Map teacher-match frequency to a bounded soft-consensus reward.

    A matched action always receives a base reward of one. The configurable
    bonus reaches frequency_bonus_scale only when all teacher samples agree:

        1 + frequency_bonus_scale * (frequency - 1) / (K - 1)

    With K=3, scale 0.0 is any-match, 0.5 maps frequencies to
    1.0/1.25/1.5, and 2.0 recovers the legacy raw-count reward 1/2/3.
    """
    if isinstance(frequency, bool) or int(frequency) != frequency or frequency < 0:
        raise ValueError("teacher frequency must be a non-negative integer")
    if isinstance(teacher_sample_count, bool) or int(teacher_sample_count) != teacher_sample_count or teacher_sample_count < 0:
        raise ValueError("teacher sample count must be a non-negative integer")
    scale = float(frequency_bonus_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("frequency bonus scale must be finite and non-negative")
    frequency = int(frequency)
    teacher_sample_count = int(teacher_sample_count)
    if frequency > teacher_sample_count:
        raise ValueError("teacher frequency cannot exceed teacher sample count")
    if frequency == 0:
        return 0.0
    if teacher_sample_count <= 1:
        return 1.0
    return 1.0 + scale * (frequency - 1) / (teacher_sample_count - 1)


def score_candidates(
    candidates: Sequence[AWMAction],
    teacher_actions: Sequence[AWMAction],
    *,
    message_match_counts: Mapping[int, int] | None = None,
    teacher_sample_count: int | None = None,
    frequency_bonus_scale: float = DEFAULT_FREQUENCY_BONUS_SCALE,
) -> list[ScoredCandidate]:
    """Score candidates against an ordered teacher multiset without deduplication."""
    tool_counts = action_multiset_frequencies(teacher_actions)
    message_match_counts = message_match_counts or {}
    if teacher_sample_count is None:
        teacher_sample_count = len(teacher_actions)
    # Validate the knob even for an all-invalid candidate group.
    semantic_match_reward(
        0,
        teacher_sample_count=teacher_sample_count,
        frequency_bonus_scale=frequency_bonus_scale,
    )
    output = []
    for index, action in enumerate(candidates):
        if action.kind == "invalid":
            output.append(ScoredCandidate(action, -1.0, -1.0, True, 0))
            continue
        if action.kind == "tool":
            frequency = int(tool_counts.get(canonical_action(action), 0))
        else:
            frequency = int(message_match_counts.get(index, 0))
        reward = semantic_match_reward(
            frequency,
            teacher_sample_count=teacher_sample_count,
            frequency_bonus_scale=frequency_bonus_scale,
        )
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


def native_system_prompt() -> str:
    return """You are operating in an interactive environment. Use the available functions to complete the user's task. The functions are supplied through the model's native tool-calling interface.

At each decision, take exactly one action: either call exactly one available function or send one ordinary assistant message. Never combine a function call with a message, and never call multiple functions in one decision. You are already logged in, and your user id is 1 if required.

At the final step, directly output the answer or summary without a function call."""


def build_native_chat(task: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": native_system_prompt()},
        {"role": "user", "content": str(task)},
    ]


def append_exchange(
    chat: Sequence[Mapping[str, Any]],
    *,
    action: AWMAction,
    raw_action: str,
    tool_response: str | None,
    tool_call_id: str | None = None,
    assistant_content: str | None = None,
    assistant_reasoning_content: str | None = None,
) -> list[dict[str, Any]]:
    """Append one structured native exchange while pinning system and task."""
    if len(chat) < 2:
        raise ValueError("AWM native chat must contain system and task messages")
    pinned = [dict(item) for item in chat[:2]]
    tail = [dict(item) for item in chat[2:]]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in tail:
        if message.get("role") == "assistant" and current:
            chunks.append(current)
            current = []
        current.append(message)
    if current:
        chunks.append(current)

    if action.kind == "tool":
        call_id = tool_call_id or f"call_{len(tail)}"
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(action.arguments or {}, ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        }
        if assistant_reasoning_content is not None:
            assistant["reasoning_content"] = assistant_reasoning_content
        new_chunk = [assistant]
        if tool_response is None:
            raise ValueError("tool action requires a tool response")
        new_chunk.append({"role": "tool", "tool_call_id": call_id, "content": tool_response})
    elif action.kind == "message":
        assistant = {
            "role": "assistant",
            "content": action.content or assistant_content or raw_action,
        }
        if assistant_reasoning_content is not None:
            assistant["reasoning_content"] = assistant_reasoning_content
        new_chunk = [assistant]
    else:
        assistant = {"role": "assistant", "content": _THINK_RE.sub("", raw_action).strip()}
        if assistant_reasoning_content is not None:
            assistant["reasoning_content"] = assistant_reasoning_content
        new_chunk = [assistant]
        if tool_response is not None:
            new_chunk.append({"role": "user", "content": f"Environment response:\n{tool_response}"})
    # Keep the logical trajectory losslessly. Context-budget trimming is a
    # rendering concern: mutating chat here used to make dropped exchanges
    # disappear from the environment state and future state fingerprints.
    chunks.append(new_chunk)
    return [*pinned, *(message for chunk in chunks for message in chunk)]
