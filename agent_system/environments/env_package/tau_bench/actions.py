"""Action parsing and canonicalization for Tau Bench training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    name: str | None = None
    arguments: dict[str, Any] | None = None
    content: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_payload(value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("tool call must be a JSON object")
    if "function" in value:
        value = value["function"]
    name = value.get("name")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be a JSON object")
    return name.strip(), arguments


def parse_action(text: str | None) -> ParsedAction:
    """Parse one Qwen/OpenAI-style tool call or one user-facing message."""
    if text is None or not str(text).strip():
        return ParsedAction(kind="invalid", error="empty action")
    raw = str(text).strip()
    matches = _TOOL_CALL_RE.findall(raw)
    if len(matches) > 1:
        return ParsedAction(kind="invalid", error="multiple tool calls")
    if len(matches) == 1:
        remainder = _THINK_RE.sub("", _TOOL_CALL_RE.sub("", raw)).strip()
        if remainder:
            return ParsedAction(kind="invalid", error="tool call mixed with user-facing message")
        try:
            name, arguments = _tool_payload(json.loads(matches[0]))
        except Exception as exc:
            return ParsedAction(kind="invalid", error=f"invalid tool call: {exc}")
        return ParsedAction(kind="tool", name=name, arguments=arguments)
    if "<tool_call>" in raw or "</tool_call>" in raw:
        return ParsedAction(kind="invalid", error="unclosed tool-call tag")

    candidate = _THINK_RE.sub("", raw).strip()
    if "<think>" in candidate or "</think>" in candidate:
        return ParsedAction(kind="invalid", error="unclosed reasoning tag")
    try:
        name, arguments = _tool_payload(json.loads(candidate))
    except Exception:
        if not candidate:
            return ParsedAction(kind="invalid", error="empty action after reasoning")
        return ParsedAction(kind="message", content=candidate)
    return ParsedAction(kind="tool", name=name, arguments=arguments)


def validate_tau_action(action: ParsedAction, tools: Iterable[Any]) -> ParsedAction:
    """Validate one parsed action against the current Tau tool contract."""
    if action.kind == "invalid":
        return action
    if action.kind == "message":
        if not (action.content or "").strip():
            return ParsedAction(kind="invalid", error="empty message")
        return action
    by_name = {tool.name: tool for tool in tools}
    tool = by_name.get(action.name)
    if tool is None:
        return ParsedAction(kind="invalid", error=f"unknown tool: {action.name}")
    arguments = action.arguments or {}
    unknown = set(arguments) - set(tool.params.model_fields)
    if unknown:
        return ParsedAction(
            kind="invalid",
            error=f"unknown tool arguments: {sorted(unknown)}",
        )
    try:
        validated = tool.params.model_validate(arguments, strict=True)
    except Exception as exc:
        return ParsedAction(kind="invalid", error=f"invalid tool arguments: {exc}")
    return ParsedAction(
        kind="tool",
        name=action.name,
        arguments=validated.model_dump(mode="json"),
    )


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_action(action: ParsedAction | dict[str, Any]) -> str:
    if isinstance(action, dict):
        action = ParsedAction(**action)
    if action.kind == "tool":
        value = {
            "kind": "tool",
            "name": action.name,
            "arguments": normalize_json(action.arguments or {}),
        }
    elif action.kind == "message":
        value = {"kind": "message", "content": " ".join((action.content or "").split())}
    else:
        value = {"kind": "invalid", "error": action.error or "invalid"}
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def deduplicate_actions(actions: Iterable[ParsedAction | dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in actions:
        action = ParsedAction(**value) if isinstance(value, dict) else value
        if action.kind == "invalid":
            continue
        key = canonical_action(action)
        if key in seen:
            continue
        seen.add(key)
        output.append(action.to_dict())
    return output


def to_tau_action(action: ParsedAction | dict[str, Any]) -> str:
    if isinstance(action, dict):
        action = ParsedAction(**action)
    if action.kind == "tool":
        return json.dumps(
            {"name": action.name, "arguments": action.arguments or {}},
            ensure_ascii=False,
        )
    if action.kind == "message":
        return action.content or ""
    raise ValueError(f"cannot execute invalid action: {action.error}")


def tau_messages_to_openai(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert Tau messages while preserving valid assistant tool-call linkage."""
    output: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if role == "assistant" and tool_calls:
            calls = []
            for index, call in enumerate(tool_calls):
                call_id = getattr(call, "id", "") or f"call_{len(output)}_{index}"
                pending_tool_ids.append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                )
            output.append({"role": "assistant", "content": content or "", "tool_calls": calls})
        elif role == "user" and tool_calls:
            rendered = [
                {"name": call.name, "arguments": call.arguments}
                for call in tool_calls
            ]
            output.append(
                {"role": "user", "content": "User-side tool call: " + json.dumps(rendered, ensure_ascii=False)}
            )
        elif role == "tool" and getattr(message, "requestor", "assistant") == "assistant":
            call_id = getattr(message, "id", "")
            if not call_id and pending_tool_ids:
                call_id = pending_tool_ids.pop(0)
            elif call_id in pending_tool_ids:
                pending_tool_ids.remove(call_id)
            output.append(
                {
                    "role": "tool",
                    "content": content or "",
                    "tool_call_id": call_id or "unknown",
                }
            )
        elif role == "tool":
            output.append({"role": "user", "content": f"User-side tool result: {content or ''}"})
        elif role in {"system", "user", "assistant"}:
            output.append({"role": role, "content": content or ""})
    return output


def state_fingerprint(
    domain: str,
    task_id: str,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "domain": domain,
            "task_id": task_id,
            "history": history,
            "tools": tools,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
