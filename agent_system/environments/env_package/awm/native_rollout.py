"""Shared AWM-native rollout primitives for evaluation and qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .actions import (
    append_exchange,
    build_native_chat,
    canonical_action,
    openai_tools,
    parse_native_action,
    tool_schema_audit,
    tool_schema_hash,
    validate_action,
)

MODEL_CONTEXT_TOKENS = 32000
MAX_PROMPT_TOKENS = 29952
MAX_RESPONSE_TOKENS = 2048
HISTORY_WINDOW = 3
MAX_DECISIONS = 20

GenerateAction = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[Mapping[str, Any]]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_identity(reference: str) -> dict[str, Any]:
    path = Path(reference).expanduser()
    if not path.exists():
        return {"reference": reference, "local": False}
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "reference": reference,
            "local": True,
            "resolved_path": str(resolved),
            "sha256": sha256_file(resolved),
        }

    records = []
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        records.append(
            {
                "path": str(item.relative_to(resolved)),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "reference": reference,
        "local": True,
        "resolved_path": str(resolved),
        "artifact_manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "files": len(records),
    }


def observation_dict(result: Any) -> dict[str, Any]:
    observation = getattr(result, "observation", result)
    if hasattr(observation, "model_dump"):
        return observation.model_dump(mode="json")
    return dict(observation) if isinstance(observation, Mapping) else {"value": str(observation)}


def tool_response(result: Any) -> str:
    payload = observation_dict(result)
    value = payload.get("tool_result")
    if value is None:
        value = {"error": payload["error"]} if payload.get("error") else payload
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def fit_context(tokenizer, chat: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the fixed AWM 29,952-token prompt budget and w=3 history."""
    if len(chat) < 2:
        raise ValueError("AWM native rollout chat is missing system/task prefix")
    pinned = [dict(message) for message in chat[:2]]
    tail = [dict(message) for message in chat[2:]]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in tail:
        if message.get("role") == "assistant" and current:
            chunks.append(current)
            current = []
        current.append(message)
    if current:
        chunks.append(current)
    chunks = chunks[-HISTORY_WINDOW:]

    def length(messages):
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    while len(chunks) > 1:
        candidate = [*pinned, *(message for chunk in chunks for message in chunk)]
        if length(candidate) <= MAX_PROMPT_TOKENS:
            return candidate
        chunks.pop(0)
    candidate = [*pinned, *(message for chunk in chunks for message in chunk)]
    if length(candidate) > MAX_PROMPT_TOKENS:
        raise RuntimeError("AWM native tools, task, and newest exchange exceed the 29,952-token prompt budget")
    return candidate


def fixed_native_prompt_token_count(tokenizer, task: str, tools: list[dict[str, Any]]) -> int:
    chat = build_native_chat(task)
    rendered = tokenizer.apply_chat_template(
        chat,
        tools=openai_tools(tools),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def response_is_error(value: str) -> bool:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and bool(payload.get("error"))


async def run_native_trajectory(
    row: Mapping[str, Any],
    *,
    generate_action: GenerateAction,
    tokenizer,
    awm_base_url: str,
    seed: int,
    verifier_mode: str = "code",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    max_decisions: int = MAX_DECISIONS,
) -> dict[str, Any]:
    """Run one policy trajectory through AWM's native interface."""
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    if verifier_mode not in {"code", "sql"}:
        raise ValueError(f"unsupported AWM verifier mode: {verifier_mode}")
    reset_kwargs: dict[str, Any] = {
        "scenario": str(row["scenario"]),
        "task_idx": int(row["task_idx"]),
        "seed": int(seed),
    }
    if verifier_mode == "sql":
        if not all((judge_base_url, judge_api_key, judge_model)):
            raise ValueError("SQL verification requires judge base URL, API key, and model")
        reset_kwargs.update(
            llm_base_url=judge_base_url,
            llm_api_key=judge_api_key,
            llm_model=judge_model,
        )
    trajectory = []
    async with AWMEnv(base_url=awm_base_url) as env:
        reset = await env.reset(**reset_kwargs)
        reset_payload = observation_dict(reset)
        if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
            raise RuntimeError(f"AWM reset failed for {row['task_id']}: {reset_payload}")
        reset_task = str(reset_payload.get("task") or "")
        if reset_task != str(row["task"]):
            raise RuntimeError(f"AWM reset task changed for {row['task_id']}: {reset_task!r}")
        schema_audit = tool_schema_audit(await env.list_tools(use_cache=False))
        tools = schema_audit["canonical_tools"]
        native_tools = openai_tools(tools)
        actual_schema_hash = tool_schema_hash(tools)
        if row.get("tool_schema_hash") and actual_schema_hash != row["tool_schema_hash"]:
            raise RuntimeError(f"AWM tool schema changed for {row['task_id']}")
        actual_raw_schema_hash = schema_audit["raw_tool_schema_hash"]
        if row.get("raw_tool_schema_hash") and actual_raw_schema_hash != row["raw_tool_schema_hash"]:
            raise RuntimeError(f"AWM raw tool schema changed for {row['task_id']}")
        chat = build_native_chat(str(reset_payload.get("task") or row["task"]))
        final_answer = None
        terminal_reason = "decision_limit"
        for decision in range(1, int(max_decisions) + 1):
            visible_chat = fit_context(tokenizer, chat, native_tools)
            generated = dict(await generate_action(visible_chat, native_tools))
            raw_action = str(generated.get("content") or "")
            action, skipped_tool_calls = parse_native_action(
                raw_action,
                generated.get("tool_calls"),
                take_first=True,
            )
            action = validate_action(action, tools)
            entry = {
                "decision": decision,
                "raw_action": raw_action,
                "reasoning_content": str(generated.get("reasoning_content") or ""),
                "finish_reason": generated.get("finish_reason"),
                "raw_tool_calls": list(generated.get("tool_calls") or []),
                "native_tool_calls": len(generated.get("tool_calls") or []),
                "skipped_tool_calls": skipped_tool_calls,
                "parsed_action": canonical_action(action),
                "action_kind": action.kind,
                "parse_error": action.error,
                "model": generated.get("model"),
                "system_fingerprint": generated.get("system_fingerprint"),
                "usage": dict(generated.get("usage") or {}),
            }
            if action.kind == "tool":
                step = await env.step(
                    CallToolAction(
                        tool_name=action.name or "",
                        arguments=action.arguments or {},
                    )
                )
                tool_text = tool_response(step)
                entry["tool_response"] = tool_text
                entry["tool_response_is_error"] = response_is_error(tool_text)
                chat = append_exchange(
                    chat,
                    action=action,
                    raw_action=raw_action,
                    tool_response=tool_text,
                    history_window=HISTORY_WINDOW,
                    tool_call_id=generated.get("tool_call_id"),
                    assistant_content=generated.get("content"),
                )
            elif action.kind == "message":
                final_answer = action.content or ""
                terminal_reason = "final_response"
                chat = append_exchange(
                    chat,
                    action=action,
                    raw_action=raw_action,
                    tool_response=None,
                    history_window=HISTORY_WINDOW,
                )
                trajectory.append(entry)
                break
            else:
                error_text = json.dumps(
                    {"error": action.error or "invalid action"},
                    ensure_ascii=False,
                )
                entry["tool_response"] = error_text
                entry["tool_response_is_error"] = True
                chat = append_exchange(
                    chat,
                    action=action,
                    raw_action=raw_action,
                    tool_response=error_text,
                    history_window=HISTORY_WINDOW,
                )
            trajectory.append(entry)

        verify = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": verifier_mode, "final_answer": final_answer},
            )
        )
        verify_payload = observation_dict(verify)
        await env.step(CallToolAction(tool_name="done", arguments={}))
    return {
        "task_id": str(row["task_id"]),
        "scenario": str(row["scenario"]),
        "task_idx": int(row["task_idx"]),
        "task": str(row["task"]),
        "seed": int(seed),
        "reset_reward_type": reset_payload.get("reward_type"),
        "tool_schema_hash": actual_schema_hash,
        "raw_tool_schema_hash": schema_audit["raw_tool_schema_hash"],
        "schema_repairs": schema_audit["schema_repairs"],
        "success": verify_payload.get("reward_type") == "complete",
        "reward": float(getattr(verify, "reward", 0.0) or 0.0),
        "reward_type": verify_payload.get("reward_type"),
        "verify_result": verify_payload.get("verify_result"),
        "terminal_reason": terminal_reason,
        "decisions": len(trajectory),
        "trajectory": trajectory,
    }


def summarize_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    invalid_reasons: dict[str, int] = {}
    tool_errors = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    decisions = 0
    for result in results:
        for entry in result.get("trajectory") or []:
            decisions += 1
            kind = str(entry.get("action_kind") or "unknown")
            action_counts[kind] = action_counts.get(kind, 0) + 1
            if kind == "invalid":
                reason = str(entry.get("parse_error") or "unknown")
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            if kind == "tool" and entry.get("tool_response_is_error"):
                tool_errors += 1
            for key in usage:
                usage[key] += int((entry.get("usage") or {}).get(key, 0) or 0)
    successes = sum(int(bool(result.get("success"))) for result in results)
    tool_actions = action_counts.get("tool", 0)
    return {
        "tasks": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "mean_reward": (sum(float(result.get("reward", 0.0) or 0.0) for result in results) / len(results) if results else 0.0),
        "total_decisions": decisions,
        "mean_decisions": decisions / len(results) if results else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "valid_action_rate": ((decisions - action_counts.get("invalid", 0)) / decisions if decisions else 0.0),
        "schema_valid_tool_calls": tool_actions,
        "tool_execution_errors": tool_errors,
        "tool_execution_error_rate": tool_errors / tool_actions if tool_actions else 0.0,
        "usage": usage,
    }
