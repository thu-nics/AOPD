#!/usr/bin/env python3
"""Standalone AWM-native evaluation with one persistent OpenAI/vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .actions import (
    append_exchange,
    build_scaffold_chat,
    canonical_action,
    format_tools_for_response,
    normalize_tools,
    parse_action,
    validate_action,
)
from .logical_time import fetch_server_protocol
from .native_rollout import response_is_error, summarize_results

EVAL_PROTOCOL_VERSION = 5
EXPECTED_DATASET_REVISION = "dde80a0283fe781bdc51656bce57063dc5650213"
EXPECTED_SOURCE_SHA256 = {
    "gen_db.jsonl": "ae8acb3c23765ca4866b35799ffb980fbb15831240fdc35c046e8a7d27a2c0e8",
    "gen_envs.jsonl": "2c7749c1710303f0f663bbe14aea689ade77c19282e2dd0ef0c54e2a95f5e7d8",
    "gen_sample.jsonl": "39c40969ad76d52a3ea51384752a639cf55dce15bc1a8f0f022cfb6bbd25db3c",
    "gen_scenario.jsonl": "6362e31af6e39bc914c6606b32f62b3d5f571f9653e86a6a9c29e507ae0e647e",
    "gen_tasks.jsonl": "0537871c8824cd56d23cc51118294ae3c3070d990cb9a3f766f9dc690f91bf45",
    "gen_verifier.jsonl": "269a97a085dce103afed6e5c48d001b8d47a9e7e2b142f2edc0ddaf6da08a348",
    "gen_verifier.pure_code.jsonl": "2de0b668bd0c6b37a033dda7d697bcd8ddc3bf5b0c9bf83fbd691fbd2c3827f7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_identity(reference: str) -> dict[str, Any]:
    path = Path(reference).expanduser()
    if not path.exists():
        return {"reference": reference, "local": False}
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "reference": reference,
            "local": True,
            "resolved_path": str(resolved),
            "sha256": _sha256(resolved),
        }

    records = []
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(resolved))
        stat = item.stat()
        record = {
            "path": relative,
            "size": stat.st_size,
            "sha256": _sha256(item),
        }
        records.append(record)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "reference": reference,
        "local": True,
        "resolved_path": str(resolved),
        "artifact_manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "files": len(records),
    }


def _observation_dict(result: Any) -> dict[str, Any]:
    observation = getattr(result, "observation", result)
    if hasattr(observation, "model_dump"):
        return observation.model_dump(mode="json")
    return dict(observation) if isinstance(observation, dict) else {"value": str(observation)}


def _tool_response(result: Any) -> str:
    payload = _observation_dict(result)
    value = payload.get("tool_result")
    if value is None:
        value = {"error": payload["error"]} if payload.get("error") else payload
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _fit_context(tokenizer, chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(chat) < 4:
        raise ValueError("AWM evaluation chat is missing its scaffold prefix")
    pinned = [dict(message) for message in chat[:4]]
    tail = [dict(message) for message in chat[4:]]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in tail:
        if message.get("role") == "assistant" and current:
            chunks.append(current)
            current = []
        current.append(message)
    if current:
        chunks.append(current)
    chunks = chunks[-3:]

    def length(messages):
        return len(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True,
            )
        )

    while len(chunks) > 1:
        candidate = [*pinned, *(message for chunk in chunks for message in chunk)]
        if length(candidate) <= 29952:
            return candidate
        chunks.pop(0)
    candidate = [*pinned, *(message for chunk in chunks for message in chunk)]
    if length(candidate) > 29952:
        raise RuntimeError("AWM pinned scaffold and newest exchange exceed the 29,952-token prompt budget")
    return candidate


async def _evaluate_one(
    row: dict[str, Any],
    *,
    client: AsyncOpenAI,
    model: str,
    tokenizer,
    awm_base_url: str,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    async with semaphore:
        trajectory = []
        async with AWMEnv(base_url=awm_base_url) as env:
            reset = await env.reset(
                scenario=str(row["scenario"]),
                task_idx=int(row["task_idx"]),
                seed=int(seed),
            )
            reset_payload = _observation_dict(reset)
            if reset_payload.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"AWM reset failed for {row['task_id']}: {reset_payload}")
            tools = normalize_tools(await env.list_tools(use_cache=False))
            chat = build_scaffold_chat(str(reset_payload.get("task") or row["task"]), tools)
            final_answer = None
            terminal_reason = "decision_limit"
            for decision in range(1, 21):
                visible_chat = _fit_context(tokenizer, chat)
                response = await client.chat.completions.create(
                    model=model,
                    messages=visible_chat,
                    max_tokens=2048,
                    temperature=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                message = response.choices[0].message
                raw_action = message.content or ""
                action = validate_action(parse_action(raw_action), tools)
                entry = {
                    "decision": decision,
                    "raw_action": raw_action,
                    "reasoning_content": getattr(message, "reasoning_content", "") or "",
                    "parsed_action": canonical_action(action),
                    "action_kind": action.kind,
                    "parse_error": action.error,
                    "model": response.model,
                    "system_fingerprint": response.system_fingerprint,
                    "usage": response.usage.model_dump() if response.usage is not None else {},
                }
                if action.kind == "tool":
                    step = await env.step(
                        CallToolAction(
                            tool_name=action.name or "",
                            arguments=action.arguments or {},
                        )
                    )
                    tool_text = _tool_response(step)
                    entry["tool_response"] = tool_text
                    entry["tool_response_is_error"] = response_is_error(tool_text)
                    chat = append_exchange(
                        chat,
                        assistant_content=raw_action,
                        tool_response=tool_text,
                        history_window=3,
                    )
                elif action.kind == "meta_list_tools":
                    repeated = await env.list_tools(use_cache=False)
                    tool_text = format_tools_for_response(repeated)
                    entry["tool_response"] = tool_text
                    entry["tool_response_is_error"] = False
                    chat = append_exchange(
                        chat,
                        assistant_content=raw_action,
                        tool_response=tool_text,
                        history_window=3,
                    )
                elif action.kind == "message":
                    final_answer = action.content or ""
                    terminal_reason = "final_response"
                    chat = append_exchange(
                        chat,
                        assistant_content=raw_action,
                        tool_response=None,
                        history_window=3,
                    )
                    trajectory.append(entry)
                    break
                else:
                    error_text = json.dumps({"error": action.error or "invalid action"}, ensure_ascii=False)
                    entry["tool_response"] = error_text
                    entry["tool_response_is_error"] = True
                    chat = append_exchange(
                        chat,
                        assistant_content=raw_action,
                        tool_response=error_text,
                        history_window=3,
                    )
                trajectory.append(entry)

            verify = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={
                        "verifier_mode": "code",
                        "final_answer": final_answer,
                    },
                )
            )
            verify_payload = _observation_dict(verify)
            await env.step(CallToolAction(tool_name="done", arguments={}))
        return {
            "task_id": row["task_id"],
            "scenario": row["scenario"],
            "task_idx": int(row["task_idx"]),
            "task": row["task"],
            "seed": int(seed),
            "success": verify_payload.get("reward_type") == "complete",
            "reward": float(getattr(verify, "reward", 0.0) or 0.0),
            "reward_type": verify_payload.get("reward_type"),
            "verify_result": verify_payload.get("verify_result"),
            "terminal_reason": terminal_reason,
            "decisions": len(trajectory),
            "trajectory": trajectory,
        }


async def _run(args) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError("AWM manifest dataset revision does not match the eval protocol")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("AWM manifest source hashes do not match the eval protocol")
    selection_sha256 = None
    if args.selection_manifest is not None:
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        split_ids = list(selection.get("task_ids") or [])
        selection_sha256 = _sha256(args.selection_manifest)
    else:
        split_ids = list((manifest.get("split_task_ids") or {}).get(args.split) or [])
    if not split_ids:
        raise RuntimeError("AWM evaluation selection contains no task IDs")
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError("AWM evaluation selection contains duplicate task IDs")
    frame = pd.read_parquet(args.data)
    rows_by_id = {
        str(row["extra_info"]["task_id"]): {
            "task_id": str(row["extra_info"]["task_id"]),
            "scenario": str(row["env_kwargs"]["scenario"]),
            "task_idx": int(row["env_kwargs"]["task_idx"]),
            "task": str(row["extra_info"]["task"]),
        }
        for _, row in frame.iterrows()
    }
    missing = [task_id for task_id in split_ids if task_id not in rows_by_id]
    if missing:
        raise RuntimeError(f"evaluation parquet is missing {len(missing)} manifest task(s)")
    if args.limit is not None:
        split_ids = split_ids[: args.limit]

    logical_time_protocol = fetch_server_protocol(args.awm_base_url)
    identity = {
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "manifest_sha256": _sha256(args.manifest),
        "selection_manifest_sha256": selection_sha256,
        "split": args.split,
        "task_ids": split_ids,
        "seed": int(args.seed),
        "model": args.model,
        "checkpoint": _model_artifact_identity(args.tokenizer or args.model),
        "api_base": args.api_base,
        "awm_base_url": args.awm_base_url,
        "awm_logical_time": logical_time_protocol,
        "max_model_len": 32000,
        "max_prompt_tokens": 29952,
        "max_response_tokens": 2048,
        "decoding": {
            "temperature": 0,
            "thinking": True,
        },
        "history_window": 3,
        "max_decisions": 20,
        "verifier_mode": "code",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    results_path = args.output_dir / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume:
        if not config_path.is_file():
            raise RuntimeError("--resume requires an existing config.json")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError("AWM eval resume configuration mismatch")
        if results_path.is_file():
            with results_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    completed[str(record["task_id"])] = record
    elif config_path.exists() or results_path.exists():
        raise FileExistsError(f"refusing to overwrite existing eval artifacts in {args.output_dir}")
    config_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)
    semaphore = asyncio.Semaphore(args.concurrency)
    pending_ids = [task_id for task_id in split_ids if task_id not in completed]
    pending_tasks = [
        _evaluate_one(
            rows_by_id[task_id],
            client=client,
            model=args.model,
            tokenizer=tokenizer,
            awm_base_url=args.awm_base_url,
            seed=args.seed,
            semaphore=semaphore,
        )
        for task_id in pending_ids
    ]
    with results_path.open("a", encoding="utf-8") as handle:
        for future in asyncio.as_completed(pending_tasks):
            result = await future
            completed[result["task_id"]] = result
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    with results_path.open("w", encoding="utf-8") as handle:
        for task_id in split_ids:
            handle.write(json.dumps(completed[task_id], ensure_ascii=False) + "\n")
    ordered = [completed[task_id] for task_id in split_ids]
    summary = summarize_results(ordered)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--split", choices=("all", "dev", "smoke"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--awm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
