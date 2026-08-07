"""Shared candidate loading, ordering, and DeepSeek policy for AWM screening."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from openai import AsyncOpenAI

from ..data.integrity import INTEGRITY_PROTOCOL_VERSION, PREFILTER_PROTOCOL_VERSION
from ..data.selection import SELECTION_PROTOCOL_VERSION, stable_rank
from ..runtime.rollout import sha256_file


def load_candidate_rows(
    data_path: Path,
    manifest_path: Path,
    integrity_manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SELECTION_PROTOCOL_VERSION:
        raise RuntimeError("AWM candidate selection protocol mismatch")
    integrity_manifest = None
    expected_data_sha256 = manifest.get("candidate_data_sha256")
    expected_task_ids = manifest.get("task_ids")
    actual_data_sha256 = sha256_file(data_path)
    if integrity_manifest_path is not None:
        integrity_manifest = json.loads(integrity_manifest_path.read_text(encoding="utf-8"))
        if integrity_manifest.get("protocol_version") != INTEGRITY_PROTOCOL_VERSION:
            raise RuntimeError("AWM integrity filter protocol mismatch")
        if integrity_manifest.get("selection_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("AWM integrity filter selection-manifest mismatch")
        if integrity_manifest.get("prefilter_protocol_version") != PREFILTER_PROTOCOL_VERSION:
            raise RuntimeError("AWM deterministic prefilter protocol mismatch")
        if actual_data_sha256 == integrity_manifest.get("training_pool_data_sha256"):
            expected_data_sha256 = integrity_manifest.get("training_pool_data_sha256")
            expected_task_ids = integrity_manifest.get("training_pool_task_ids")
        else:
            raise RuntimeError("AWM candidate Parquet is not hash-bound by the integrity manifest")
    if actual_data_sha256 != expected_data_sha256:
        raise RuntimeError("AWM candidate parquet hash mismatch")

    rows = []
    for _, raw in pd.read_parquet(data_path).iterrows():
        extra = dict(raw["extra_info"])
        env_kwargs = dict(raw["env_kwargs"])
        rows.append(
            {
                "task_id": str(extra["task_id"]),
                "scenario": str(env_kwargs["scenario"]),
                "task_idx": int(env_kwargs["task_idx"]),
                "task": str(extra["task"]),
                "native_prompt_tokens": int(extra["native_prompt_tokens"]),
                "tool_schema_hash": str(extra["tool_schema_hash"]),
                "raw_tool_schema_hash": str(extra["raw_tool_schema_hash"]),
                "tool_schema_repair_count": int(extra["tool_schema_repair_count"]),
                "training_row": raw.to_dict(),
            }
        )
    if [row["task_id"] for row in rows] != expected_task_ids:
        raise RuntimeError("AWM candidate manifest and parquet IDs differ")
    return rows, manifest, integrity_manifest


def environment_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    for scenario in by_scenario:
        by_scenario[scenario].sort(key=lambda row: stable_rank(row["task_id"]))
    scenarios = sorted(by_scenario, key=stable_rank)
    maximum = max((len(items) for items in by_scenario.values()), default=0)
    return [by_scenario[scenario][rank] for rank in range(maximum) for scenario in scenarios if rank < len(by_scenario[scenario])]


class DeepSeekExpertPolicy:
    """Strict asynchronous DeepSeek native-tool policy for one-pass screening."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        api_base: str,
        max_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        expected_identity: Mapping[str, Any] | None = None,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.model = str(model)
        self.max_tokens = int(max_tokens)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._identity = dict(expected_identity) if expected_identity is not None else None
        self._identity_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self._stats = {
            "requests": 0,
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=self.max_tokens,
            extra_body={
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
        )
        identity = {
            "model": str(response.model or ""),
            "system_fingerprint": response.system_fingerprint,
        }
        if identity["model"] != self.model:
            raise RuntimeError(f"DeepSeek returned model {identity['model']!r}, expected {self.model!r}")
        async with self._identity_lock:
            if self._identity is not None and self._identity != identity:
                raise RuntimeError(f"DeepSeek provider identity changed: {self._identity!r} -> {identity!r}")
            self._identity = identity
        message = response.choices[0].message
        tool_calls = [call.model_dump(mode="json") for call in (message.tool_calls or [])]
        usage = response.usage.model_dump() if response.usage is not None else {}
        async with self._stats_lock:
            self._stats["requests"] += 1
            for name in self._stats.keys() - {"requests"}:
                self._stats[name] += int(usage.get(name, 0) or 0)
        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_calls[0].get("id") if tool_calls else None,
            "reasoning_content": getattr(message, "reasoning_content", "") or "",
            "finish_reason": response.choices[0].finish_reason,
            "model": identity["model"],
            "system_fingerprint": identity["system_fingerprint"],
            "usage": usage,
        }

    async def identity(self) -> dict[str, Any] | None:
        async with self._identity_lock:
            return dict(self._identity) if self._identity is not None else None

    async def stats(self) -> dict[str, int]:
        async with self._stats_lock:
            return dict(self._stats)
