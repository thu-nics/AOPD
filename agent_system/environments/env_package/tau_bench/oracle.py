"""OpenAI-compatible teacher policy for Tau Bench Agentic OPD."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, build_opener

import ray
from jsonschema import Draft202012Validator

from agent_system.environments.action_matching import (
    MESSAGE_MATCHER_INSTRUCTION,
    TOOL_MATCHER_INSTRUCTION,
    TOOL_MATCHER_PROTOCOL_VERSION,
    TOOL_MATCHER_SCOPE,
    message_evidence,
    tool_pair_fingerprint,
)
from agent_system.environments.semantic_matcher import match_message_matrix, matcher_boolean, matcher_decoding_config
from agent_system.environments.teacher_cache_import import TeacherCacheImport, valid_vote_records

from .actions import ParsedAction, parse_action

DEFAULT_TEACHER_API_BASE = "http://127.0.0.1:8000/v1"
ORACLE_PROTOCOL_VERSION = 10
MATCHER_PROTOCOL_VERSION = 5
DEFAULT_MATCHER_DECODING = matcher_decoding_config("openai-compatible")
MATCHER_SEMANTICS = MESSAGE_MATCHER_INSTRUCTION
MATCHER_SEMANTICS_HASH = hashlib.sha256(MATCHER_SEMANTICS.encode()).hexdigest()
logger = logging.getLogger(__name__)


def _normalize_message(value: str) -> str:
    return str(value).strip()


def _matcher_pair_fingerprint(
    *,
    model: str,
    api_base: str,
    teacher: str,
    candidate: str,
    chat=(),
    tools=(),
    decoding_config=None,
    semantics_hash=None,
) -> str:
    payload = {
        "protocol_version": MATCHER_PROTOCOL_VERSION,
        "model": str(model),
        "api_base": str(api_base).rstrip("/"),
        "semantics_hash": MATCHER_SEMANTICS_HASH if semantics_hash is None else semantics_hash,
        "decoding_config": DEFAULT_MATCHER_DECODING if decoding_config is None else dict(decoding_config),
        "teacher": _normalize_message(teacher),
        "candidate": _normalize_message(candidate),
        "evidence": message_evidence(teacher, candidate, chat, tools),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class TauTeacherClient:
    """Thread-safe client with deterministic seeds and an append-only state cache."""

    def __init__(
        self,
        *,
        model: str = "qwen3-32b",
        provider: str = "vllm",
        api_base: str = DEFAULT_TEACHER_API_BASE,
        api_key_env: str = "TAU_TEACHER_API_KEY",
        samples: int = 3,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        max_tokens: int = 8192,
        cache_path: str | None = None,
        teacher_cache_import_paths: Sequence[str] = (),
        matcher_cache_path: str | None = None,
        matcher_enabled: bool = True,
        matcher_provider: str = "vllm",
        matcher_profile: str = "default",
        matcher_model: str | None = None,
        matcher_api_base: str | None = None,
        matcher_api_key_env: str | None = None,
        matcher_enable_thinking: bool | None = None,
        matcher_max_tokens: int | None = None,
        matcher_reasoning_effort: str | None = None,
        matcher_max_concurrent_requests: int = 32,
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
        teacher_validity_max_retries: int = 2,
        max_concurrent_requests: int = 24,
    ):
        if type(matcher_enabled) is not bool:
            raise ValueError("matcher_enabled must be a boolean")
        self.matcher_enabled = matcher_enabled
        if not matcher_enabled:
            # Teacher-only clients need neither matcher credentials nor caches,
            # including when unrelated matcher settings are inherited.
            matcher_cache_path = None
            matcher_provider = "openai-compatible"
            matcher_profile = "default"
            matcher_model, matcher_api_base, matcher_api_key_env = model, api_base, api_key_env
            matcher_enable_thinking, matcher_max_tokens, matcher_reasoning_effort = False, 128, None
        if samples <= 0:
            raise ValueError("oracle samples must be positive")
        if teacher_validity_max_retries < 0:
            raise ValueError("teacher validity max retries must be non-negative")
        if type(matcher_max_concurrent_requests) is not int or matcher_max_concurrent_requests <= 0:
            raise ValueError("matcher_max_concurrent_requests must be positive")
        from aopd.providers import PROVIDERS

        if provider not in PROVIDERS or matcher_provider not in PROVIDERS:
            raise ValueError("unsupported teacher or matcher provider")
        self.provider = provider
        if matcher_provider == "deepseek" and not all((matcher_model, matcher_api_base, matcher_api_key_env)):
            raise ValueError("DeepSeek matcher requires explicit model, api_base and api_key_env")
        if matcher_api_base and str(matcher_api_base).rstrip("/") != str(api_base).rstrip("/") and not matcher_api_key_env:
            raise ValueError("a separate matcher endpoint requires an explicit matcher_api_key_env")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.api_key = api_key
        self.model = str(model)
        self.api_base = str(api_base).rstrip("/")
        self._http_opener = build_opener()
        self.samples = int(samples)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.teacher_validity_max_retries = int(teacher_validity_max_retries)
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self._cache_import = TeacherCacheImport(teacher_cache_import_paths, destination=self.cache_path)
        self.matcher_cache_path = Path(matcher_cache_path).expanduser() if matcher_cache_path else None
        self.matcher_provider = matcher_provider
        if matcher_profile not in {"default", "qwen38_concise"}:
            raise ValueError(f"unknown matcher profile: {matcher_profile}")
        self.matcher_profile = matcher_profile
        self.message_instruction = MATCHER_SEMANTICS
        self.message_semantics_hash = MATCHER_SEMANTICS_HASH
        self.tool_instruction = TOOL_MATCHER_INSTRUCTION
        self.tool_prompt_hash = None
        self.matcher_model = str(matcher_model or self.model)
        self.matcher_api_base = str(matcher_api_base or self.api_base).rstrip("/")
        self.matcher_api_key = os.environ.get(matcher_api_key_env or api_key_env)
        if self.matcher_enabled and not self.matcher_api_key:
            raise RuntimeError(f"missing required environment variable {matcher_api_key_env or api_key_env}")
        self.matcher_decoding = matcher_decoding_config(matcher_provider, enable_thinking=matcher_enable_thinking, reasoning_effort=matcher_reasoning_effort, max_tokens=matcher_max_tokens)
        self.tool_matcher_decoding = dict(self.matcher_decoding) if matcher_provider in {"deepseek", "zai"} else matcher_decoding_config(matcher_provider, enable_thinking=False, max_tokens=1024)
        if matcher_profile == "qwen38_concise":
            from .matcher_profiles import MESSAGE_CONCISE, TOOL_CONCISE, concise_decoding

            if matcher_provider != "vllm" or matcher_enable_thinking is True or matcher_reasoning_effort is not None:
                raise ValueError("qwen38_concise requires non-thinking OpenAI-compatible serving")
            if matcher_max_tokens not in {None, 32768}:
                raise ValueError("qwen38_concise freezes max_tokens=32768")
            self.message_instruction, self.tool_instruction = MESSAGE_CONCISE, TOOL_CONCISE
            self.message_semantics_hash = hashlib.sha256((MESSAGE_CONCISE + ":qwen38_concise/evidence-order-v1").encode()).hexdigest()
            self.tool_prompt_hash = hashlib.sha256((TOOL_CONCISE + ":qwen38_concise/evidence-order-v1").encode()).hexdigest()
            self.matcher_decoding = concise_decoding()
            self.tool_matcher_decoding = concise_decoding()
        self.matcher_max_concurrent_requests = matcher_max_concurrent_requests
        self._matcher_request_slots = threading.BoundedSemaphore(matcher_max_concurrent_requests)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._matcher_cache: dict[str, bool] = {}
        self._matcher_flights: dict[str, Future] = {}
        self._flights: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._stats = {
            "requests": 0,
            "cache_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_singleflight_waits": 0,
            "cache_generated_sets": 0,
            "cache_partial_hits": 0,
            "cache_refill_attempts": 0,
            "cache_refill_votes": 0,
            "cache_records_loaded": 0,
            "retries": 0,
            "teacher_validity_retries": 0,
            "teacher_validity_retry_recovered": 0,
            "teacher_validity_retry_exhausted": 0,
            "teacher_vote_request_failures": 0,
            "failures": 0,
            "semantic_exact_matches": 0,
            "semantic_failures": 0,
            "semantic_requests": 0,
            "parallel_tool_calls_truncated": 0,
            "matcher_cache_lookups": 0,
            "matcher_cache_hits": 0,
            "matcher_cache_misses": 0,
            "matcher_cache_records_loaded": 0,
        }
        self._load_cache()
        if self.matcher_enabled:
            self._load_matcher_cache()

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("protocol_version") != ORACLE_PROTOCOL_VERSION:
                    continue
                if (
                    record.get("model") != self.model
                    or record.get("provider") != self.provider
                    or record.get("effective_decoding") != self._teacher_decoding()
                    or int(record.get("samples", -1)) != self.samples
                    or float(record.get("temperature", -1)) != self.temperature
                    or float(record.get("top_p", -1)) != self.top_p
                    or int(record.get("top_k", -999)) != self.top_k
                    or float(record.get("min_p", -1)) != self.min_p
                    or bool(record.get("enable_thinking")) != self.enable_thinking
                    or int(record.get("max_tokens", -1)) != self.max_tokens
                    or int(record.get("teacher_validity_max_retries", -1)) != self.teacher_validity_max_retries
                ):
                    continue
                samples = record.get("teacher_samples")
                if not isinstance(samples, list) or len(samples) > self.samples:
                    continue
                if record.get("valid_samples") != len(samples):
                    continue
                if any(not isinstance(sample, Mapping) for sample in samples):
                    continue
                sample_indices = [sample.get("sample_index") for sample in samples]
                if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.samples for index in sample_indices) or len(set(sample_indices)) != len(sample_indices) or any(not isinstance(sample.get("action"), Mapping) for sample in samples):
                    continue
                self._cache[str(record["state_fingerprint"])] = [dict(sample) for sample in samples]
                self._stats["cache_records_loaded"] += 1

    def _load_matcher_cache(self) -> None:
        if self.matcher_cache_path is None or not self.matcher_cache_path.exists():
            return
        with self.matcher_cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("match_scope") == TOOL_MATCHER_SCOPE:
                    evidence = record.get("evidence")
                    if not isinstance(evidence, dict) or not isinstance(record.get("equivalent"), bool):
                        continue
                    key = self._tool_pair_key(evidence)
                    if record.get("protocol_version") == TOOL_MATCHER_PROTOCOL_VERSION and record.get("pair_fingerprint") == key:
                        self._matcher_cache[key] = record["equivalent"]
                        self._stats["matcher_cache_records_loaded"] += 1
                    continue
                if (
                    record.get("protocol_version") != MATCHER_PROTOCOL_VERSION
                    or record.get("provider", "openai-compatible") != self.matcher_provider
                    or record.get("model") != self.matcher_model
                    or record.get("api_base") != self.matcher_api_base
                    or record.get("semantics_hash") != self.message_semantics_hash
                    or not isinstance(record.get("equivalent"), bool)
                ):
                    continue
                evidence = record.get("evidence")
                if not isinstance(evidence, dict) or record.get("pair_fingerprint") != _matcher_pair_fingerprint(
                    model=self.matcher_model, api_base=self.matcher_api_base, teacher=record["teacher"], candidate=record["candidate"], chat=evidence.get("public_context", []), tools=evidence.get("tools", []), decoding_config=self.matcher_decoding, semantics_hash=self.message_semantics_hash
                ):
                    continue
                self._matcher_cache[str(record["pair_fingerprint"])] = bool(record["equivalent"])
                self._stats["matcher_cache_records_loaded"] += 1

    @staticmethod
    def _append_jsonl(path: Path | None, record: Mapping[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(dict(record), sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        with path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell() > 0:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(encoded)
            handle.flush()

    def _append_cache(
        self,
        state_fingerprint: str,
        samples: list[dict[str, Any]],
        *,
        messages: list[dict[str, Any]],
        teacher_context_mode: str,
    ) -> None:
        record = {
            "protocol_version": ORACLE_PROTOCOL_VERSION,
            "state_fingerprint": state_fingerprint,
            "model": self.model,
            "api_base": self.api_base,
            "provider": self.provider,
            "effective_decoding": self._teacher_decoding(),
            "cache_identity_scope": "model_not_endpoint",
            "samples": self.samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "teacher_validity_max_retries": self.teacher_validity_max_retries,
            "teacher_context_mode": teacher_context_mode,
            "teacher_prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest(),
            "valid_samples": len(samples),
            "teacher_samples": samples,
        }
        self._append_jsonl(self.cache_path, record)

    @staticmethod
    def _seed(state_fingerprint: str, sample_index: int, validity_retry: int = 0) -> int:
        digest = hashlib.sha256(f"{state_fingerprint}:{sample_index}:{validity_retry}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def _matcher_payload(self, messages, *, tool: bool = False) -> dict[str, Any]:
        from aopd.providers import adapt_chat_payload

        decoding = dict(self.tool_matcher_decoding if tool else self.matcher_decoding)
        return adapt_chat_payload({"model": self.matcher_model, "messages": messages, **decoding}, self.matcher_provider)

    def _matcher_evidence(self, evidence, *, tool=False):
        if self.matcher_profile == "qwen38_concise":
            from .matcher_profiles import order_evidence

            return order_evidence(evidence, tool=tool)
        return evidence

    def _post_matcher(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.matcher_enabled:
            raise RuntimeError("Tau matcher is disabled for programmatic-only supervision")
        response = self._post(payload, matcher=True)
        if self.matcher_provider == "deepseek":
            from agent_system.environments.env_package.awm.runtime.api_identity import checked_provider_identity

            checked_provider_identity(response, provider="deepseek", api_base=self.matcher_api_base, requested_model=self.matcher_model, role="Tau matcher")
        return response

    def _post(self, payload: dict[str, Any], *, matcher: bool = False) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        base = self.matcher_api_base if matcher else self.api_base
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        request = Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.matcher_api_key if matcher else self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._matcher_request_slots if matcher else self._request_slots:
                    with self._http_opener.open(request, timeout=self.timeout_seconds) as response:
                        result = json.loads(response.read().decode("utf-8"))
                with self._lock:
                    self._stats["requests"] += 1
                    self._stats["retries"] += attempt
                return result
            except (
                OSError,
                TimeoutError,
                HTTPException,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 == self.max_retries:
                    break
                retry_after = exc.headers.get("Retry-After") if isinstance(exc, HTTPError) else None
                try:
                    delay = float(retry_after) if retry_after else min(30.0, 2.0**attempt)
                except (TypeError, ValueError):
                    delay = min(30.0, 2.0**attempt)
                time.sleep(delay + random.random() * 0.25)
        with self._lock:
            self._stats["failures"] += 1
        raise RuntimeError(f"{'Matcher' if matcher else 'Teacher'} request failed after {self.max_retries} attempts: {last_error}")

    def _response_action(self, response: dict[str, Any]) -> ParsedAction:
        choices = response.get("choices") or []
        if not choices:
            return ParsedAction(kind="invalid", error="oracle response has no choices")
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) > 1:
                # Some providers ignore parallel_tool_calls=False.
                # Tau executes one action per decision, so preserve the first
                # action from each independent expert sample.
                with self._lock:
                    self._stats["parallel_tool_calls_truncated"] += 1
            function = tool_calls[0].get("function") or {}
            try:
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                return ParsedAction(
                    kind="tool",
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            except Exception as exc:
                return ParsedAction(kind="invalid", error=f"invalid oracle tool call: {exc}")
        if choices[0].get("finish_reason") == "length":
            return ParsedAction(kind="invalid", error="truncated teacher message")
        return parse_action(message.get("content"))

    def _teacher_decoding(self):
        from aopd.providers import adapt_chat_payload

        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.reasoning_effort is not None:
            config["reasoning_effort"] = self.reasoning_effort
        if self.thinking_budget is not None:
            config["thinking_budget"] = self.thinking_budget
        return adapt_chat_payload(config, self.provider)

    def _sample_once(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int,
    ) -> ParsedAction:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "seed": seed,
            **self._teacher_decoding(),
        }
        from aopd.providers import adapt_chat_payload

        return self._response_action(self._post(adapt_chat_payload(payload, self.provider)))

    @staticmethod
    def _validate_teacher_action(
        action: ParsedAction,
        tools: Sequence[Mapping[str, Any]],
    ) -> ParsedAction:
        """Validate one teacher vote against the student-visible native schema."""
        if action.kind == "invalid":
            return action
        if action.kind == "message":
            content = (action.content or "").strip()
            if not content:
                return ParsedAction(kind="invalid", error="empty teacher message")
            return ParsedAction(kind="message", content=content)
        if action.kind != "tool":
            return ParsedAction(
                kind="invalid",
                error=f"unsupported teacher action kind: {action.kind!r}",
            )

        functions = {}
        for tool in tools:
            function = tool.get("function")
            if isinstance(function, Mapping):
                functions[str(function.get("name") or "")] = function
        function = functions.get(action.name or "")
        if function is None:
            return ParsedAction(
                kind="invalid",
                error=f"unknown teacher tool: {action.name}",
            )
        arguments = action.arguments
        if not isinstance(arguments, dict):
            return ParsedAction(
                kind="invalid",
                error="teacher tool arguments must be a JSON object",
            )
        schema = function.get("parameters") or {"type": "object"}
        if not isinstance(schema, Mapping):
            return ParsedAction(
                kind="invalid",
                error=f"invalid schema for teacher tool: {action.name}",
            )
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            unknown = set(arguments) - set(properties)
            if unknown:
                return ParsedAction(
                    kind="invalid",
                    error=f"unknown teacher tool arguments: {sorted(unknown)}",
                )
        try:
            Draft202012Validator(dict(schema)).validate(arguments)
        except Exception as exc:
            return ParsedAction(
                kind="invalid",
                error=f"invalid teacher tool arguments: {exc}",
            )
        return ParsedAction(
            kind="tool",
            name=action.name,
            arguments=arguments,
        )

    def _sample_valid_teacher_vote(
        self,
        *,
        state_fingerprint: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sample_index: int,
    ) -> dict[str, Any] | None:
        """Generate one valid vote without resampling any valid peer vote."""
        for validity_retry in range(self.teacher_validity_max_retries + 1):
            action = self._sample_once(
                messages=messages,
                tools=tools,
                seed=self._seed(
                    state_fingerprint,
                    sample_index,
                    validity_retry,
                ),
            )
            checked = self._validate_teacher_action(action, tools)
            if checked.kind != "invalid":
                if validity_retry:
                    with self._lock:
                        self._stats["teacher_validity_retry_recovered"] += 1
                return {
                    "sample_index": sample_index,
                    "action": checked.to_dict(),
                    "validity_retry_count": validity_retry,
                }
            if validity_retry < self.teacher_validity_max_retries:
                with self._lock:
                    self._stats["teacher_validity_retries"] += 1
                continue
            with self._lock:
                self._stats["teacher_validity_retry_exhausted"] += 1
            return None
        raise AssertionError("unreachable teacher validity retry state")

    def _import_samples(self, fingerprint, messages, tools, context_mode):
        for record in self._cache_import.records(fingerprint):
            # Endpoint is transport/provenance, not the identity of a frozen teacher.
            settings = {name: getattr(self, name) for name in ("model", "samples", "temperature", "top_p", "top_k", "min_p", "enable_thinking", "max_tokens", "teacher_validity_max_retries")}
            if (
                record.get("protocol_version") != ORACLE_PROTOCOL_VERSION
                or record.get("provider") != self.provider
                or record.get("effective_decoding") != self._teacher_decoding()
                or any(record.get(k) != v for k, v in settings.items())
                or record.get("teacher_context_mode") != context_mode
                or record.get("teacher_prompt_sha256") != hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
                or not valid_vote_records(record, self.samples)
            ):
                continue
            imported = []
            # Tau v8's teacher validator did not insert defaults, drop nulls or
            # coerce argument values: saved tool arguments are lossless.
            for sample in record["teacher_samples"]:
                try:
                    action = ParsedAction(**sample["action"])
                    checked = self._validate_teacher_action(action, tools)
                except (TypeError, KeyError, ValueError):
                    continue
                if checked.kind != "invalid":
                    imported.append({**sample, "action": checked.to_dict(), "imported_from_protocol": record["protocol_version"], "source_api_base": record.get("api_base")})
            if imported:
                return imported
        return []

    def sample_multiset(
        self,
        *,
        state_fingerprint: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        teacher_context_mode: str = "student_visible",
    ) -> list[dict[str, Any]]:
        """Return up to K valid votes and refill partial exact-state cache rows."""
        if teacher_context_mode not in {"student_visible", "privileged"}:
            raise ValueError("unsupported teacher_context_mode")
        with self._lock:
            self._stats["cache_lookups"] += 1
            cached = self._cache.get(state_fingerprint)
            if cached is not None:
                self._stats["cache_hits"] += 1
                if len(cached) == self.samples:
                    return [dict(sample["action"]) for sample in cached]
                self._stats["cache_partial_hits"] += 1
            cached_samples = list(cached or [])
            flight = self._flights.get(state_fingerprint)
            if flight is None:
                flight = Future()
                self._flights[state_fingerprint] = flight
                if cached is None:
                    self._stats["cache_misses"] += 1
                else:
                    self._stats["cache_refill_attempts"] += 1
                leader = True
            else:
                self._stats["cache_singleflight_waits"] += 1
                leader = False
        if not leader:
            return list(flight.result())

        try:
            if cached is None:
                cached_samples = self._import_samples(state_fingerprint, messages, tools, teacher_context_mode)
                if cached_samples:
                    with self._lock:
                        self._stats["cache_imported_votes"] = self._stats.get("cache_imported_votes", 0) + len(cached_samples)
            existing_indices = {int(sample["sample_index"]) for sample in cached_samples}
            missing_indices = [index for index in range(self.samples) if index not in existing_indices]
            generated = []
            vote_errors = []
            with ThreadPoolExecutor(max_workers=max(1, len(missing_indices))) as pool:
                futures = [
                    (
                        index,
                        pool.submit(
                            self._sample_valid_teacher_vote,
                            state_fingerprint=state_fingerprint,
                            messages=messages,
                            tools=tools,
                            sample_index=index,
                        ),
                    )
                    for index in missing_indices
                ]
                for index, future in futures:
                    try:
                        sample = future.result()
                    except Exception as exc:
                        vote_errors.append(f"vote {index}: {exc}")
                        with self._lock:
                            self._stats["teacher_vote_request_failures"] += 1
                    else:
                        if sample is not None:
                            generated.append(sample)
            samples = sorted(
                [*cached_samples, *generated],
                key=lambda sample: int(sample["sample_index"]),
            )
            actions = [dict(sample["action"]) for sample in samples]
            if vote_errors:
                logger.warning(
                    "Tau teacher vote generation failed partially: %s",
                    "; ".join(vote_errors),
                )
            with self._lock:
                if cached is None or generated:
                    self._append_cache(
                        state_fingerprint,
                        samples,
                        messages=messages,
                        teacher_context_mode=teacher_context_mode,
                    )
                self._cache[state_fingerprint] = samples
                if cached is None:
                    self._stats["cache_generated_sets"] += 1
                else:
                    self._stats["cache_refill_votes"] += len(generated)
                self._flights.pop(state_fingerprint, None)
                flight.set_result(tuple(actions))
            return list(actions)
        except BaseException as exc:
            with self._lock:
                self._flights.pop(state_fingerprint, None)
                flight.set_exception(exc)
            raise

    def _remember_matcher_decision(
        self,
        *,
        teacher: str,
        candidate: str,
        equivalent: bool,
        chat=(),
        tools=(),
    ) -> bool:
        fingerprint = _matcher_pair_fingerprint(
            model=self.matcher_model,
            api_base=self.matcher_api_base,
            teacher=teacher,
            candidate=candidate,
            chat=chat,
            tools=tools,
            decoding_config=self.matcher_decoding,
            semantics_hash=self.message_semantics_hash,
        )
        with self._lock:
            if fingerprint in self._matcher_cache:
                return self._matcher_cache[fingerprint]
            self._matcher_cache[fingerprint] = bool(equivalent)
            self._append_jsonl(
                self.matcher_cache_path,
                {
                    "protocol_version": MATCHER_PROTOCOL_VERSION,
                    "pair_fingerprint": fingerprint,
                    "provider": self.matcher_provider,
                    "model": self.matcher_model,
                    "api_base": self.matcher_api_base,
                    "semantics_hash": self.message_semantics_hash,
                    "decoding_config": self.matcher_decoding,
                    "teacher": _normalize_message(teacher),
                    "candidate": _normalize_message(candidate),
                    "equivalent": bool(equivalent),
                    "evidence": message_evidence(teacher, candidate, chat, tools),
                },
            )
        return bool(equivalent)

    def _tool_pair_key(self, evidence):
        return tool_pair_fingerprint(provider=self.matcher_provider, model=self.matcher_model, endpoint=self.matcher_api_base, decoding_config=self.tool_matcher_decoding, evidence=evidence, prompt_hash=self.tool_prompt_hash)

    def _match_tool_pair(self, evidence):
        if not self.matcher_enabled:
            raise RuntimeError("Tau matcher is disabled for programmatic-only supervision")
        key = self._tool_pair_key(evidence)
        with self._lock:
            self._stats["matcher_cache_lookups"] += 1
            if key in self._matcher_cache:
                self._stats["matcher_cache_hits"] += 1
                return self._matcher_cache[key]
            flight = self._matcher_flights.get(key)
            owner = flight is None
            if owner:
                flight = Future()
                self._matcher_flights[key] = flight
                self._stats["matcher_cache_misses"] += 1
        if not owner:
            return flight.result()
        try:
            response = self._post_matcher(self._matcher_payload([{"role": "system", "content": self.tool_instruction}, {"role": "user", "content": json.dumps(self._matcher_evidence(evidence, tool=True), ensure_ascii=False)}], tool=True))
            value = matcher_boolean(response)
            with self._lock:
                self._append_jsonl(
                    self.matcher_cache_path,
                    {
                        "match_scope": TOOL_MATCHER_SCOPE,
                        "protocol_version": TOOL_MATCHER_PROTOCOL_VERSION,
                        "pair_fingerprint": key,
                        "provider": self.matcher_provider,
                        "model": self.matcher_model,
                        "api_base": self.matcher_api_base,
                        "decoding_config": self.tool_matcher_decoding,
                        "evidence": evidence,
                        "equivalent": value,
                    },
                )
                self._matcher_cache[key] = value
            flight.set_result(value)
            return value
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._matcher_flights.pop(key, None)

    def match_tool_argument_pairs(self, pairs):
        keys = [self._tool_pair_key(pair) for pair in pairs]
        unique = dict(zip(keys, pairs, strict=True))
        with ThreadPoolExecutor(max_workers=max(1, min(self.matcher_max_concurrent_requests, len(unique)))) as pool:
            decisions = dict(zip(unique, pool.map(self._match_tool_pair, unique.values()), strict=True))
        return [decisions[key] for key in keys]

    def _match_pair(self, teacher, candidate, chat=(), tools=()):
        if _normalize_message(teacher) == _normalize_message(candidate):
            with self._lock:
                self._stats["semantic_exact_matches"] += 1
            return True
        key = _matcher_pair_fingerprint(model=self.matcher_model, api_base=self.matcher_api_base, teacher=teacher, candidate=candidate, chat=chat, tools=tools, decoding_config=self.matcher_decoding, semantics_hash=self.message_semantics_hash)
        with self._lock:
            self._stats["matcher_cache_lookups"] += 1
            if key in self._matcher_cache:
                self._stats["matcher_cache_hits"] += 1
                return self._matcher_cache[key]
            flight = self._matcher_flights.get(key)
            owner = flight is None
            if owner:
                flight = Future()
                self._matcher_flights[key] = flight
                self._stats["matcher_cache_misses"] += 1
        if not owner:
            return flight.result()
        try:
            with self._lock:
                self._stats["semantic_requests"] += 1
            response = self._post_matcher(
                self._matcher_payload(
                    [
                        {"role": "system", "content": self.message_instruction},
                        {"role": "user", "content": json.dumps(self._matcher_evidence(message_evidence(teacher, candidate, chat, tools)), ensure_ascii=False)},
                    ]
                )
            )
            value = matcher_boolean(response)
            value = self._remember_matcher_decision(teacher=teacher, candidate=candidate, equivalent=value, chat=chat, tools=tools)
            flight.set_result(value)
            return value
        except BaseException as exc:
            with self._lock:
                self._stats["semantic_failures"] += 1
            flight.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._matcher_flights.pop(key, None)

    def match_message_pairs(self, teacher_messages, candidate_messages, chat=(), tools=()):
        if not self.matcher_enabled:
            raise RuntimeError("Tau matcher is disabled for programmatic-only supervision")
        return match_message_matrix(teacher_messages, candidate_messages, lambda teacher, candidate: self._match_pair(teacher, candidate, chat, tools), normalize=_normalize_message, max_workers=self.matcher_max_concurrent_requests)

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            stats = dict(self._stats)
            lookups = stats["cache_lookups"]
            stats["cache_hit_rate"] = stats["cache_hits"] / lookups if lookups else 0.0
            matcher_lookups = stats["matcher_cache_lookups"]
            stats["matcher_cache_hit_rate"] = stats["matcher_cache_hits"] / matcher_lookups if matcher_lookups else 0.0
            return stats


def build_teacher_messages(
    visible_chat: Sequence[Mapping[str, Any]],
    *,
    privileged_context: Mapping[str, Any] | None = None,
    use_privileged_context: bool = False,
) -> list[dict[str, Any]]:
    """Build Tau teacher messages from the exact student-visible native chat."""
    if not visible_chat:
        raise ValueError("Tau teacher requires a non-empty visible chat")
    messages = [dict(message) for message in visible_chat]
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message.setdefault("reasoning_content", "")
    if use_privileged_context:
        if not privileged_context:
            raise ValueError("privileged teacher context was enabled without structured context")
        if messages[0].get("role") != "system":
            raise ValueError("privileged teacher context requires a system message")
        messages[0] = dict(messages[0])
        messages[0]["content"] = f"{messages[0].get('content') or ''}\n\nPRIVILEGED TEACHER CONTEXT:\n" + json.dumps(privileged_context, ensure_ascii=False, sort_keys=True)
    return messages


@ray.remote(max_concurrency=32)
class TauTeacherActor:
    """Central async Ray actor shared by all Tau environment workers."""

    def __init__(self, **kwargs):
        self.client = TauTeacherClient(**kwargs)

    async def sample_multiset(self, **kwargs):
        return await asyncio.to_thread(self.client.sample_multiset, **kwargs)

    async def match_message_pairs(self, teacher_messages, candidate_messages, chat=(), tools=()):
        return await asyncio.to_thread(
            self.client.match_message_pairs,
            teacher_messages,
            candidate_messages,
            chat,
            tools,
        )

    async def match_tool_argument_pairs(self, pairs):
        return await asyncio.to_thread(self.client.match_tool_argument_pairs, pairs)

    def get_stats(self):
        return self.client.stats()
