"""Provider-aware teacher multiset and frozen semantic matcher for AWM."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import ray

from agent_system.environments.env_package.envscaler.runtime_judge import (
    ENVSCALER_RUNTIME_JUDGE_INSTRUCTION,
    ENVSCALER_RUNTIME_JUDGE_PROMPT_HASH,
    ENVSCALER_RUNTIME_JUDGE_PROTOCOL_VERSION,
    ENVSCALER_RUNTIME_JUDGE_SCOPE,
    envscaler_runtime_judge_fingerprint,
    validate_envscaler_runtime_judge_verdict,
)

from .actions import (
    canonical_action,
    normalize_message,
    parse_native_action,
    tool_schema_hash,
    validate_action,
)
from .judge import (
    RUNTIME_JUDGE_INSTRUCTION,
    RUNTIME_JUDGE_PROMPT_HASH,
    RUNTIME_JUDGE_PROTOCOL_VERSION,
    RuntimeJudgeEvidenceStore,
    runtime_judge_decoding_config,
    runtime_judge_fingerprint,
    validate_runtime_judge_verdict,
)

DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SUPPORTED_ORACLE_PROVIDERS = frozenset({"deepseek", "dashscope"})
ORACLE_PROTOCOL_VERSION = 13
MATCHER_PROTOCOL_VERSION = 3
DEFAULT_MODEL = "deepseek-v4-flash"
TEACHER_PROMPT_REVISION = "single_action_v1"
TEACHER_SINGLE_ACTION_INSTRUCTION = (
    "CRITICAL SINGLE-ACTION PROTOCOL: Return exactly one next action. If a tool "
    "is needed, emit at most one function call and no message; never batch or "
    "parallelize calls. Even when several calls are independent, choose only "
    "the single best next call and wait for its result. Otherwise return one "
    "communicative message and no function call."
)
TEACHER_PROMPT_HASH = hashlib.sha256(TEACHER_SINGLE_ACTION_INSTRUCTION.encode()).hexdigest()
MATCHER_INSTRUCTION = (
    "You are a frozen semantic equivalence matcher, not an action-quality judge. "
    "Decide only whether the candidate and teacher messages express the same "
    "immediate communicative action with materially equivalent information. "
    "Do not reward helpfulness, correctness, or topic similarity. Return exactly "
    'this JSON object: {"equivalent":true} or {"equivalent":false}.'
)
MATCHER_DECODING_CONFIG = {
    "thinking": {"type": "disabled"},
    "temperature": 0,
    "max_tokens": 128,
    "response_format": {"type": "json_object"},
    "stream": False,
}
MATCHER_PROMPT_HASH = hashlib.sha256(MATCHER_INSTRUCTION.encode()).hexdigest()
_RUNTIME_JUDGE_CLASS_STATS = {
    "policy_execution_error": "runtime_judge_policy_execution_errors",
    "infrastructure_error": "runtime_judge_infrastructure_errors",
    "uncertain": "runtime_judge_uncertain",
}
_ENVSCALER_RUNTIME_JUDGE_CLASS_STATS = {
    "policy_execution_error": "envscaler_runtime_judge_policy_execution_errors",
    "infrastructure_error": "envscaler_runtime_judge_infrastructure_errors",
    "uncertain": "envscaler_runtime_judge_uncertain",
}

logger = logging.getLogger(__name__)


def _json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("matcher response must be a JSON object")
    return parsed


def _pair_fingerprint(
    model: str,
    teacher: str,
    candidate: str,
    *,
    decoding_config: Mapping[str, Any] = MATCHER_DECODING_CONFIG,
) -> str:
    payload = {
        "protocol_version": MATCHER_PROTOCOL_VERSION,
        "model": model,
        "prompt_hash": MATCHER_PROMPT_HASH,
        "decoding_config": dict(decoding_config),
        "teacher": normalize_message(teacher),
        "candidate": normalize_message(candidate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _chat_completions_url(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _matcher_decoding_config(provider: str) -> dict[str, Any]:
    if provider == "deepseek":
        return dict(MATCHER_DECODING_CONFIG)
    if provider == "dashscope":
        return {
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
    raise ValueError(f"unsupported oracle provider: {provider!r}")


class DeepSeekAWMOracleClient:
    """Thread-safe provider-aware client with append-only strict caches.

    The historical class name is retained for import compatibility. Teacher,
    matcher, and runtime-judge transports are configured independently.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        provider: str = "deepseek",
        api_base: str = DEFAULT_DEEPSEEK_API_BASE,
        api_key_env: str = "DEEPSEEK_API_KEY",
        samples: int = 3,
        enable_thinking: bool = True,
        reasoning_effort: str | None = "max",
        thinking_budget: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int = 4096,
        matcher_provider: str = "deepseek",
        matcher_model: str = DEFAULT_MODEL,
        matcher_api_base: str = DEFAULT_DEEPSEEK_API_BASE,
        matcher_api_key_env: str = "DEEPSEEK_API_KEY",
        cache_path: str | None = None,
        matcher_cache_path: str | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 5,
        max_concurrent_requests: int = 32,
        teacher_multi_call_fallback_enabled: bool = True,
        teacher_multi_call_fallback_min_repeat_streak: int = 2,
        runtime_judge_enabled: bool = False,
        runtime_judge_data_dir: str | None = None,
        runtime_judge_reference_trials_path: str | None = None,
        runtime_judge_cache_path: str | None = None,
        runtime_judge_provider: str = "deepseek",
        runtime_judge_model: str = DEFAULT_MODEL,
        runtime_judge_api_base: str = DEFAULT_DEEPSEEK_API_BASE,
        runtime_judge_api_key_env: str = "DEEPSEEK_API_KEY",
        runtime_judge_reasoning_effort: str = "max",
        runtime_judge_max_tokens: int = 8192,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        if int(samples) != 3:
            raise ValueError("AWM agentic OPD training requires exactly three teacher samples")
        provider = str(provider).lower()
        matcher_provider = str(matcher_provider).lower()
        runtime_judge_provider = str(runtime_judge_provider).lower()
        for service, selected in (
            ("teacher", provider),
            ("matcher", matcher_provider),
            ("runtime judge", runtime_judge_provider),
        ):
            if selected not in SUPPORTED_ORACLE_PROVIDERS:
                raise ValueError(f"unsupported {service} provider: {selected!r}")
        if runtime_judge_provider != "deepseek":
            raise ValueError("AWM runtime judges currently require provider='deepseek'")
        if provider == "deepseek" and reasoning_effort != "max":
            raise ValueError("DeepSeek AWM teacher protocol requires reasoning_effort='max'")
        if provider == "dashscope" and not bool(enable_thinking):
            raise ValueError("DashScope Qwen teacher protocol requires thinking enabled")
        if thinking_budget is not None and int(thinking_budget) <= 0:
            raise ValueError("teacher thinking_budget must be positive when configured")
        service_specs = {
            "teacher": (str(model), str(api_base), str(api_key_env)),
            "matcher": (str(matcher_model), str(matcher_api_base), str(matcher_api_key_env)),
            "runtime_judge": (str(runtime_judge_model), str(runtime_judge_api_base), str(runtime_judge_api_key_env)),
            "envscaler_runtime_judge": (str(runtime_judge_model), str(runtime_judge_api_base), str(runtime_judge_api_key_env)),
        }
        self._service_models = {name: spec[0] for name, spec in service_specs.items()}
        for name, (selected_model, selected_base, key_env) in service_specs.items():
            if not selected_model.strip() or not selected_base.strip() or not key_env.strip():
                raise ValueError(f"{name} model, api_base, and api_key_env must be non-empty")
        self._service_urls = {name: _chat_completions_url(spec[1]) for name, spec in service_specs.items()}
        self._service_api_keys = {}
        for name, (_, _, key_env) in service_specs.items():
            api_key = os.environ.get(key_env)
            optional_runtime_service = name in {"runtime_judge", "envscaler_runtime_judge"}
            if not api_key and request_fn is None and (not optional_runtime_service or runtime_judge_enabled):
                raise RuntimeError(f"missing required environment variable {key_env} for {name}")
            self._service_api_keys[name] = api_key or ""
        self.model = str(model)
        self.provider = provider
        self.matcher_model = str(matcher_model)
        self.matcher_provider = matcher_provider
        self.matcher_decoding_config = _matcher_decoding_config(matcher_provider)
        self.runtime_judge_model = str(runtime_judge_model)
        self.runtime_judge_provider = runtime_judge_provider
        self.samples = int(samples)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None
        self.temperature = float(temperature) if temperature is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.presence_penalty = float(presence_penalty) if presence_penalty is not None else None
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.teacher_multi_call_fallback_enabled = bool(teacher_multi_call_fallback_enabled)
        self.teacher_multi_call_fallback_min_repeat_streak = int(teacher_multi_call_fallback_min_repeat_streak)
        if self.teacher_multi_call_fallback_min_repeat_streak < 2:
            raise ValueError("teacher multi-call fallback minimum repeat streak must be at least two")
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.matcher_cache_path = Path(matcher_cache_path).expanduser() if matcher_cache_path else None
        self.runtime_judge_enabled = bool(runtime_judge_enabled)
        self.runtime_judge_cache_path = Path(runtime_judge_cache_path).expanduser() if runtime_judge_cache_path else None
        self.runtime_judge_decoding_config = runtime_judge_decoding_config(
            reasoning_effort=runtime_judge_reasoning_effort,
            max_tokens=runtime_judge_max_tokens,
        )
        self.runtime_judge_evidence = (
            RuntimeJudgeEvidenceStore(
                data_dir=str(runtime_judge_data_dir or ""),
                reference_trials_path=runtime_judge_reference_trials_path,
            )
            if self.runtime_judge_enabled
            else None
        )
        self._request_fn = request_fn
        self._state_cache: dict[str, list[dict[str, Any]]] = {}
        self._state_flights: dict[str, Future] = {}
        self._matcher_cache: dict[str, bool] = {}
        self._runtime_judge_cache: dict[str, dict[str, Any]] = {}
        self._runtime_judge_flights: dict[str, Future] = {}
        self._envscaler_runtime_judge_cache: dict[str, dict[str, Any]] = {}
        self._envscaler_runtime_judge_flights: dict[str, Future] = {}
        self._provider_identities: dict[str, dict[str, Any] | None] = {
            "teacher": None,
            "matcher": None,
            "runtime_judge": None,
            "envscaler_runtime_judge": None,
        }
        self._lock = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(int(max_concurrent_requests))
        self._stats = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
            "teacher_requests": 0,
            "teacher_prompt_tokens": 0,
            "teacher_completion_tokens": 0,
            "teacher_total_tokens": 0,
            "teacher_cache_lookups": 0,
            "teacher_cache_hits": 0,
            "teacher_cache_misses": 0,
            "teacher_cache_singleflight_waits": 0,
            "teacher_cache_generated_sets": 0,
            "teacher_cache_records_loaded": 0,
            "teacher_parallel_calls_truncated": 0,
            "teacher_parallel_responses": 0,
            "teacher_tool_calls_total": 0,
            "teacher_parallel_alternative_available": 0,
            "teacher_parallel_fallback_applied": 0,
            "matcher_requests": 0,
            "matcher_prompt_tokens": 0,
            "matcher_completion_tokens": 0,
            "matcher_total_tokens": 0,
            "matcher_cache_hits": 0,
            "matcher_cache_records_loaded": 0,
            "matcher_exact_matches": 0,
            "matcher_pair_evaluations": 0,
            "matcher_unique_pairs": 0,
            "matcher_failures": 0,
            "runtime_judge_requests": 0,
            "runtime_judge_prompt_tokens": 0,
            "runtime_judge_completion_tokens": 0,
            "runtime_judge_total_tokens": 0,
            "runtime_judge_cache_lookups": 0,
            "runtime_judge_cache_hits": 0,
            "runtime_judge_cache_misses": 0,
            "runtime_judge_cache_singleflight_waits": 0,
            "runtime_judge_cache_records_loaded": 0,
            "runtime_judge_policy_execution_errors": 0,
            "runtime_judge_infrastructure_errors": 0,
            "runtime_judge_uncertain": 0,
            "runtime_judge_failures": 0,
            "envscaler_runtime_judge_requests": 0,
            "envscaler_runtime_judge_prompt_tokens": 0,
            "envscaler_runtime_judge_completion_tokens": 0,
            "envscaler_runtime_judge_total_tokens": 0,
            "envscaler_runtime_judge_cache_lookups": 0,
            "envscaler_runtime_judge_cache_hits": 0,
            "envscaler_runtime_judge_cache_misses": 0,
            "envscaler_runtime_judge_cache_singleflight_waits": 0,
            "envscaler_runtime_judge_cache_records_loaded": 0,
            "envscaler_runtime_judge_policy_execution_errors": 0,
            "envscaler_runtime_judge_infrastructure_errors": 0,
            "envscaler_runtime_judge_uncertain": 0,
            "envscaler_runtime_judge_failures": 0,
        }
        for prefix in self._provider_identities:
            self._stats[f"{prefix}_provider_fingerprint_count"] = 0
            self._stats[f"{prefix}_provider_fingerprint_changes"] = 0
        self._provider_fingerprints: dict[str, set[str | None]] = {prefix: set() for prefix in self._provider_identities}
        self._load_state_cache()
        self._load_matcher_cache()
        self._load_runtime_judge_cache()

    def _teacher_decoding_config(self) -> dict[str, Any]:
        if self.provider == "deepseek":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.reasoning_effort,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        config = {
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        for name, value in (
            ("thinking_budget", self.thinking_budget),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("presence_penalty", self.presence_penalty),
        ):
            if value is not None:
                config[name] = value
        return config

    def _teacher_protocol_config(self) -> dict[str, Any]:
        return {
            "teacher_prompt_revision": TEACHER_PROMPT_REVISION,
            "teacher_prompt_hash": TEACHER_PROMPT_HASH,
            "multi_call_fallback_enabled": self.teacher_multi_call_fallback_enabled,
            "multi_call_fallback_min_repeat_streak": (self.teacher_multi_call_fallback_min_repeat_streak),
        }

    def _teacher_progress_context(
        self,
        *,
        previous_canonical_action: str | None,
        no_progress_repeat_streak: int,
    ) -> dict[str, Any]:
        fallback_eligible = bool(self.teacher_multi_call_fallback_enabled and previous_canonical_action is not None and int(no_progress_repeat_streak) >= self.teacher_multi_call_fallback_min_repeat_streak)
        return {
            "multi_call_fallback_eligible": fallback_eligible,
            "previous_canonical_action": (previous_canonical_action if fallback_eligible else None),
        }

    @staticmethod
    def _teacher_cache_fingerprint(
        state_fingerprint: str,
        progress_context: Mapping[str, Any],
    ) -> str:
        payload = {
            "protocol_version": ORACLE_PROTOCOL_VERSION,
            "state_fingerprint": str(state_fingerprint),
            "progress_context": dict(progress_context),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _record_usage(
        self,
        usage: Mapping[str, Any] | None,
        *,
        prefix: str,
    ) -> None:
        if not isinstance(usage, Mapping):
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        self._stats[f"{prefix}_prompt_tokens"] += prompt
        self._stats[f"{prefix}_completion_tokens"] += completion
        self._stats[f"{prefix}_total_tokens"] += total

    def _accept_provider_identity(
        self,
        response_or_identity: Mapping[str, Any],
        *,
        prefix: str,
    ) -> dict[str, Any]:
        identity = {
            "model": str(response_or_identity.get("model") or ""),
            "system_fingerprint": (str(response_or_identity["system_fingerprint"]) if response_or_identity.get("system_fingerprint") is not None else None),
        }
        expected_model = self._service_models[prefix]
        if identity["model"] != expected_model:
            returned_model = identity["model"]
            raise RuntimeError(f"{prefix} returned model {returned_model!r}, expected {expected_model!r}")
        new_fingerprint = False
        with self._lock:
            fingerprints = self._provider_fingerprints[prefix]
            fingerprint = identity["system_fingerprint"]
            if fingerprint not in fingerprints:
                new_fingerprint = bool(fingerprints)
                fingerprints.add(fingerprint)
                self._stats[f"{prefix}_provider_fingerprint_count"] = len(fingerprints)
                if new_fingerprint:
                    self._stats[f"{prefix}_provider_fingerprint_changes"] += 1
            self._provider_identities[prefix] = identity
        if new_fingerprint:
            logger.warning(
                "%s system_fingerprint changed while model remained %s; accepting the response and retaining the per-response identity: %r",
                prefix,
                expected_model,
                identity["system_fingerprint"],
            )
        return identity

    def _load_state_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.is_file():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                compatible = (
                    record.get("protocol_version") == ORACLE_PROTOCOL_VERSION
                    and record.get("provider", "deepseek") == self.provider
                    and record.get("model") == self.model
                    and int(record.get("samples", -1)) == self.samples
                    and record.get("decoding_config") == self._teacher_decoding_config()
                    and record.get("teacher_protocol_config") == self._teacher_protocol_config()
                )
                if not compatible:
                    continue
                samples = record.get("teacher_samples")
                progress_context = record.get("progress_context")
                if not isinstance(progress_context, Mapping):
                    continue
                cache_fingerprint = self._teacher_cache_fingerprint(
                    str(record.get("state_fingerprint") or ""),
                    progress_context,
                )
                if record.get("teacher_cache_fingerprint") != cache_fingerprint:
                    continue
                if isinstance(samples, list) and len(samples) == self.samples:
                    identities = [sample.get("provider_identity") for sample in samples]
                    if any(not isinstance(identity, Mapping) for identity in identities):
                        continue
                    for identity in identities:
                        self._accept_provider_identity(identity, prefix="teacher")
                    self._state_cache[cache_fingerprint] = samples
                    self._stats["teacher_cache_records_loaded"] += 1

    def _load_matcher_cache(self) -> None:
        if self.matcher_cache_path is None or not self.matcher_cache_path.is_file():
            return
        with self.matcher_cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("protocol_version") != MATCHER_PROTOCOL_VERSION
                    or record.get("provider", "deepseek") != self.matcher_provider
                    or record.get("model") != self.matcher_model
                    or record.get("prompt_hash") != MATCHER_PROMPT_HASH
                    or record.get("decoding_config") != self.matcher_decoding_config
                    or not isinstance(record.get("equivalent"), bool)
                ):
                    continue
                identity = record.get("provider_identity")
                if not isinstance(identity, Mapping):
                    continue
                self._accept_provider_identity(identity, prefix="matcher")
                self._matcher_cache[str(record["pair_fingerprint"])] = bool(record["equivalent"])
                self._stats["matcher_cache_records_loaded"] += 1

    def _load_runtime_judge_cache(self) -> None:
        path = self.runtime_judge_cache_path
        if path is None or not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("judge_scope") == ENVSCALER_RUNTIME_JUDGE_SCOPE:
                    try:
                        verdict = validate_envscaler_runtime_judge_verdict(record.get("verdict"))
                    except (TypeError, ValueError):
                        continue
                    compatible = (
                        record.get("protocol_version") == ENVSCALER_RUNTIME_JUDGE_PROTOCOL_VERSION
                        and record.get("provider", "deepseek") == self.runtime_judge_provider
                        and record.get("model") == self.runtime_judge_model
                        and record.get("prompt_hash") == ENVSCALER_RUNTIME_JUDGE_PROMPT_HASH
                        and record.get("decoding_config") == self.runtime_judge_decoding_config
                    )
                    evidence = record.get("evidence")
                    if not compatible or not isinstance(evidence, Mapping):
                        continue
                    expected_fingerprint = envscaler_runtime_judge_fingerprint(
                        model=self.runtime_judge_model,
                        decoding_config=self.runtime_judge_decoding_config,
                        evidence=evidence,
                    )
                    if record.get("evidence_fingerprint") != expected_fingerprint:
                        continue
                    identity = record.get("provider_identity")
                    if not isinstance(identity, Mapping):
                        continue
                    self._accept_provider_identity(
                        identity,
                        prefix="envscaler_runtime_judge",
                    )
                    self._envscaler_runtime_judge_cache[str(record["evidence_fingerprint"])] = verdict
                    self._stats["envscaler_runtime_judge_cache_records_loaded"] += 1
                    continue
                try:
                    verdict = validate_runtime_judge_verdict(record.get("verdict"))
                except (TypeError, ValueError):
                    continue
                if (
                    record.get("protocol_version") != RUNTIME_JUDGE_PROTOCOL_VERSION
                    or record.get("provider", "deepseek") != self.runtime_judge_provider
                    or record.get("model") != self.runtime_judge_model
                    or record.get("prompt_hash") != RUNTIME_JUDGE_PROMPT_HASH
                    or record.get("decoding_config") != self.runtime_judge_decoding_config
                ):
                    continue
                evidence = record.get("evidence")
                if not isinstance(evidence, Mapping):
                    continue
                expected_fingerprint = runtime_judge_fingerprint(
                    model=self.runtime_judge_model,
                    decoding_config=self.runtime_judge_decoding_config,
                    evidence=evidence,
                )
                if record.get("evidence_fingerprint") != expected_fingerprint:
                    continue
                identity = record.get("provider_identity")
                if not isinstance(identity, Mapping):
                    continue
                self._accept_provider_identity(
                    identity,
                    prefix="runtime_judge",
                )
                self._runtime_judge_cache[str(record["evidence_fingerprint"])] = verdict
                self._stats["runtime_judge_cache_records_loaded"] += 1

    @staticmethod
    def _append_jsonl(path: Path | None, record: Mapping[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode()
        with path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell() > 0:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(encoded)
            handle.flush()

    def _post(self, payload: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        if self._request_fn is not None:
            with self._lock:
                self._stats["requests"] += 1
            return self._request_fn(payload)
        request = Request(
            self._service_urls[prefix],
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self._service_api_keys[prefix]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._request_slots:
                    with urlopen(request, timeout=self.timeout_seconds) as response:
                        result = json.loads(response.read().decode())
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
                if isinstance(exc, HTTPError):
                    try:
                        response_body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        response_body = ""
                    last_error = RuntimeError(f"{exc}; response_body={response_body[:2048]!r}")
                else:
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
        raise RuntimeError(f"{prefix} request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _response_content(response: Mapping[str, Any]) -> tuple[str, str]:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("provider response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("provider response has no final content")
        return str(content), str(message.get("reasoning_content") or "")

    def _sample_once(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        sample_index: int,
        *,
        previous_canonical_action: str | None = None,
        no_progress_repeat_streak: int = 0,
    ) -> dict[str, Any]:
        response = self._post(
            {
                "model": self.model,
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                **self._teacher_decoding_config(),
            },
            prefix="teacher",
        )
        with self._lock:
            self._stats["teacher_requests"] += 1
            self._record_usage(response.get("usage"), prefix="teacher")
        provider_identity = self._accept_provider_identity(response, prefix="teacher")
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("teacher response has no choices")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or "")
        calls = list(message.get("tool_calls") or [])
        action, skipped_calls = self._select_single_action(
            content=content,
            calls=calls,
            tools=tools,
            previous_canonical_action=previous_canonical_action,
            no_progress_repeat_streak=no_progress_repeat_streak,
        )
        if skipped_calls:
            with self._lock:
                self._stats["teacher_parallel_calls_truncated"] += skipped_calls
        return {
            "sample_index": int(sample_index),
            "action": action.to_dict(),
            "raw_content": content,
            "raw_tool_calls": calls,
            "raw_tool_call_count": len(calls),
            "skipped_tool_calls": skipped_calls,
            "reasoning_content": reasoning,
            "finish_reason": choices[0].get("finish_reason"),
            "provider_identity": provider_identity,
            "usage": dict(response.get("usage") or {}),
        }

    def _select_single_action(
        self,
        *,
        content: str,
        calls: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        previous_canonical_action: str | None,
        no_progress_repeat_streak: int,
    ) -> tuple[Any, int]:
        """Turn one provider response into one teacher vote.

        The native single-action protocol always treats the first call as the
        primary action. Later calls are visible only to the bounded
        no-progress fallback; an invalid first call must not be silently
        replaced by a later valid call.
        """
        calls = list(calls)
        if not calls:
            return parse_native_action(content, calls, take_first=True)

        parsed_calls = []
        for call in calls:
            parsed, _ = parse_native_action(None, [call], take_first=True)
            parsed_calls.append((parsed, validate_action(parsed, tools)))
        primary, checked_primary = parsed_calls[0]
        alternative_available = False
        fallback_applied = False
        selected = primary
        if self.teacher_multi_call_fallback_enabled and previous_canonical_action is not None and int(no_progress_repeat_streak) >= self.teacher_multi_call_fallback_min_repeat_streak and checked_primary.kind == "tool" and canonical_action(checked_primary) == previous_canonical_action:
            alternatives = [parsed for parsed, checked in parsed_calls[1:] if checked.kind == "tool" and canonical_action(checked) != previous_canonical_action]
            alternative_available = bool(alternatives)
            if alternatives:
                selected = alternatives[0]
                fallback_applied = True

        with self._lock:
            self._stats["teacher_tool_calls_total"] += len(calls)
            if len(calls) > 1:
                self._stats["teacher_parallel_responses"] += 1
            if alternative_available:
                self._stats["teacher_parallel_alternative_available"] += 1
            if fallback_applied:
                self._stats["teacher_parallel_fallback_applied"] += 1
        return selected, max(0, len(calls) - 1)

    def sample_multiset(
        self,
        *,
        state_fingerprint: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        previous_canonical_action: str | None = None,
        no_progress_repeat_streak: int = 0,
    ) -> list[dict[str, Any]]:
        """Return one cached K=3 multiset, generating it once per exact state."""
        progress_context = self._teacher_progress_context(
            previous_canonical_action=previous_canonical_action,
            no_progress_repeat_streak=no_progress_repeat_streak,
        )
        cache_fingerprint = self._teacher_cache_fingerprint(
            state_fingerprint,
            progress_context,
        )
        with self._lock:
            self._stats["teacher_cache_lookups"] += 1
            cached = self._state_cache.get(cache_fingerprint)
            if cached is not None:
                self._stats["teacher_cache_hits"] += 1
                return list(cached)
            flight = self._state_flights.get(cache_fingerprint)
            if flight is None:
                flight = Future()
                self._state_flights[cache_fingerprint] = flight
                self._stats["teacher_cache_misses"] += 1
                leader = True
            else:
                self._stats["teacher_cache_singleflight_waits"] += 1
                leader = False
        if not leader:
            return list(flight.result())

        try:
            with ThreadPoolExecutor(max_workers=self.samples) as pool:
                futures = [
                    pool.submit(
                        self._sample_once,
                        messages,
                        tools,
                        index,
                        previous_canonical_action=previous_canonical_action,
                        no_progress_repeat_streak=no_progress_repeat_streak,
                    )
                    for index in range(self.samples)
                ]
                samples = [future.result() for future in futures]
            with self._lock:
                self._append_jsonl(
                    self.cache_path,
                    {
                        "protocol_version": ORACLE_PROTOCOL_VERSION,
                        "state_fingerprint": state_fingerprint,
                        "teacher_cache_fingerprint": cache_fingerprint,
                        "progress_context": progress_context,
                        "model": self.model,
                        "provider": self.provider,
                        "samples": self.samples,
                        "decoding_config": self._teacher_decoding_config(),
                        "teacher_protocol_config": self._teacher_protocol_config(),
                        "teacher_prompt_revision": TEACHER_PROMPT_REVISION,
                        "teacher_prompt_hash": TEACHER_PROMPT_HASH,
                        "native_tool_schema_hash": tool_schema_hash(tools),
                        "teacher_samples": samples,
                    },
                )
                self._state_cache[cache_fingerprint] = samples
                self._stats["teacher_cache_generated_sets"] += 1
                self._state_flights.pop(cache_fingerprint, None)
                flight.set_result(tuple(samples))
            return list(samples)
        except BaseException as exc:
            with self._lock:
                self._state_flights.pop(cache_fingerprint, None)
                flight.set_exception(exc)
            raise

    def _match_pair(self, teacher: str, candidate: str) -> bool:
        if normalize_message(teacher) == normalize_message(candidate):
            with self._lock:
                self._stats["matcher_exact_matches"] += 1
            return True
        fingerprint = _pair_fingerprint(self.matcher_model, teacher, candidate, decoding_config=self.matcher_decoding_config)
        with self._lock:
            cached = self._matcher_cache.get(fingerprint)
            if cached is not None:
                self._stats["matcher_cache_hits"] += 1
                return cached
        prompt = (
            MATCHER_INSTRUCTION
            + "\n"
            + json.dumps(
                {"teacher_message": teacher, "candidate_message": candidate},
                ensure_ascii=False,
            )
        )
        try:
            response = self._post(
                {
                    "model": self.matcher_model,
                    "messages": [{"role": "user", "content": prompt}],
                    **self.matcher_decoding_config,
                },
                prefix="matcher",
            )
            with self._lock:
                self._stats["matcher_requests"] += 1
                self._record_usage(response.get("usage"), prefix="matcher")
            provider_identity = self._accept_provider_identity(
                response,
                prefix="matcher",
            )
            content, _ = self._response_content(response)
            equivalent = _json_object(content).get("equivalent")
            if not isinstance(equivalent, bool):
                raise ValueError("matcher response lacks Boolean 'equivalent'")
        except Exception:
            with self._lock:
                self._stats["matcher_failures"] += 1
            raise
        with self._lock:
            existing = self._matcher_cache.get(fingerprint)
            if existing is not None:
                return existing
            self._matcher_cache[fingerprint] = equivalent
            self._append_jsonl(
                self.matcher_cache_path,
                {
                    "protocol_version": MATCHER_PROTOCOL_VERSION,
                    "pair_fingerprint": fingerprint,
                    "model": self.matcher_model,
                    "provider": self.matcher_provider,
                    "prompt_hash": MATCHER_PROMPT_HASH,
                    "decoding_config": self.matcher_decoding_config,
                    "teacher": normalize_message(teacher),
                    "candidate": normalize_message(candidate),
                    "equivalent": equivalent,
                    "provider_identity": provider_identity,
                    "usage": dict(response.get("usage") or {}),
                },
            )
        return equivalent

    def match_message_pairs(
        self,
        teacher_messages: Sequence[str],
        candidate_messages: Sequence[str],
    ) -> dict[str, Any]:
        """Judge every candidate×teacher pair and sum each Boolean row."""
        if not teacher_messages:
            return {"counts": [0] * len(candidate_messages), "matrix": []}
        pairs = [(candidate, teacher) for candidate in candidate_messages for teacher in teacher_messages]
        unique_pairs = {}
        pair_keys = []
        for candidate, teacher in pairs:
            key = _pair_fingerprint(self.matcher_model, teacher, candidate, decoding_config=self.matcher_decoding_config)
            pair_keys.append(key)
            unique_pairs.setdefault(key, (candidate, teacher))
        with self._lock:
            self._stats["matcher_pair_evaluations"] += len(pairs)
            self._stats["matcher_unique_pairs"] += len(unique_pairs)
        with ThreadPoolExecutor(max_workers=max(1, min(len(unique_pairs), 32))) as pool:
            unique_decisions = dict(
                zip(
                    unique_pairs,
                    pool.map(
                        lambda pair: self._match_pair(pair[1], pair[0]),
                        unique_pairs.values(),
                    ),
                    strict=True,
                )
            )
        decisions = [unique_decisions[key] for key in pair_keys]
        width = len(teacher_messages)
        matrix = [decisions[offset : offset + width] for offset in range(0, len(decisions), width)]
        return {
            "counts": [sum(int(value) for value in row) for row in matrix],
            "matrix": matrix,
        }

    def classify_runtime_failure(
        self,
        *,
        scenario: str,
        task_idx: int,
        task: str,
        failed_action: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Classify one schema-valid AWM 5xx using frozen code evidence."""
        if not self.runtime_judge_enabled or self.runtime_judge_evidence is None:
            raise RuntimeError("AWM runtime judge is disabled")
        evidence = self.runtime_judge_evidence.build(
            scenario=str(scenario),
            task_idx=int(task_idx),
            task=str(task),
            failed_action=failed_action,
            payload=payload,
        )
        fingerprint = runtime_judge_fingerprint(
            model=self.runtime_judge_model,
            decoding_config=self.runtime_judge_decoding_config,
            evidence=evidence,
        )
        with self._lock:
            self._stats["runtime_judge_cache_lookups"] += 1
            cached = self._runtime_judge_cache.get(fingerprint)
            if cached is not None:
                self._stats["runtime_judge_cache_hits"] += 1
                self._stats[_RUNTIME_JUDGE_CLASS_STATS[cached["error_class"]]] += 1
                return {
                    **cached,
                    "evidence_fingerprint": fingerprint,
                    "cache_hit": True,
                }
            flight = self._runtime_judge_flights.get(fingerprint)
            if flight is None:
                flight = Future()
                self._runtime_judge_flights[fingerprint] = flight
                self._stats["runtime_judge_cache_misses"] += 1
                leader = True
            else:
                self._stats["runtime_judge_cache_singleflight_waits"] += 1
                leader = False
        if not leader:
            verdict = dict(flight.result())
            with self._lock:
                self._stats[_RUNTIME_JUDGE_CLASS_STATS[verdict["error_class"]]] += 1
            return {
                **verdict,
                "evidence_fingerprint": fingerprint,
                "cache_hit": True,
            }

        try:
            response = self._post(
                {
                    "model": self.runtime_judge_model,
                    "messages": [
                        {"role": "system", "content": RUNTIME_JUDGE_INSTRUCTION},
                        {
                            "role": "user",
                            "content": json.dumps(
                                evidence,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        },
                    ],
                    **self.runtime_judge_decoding_config,
                },
                prefix="runtime_judge",
            )
            with self._lock:
                self._stats["runtime_judge_requests"] += 1
                self._record_usage(
                    response.get("usage"),
                    prefix="runtime_judge",
                )
            provider_identity = self._accept_provider_identity(
                response,
                prefix="runtime_judge",
            )
            content, _ = self._response_content(response)
            verdict = validate_runtime_judge_verdict(_json_object(content))
            with self._lock:
                self._runtime_judge_cache[fingerprint] = verdict
                self._stats[_RUNTIME_JUDGE_CLASS_STATS[verdict["error_class"]]] += 1
                self._append_jsonl(
                    self.runtime_judge_cache_path,
                    {
                        "protocol_version": RUNTIME_JUDGE_PROTOCOL_VERSION,
                        "evidence_fingerprint": fingerprint,
                        "model": self.runtime_judge_model,
                        "provider": self.runtime_judge_provider,
                        "prompt_hash": RUNTIME_JUDGE_PROMPT_HASH,
                        "decoding_config": self.runtime_judge_decoding_config,
                        "evidence": evidence,
                        "verdict": verdict,
                        "provider_identity": provider_identity,
                        "usage": dict(response.get("usage") or {}),
                    },
                )
                self._runtime_judge_flights.pop(fingerprint, None)
                flight.set_result(dict(verdict))
            return {
                **verdict,
                "evidence_fingerprint": fingerprint,
                "cache_hit": False,
            }
        except BaseException as exc:
            with self._lock:
                self._stats["runtime_judge_failures"] += 1
                self._runtime_judge_flights.pop(fingerprint, None)
                flight.set_exception(exc)
            raise

    def classify_envscaler_runtime_failure(
        self,
        *,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Classify one restored EnvScaler tool exception cache-first."""
        if not self.runtime_judge_enabled:
            raise RuntimeError("EnvScaler runtime judge is disabled")
        evidence = dict(evidence)
        fingerprint = envscaler_runtime_judge_fingerprint(
            model=self.runtime_judge_model,
            decoding_config=self.runtime_judge_decoding_config,
            evidence=evidence,
        )
        with self._lock:
            self._stats["envscaler_runtime_judge_cache_lookups"] += 1
            cached = self._envscaler_runtime_judge_cache.get(fingerprint)
            if cached is not None:
                self._stats["envscaler_runtime_judge_cache_hits"] += 1
                self._stats[_ENVSCALER_RUNTIME_JUDGE_CLASS_STATS[cached["error_class"]]] += 1
                return {
                    **cached,
                    "evidence_fingerprint": fingerprint,
                    "cache_hit": True,
                }
            flight = self._envscaler_runtime_judge_flights.get(fingerprint)
            if flight is None:
                flight = Future()
                self._envscaler_runtime_judge_flights[fingerprint] = flight
                self._stats["envscaler_runtime_judge_cache_misses"] += 1
                leader = True
            else:
                self._stats["envscaler_runtime_judge_cache_singleflight_waits"] += 1
                leader = False
        if not leader:
            verdict = dict(flight.result())
            with self._lock:
                self._stats[_ENVSCALER_RUNTIME_JUDGE_CLASS_STATS[verdict["error_class"]]] += 1
            return {
                **verdict,
                "evidence_fingerprint": fingerprint,
                "cache_hit": True,
            }

        try:
            response = self._post(
                {
                    "model": self.runtime_judge_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": ENVSCALER_RUNTIME_JUDGE_INSTRUCTION,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                evidence,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        },
                    ],
                    **self.runtime_judge_decoding_config,
                },
                prefix="envscaler_runtime_judge",
            )
            with self._lock:
                self._stats["envscaler_runtime_judge_requests"] += 1
                self._record_usage(
                    response.get("usage"),
                    prefix="envscaler_runtime_judge",
                )
            provider_identity = self._accept_provider_identity(
                response,
                prefix="envscaler_runtime_judge",
            )
            content, _ = self._response_content(response)
            verdict = validate_envscaler_runtime_judge_verdict(_json_object(content))
            with self._lock:
                self._envscaler_runtime_judge_cache[fingerprint] = verdict
                self._stats[_ENVSCALER_RUNTIME_JUDGE_CLASS_STATS[verdict["error_class"]]] += 1
                self._append_jsonl(
                    self.runtime_judge_cache_path,
                    {
                        "judge_scope": ENVSCALER_RUNTIME_JUDGE_SCOPE,
                        "protocol_version": (ENVSCALER_RUNTIME_JUDGE_PROTOCOL_VERSION),
                        "evidence_fingerprint": fingerprint,
                        "model": self.runtime_judge_model,
                        "provider": self.runtime_judge_provider,
                        "prompt_hash": ENVSCALER_RUNTIME_JUDGE_PROMPT_HASH,
                        "decoding_config": self.runtime_judge_decoding_config,
                        "evidence": evidence,
                        "verdict": verdict,
                        "provider_identity": provider_identity,
                        "usage": dict(response.get("usage") or {}),
                    },
                )
                self._envscaler_runtime_judge_flights.pop(
                    fingerprint,
                    None,
                )
                flight.set_result(dict(verdict))
            return {
                **verdict,
                "evidence_fingerprint": fingerprint,
                "cache_hit": False,
            }
        except BaseException as exc:
            with self._lock:
                self._stats["envscaler_runtime_judge_failures"] += 1
                self._envscaler_runtime_judge_flights.pop(fingerprint, None)
                flight.set_exception(exc)
            raise

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            stats = dict(self._stats)
            lookups = stats["teacher_cache_lookups"]
            stats["teacher_cache_hit_rate"] = stats["teacher_cache_hits"] / lookups if lookups else 0.0
            requests = stats["teacher_requests"]
            parallel = stats["teacher_parallel_responses"]
            stats["teacher_parallel_response_rate"] = stats["teacher_parallel_responses"] / requests if requests else 0.0
            stats["teacher_parallel_alternative_available_rate"] = stats["teacher_parallel_alternative_available"] / parallel if parallel else 0.0
            stats["teacher_parallel_fallback_rate"] = stats["teacher_parallel_fallback_applied"] / parallel if parallel else 0.0
            return stats


def build_teacher_messages(
    visible_chat: Sequence[Mapping[str, Any]],
    *,
    privileged_context: Mapping[str, Any] | None = None,
    use_privileged_context: bool = False,
) -> list[dict[str, Any]]:
    """Build teacher messages from the exact student-visible native history."""
    chat = visible_chat
    if not chat:
        raise ValueError("AWM expert requires a non-empty chat state")
    messages = []
    for message in chat:
        copied = dict(message)
        if copied.get("role") == "assistant" and copied.get("tool_calls"):
            copied.setdefault("reasoning_content", "")
        messages.append(copied)
    if messages[0].get("role") != "system":
        raise ValueError("AWM expert requires a leading system message")
    messages[0] = dict(messages[0])
    messages[0]["content"] = f"{messages[0].get('content') or ''}\n\n{TEACHER_SINGLE_ACTION_INSTRUCTION}"
    if use_privileged_context:
        if not privileged_context:
            raise ValueError("privileged teacher context was enabled without structured context")
        messages[0]["content"] = f"{messages[0].get('content') or ''}\n\nPRIVILEGED TEACHER CONTEXT:\n" + json.dumps(privileged_context, ensure_ascii=False, sort_keys=True)
    return messages


@ray.remote(max_concurrency=64)
class DeepSeekAWMOracleActor:
    """Central actor shared by AWM workers; API secrets stay out of datasets."""

    def __init__(self, **kwargs):
        self.client = DeepSeekAWMOracleClient(**kwargs)

    async def sample_multiset(self, **kwargs):
        return await asyncio.to_thread(self.client.sample_multiset, **kwargs)

    async def match_message_pairs(self, teacher_messages, candidate_messages):
        return await asyncio.to_thread(self.client.match_message_pairs, teacher_messages, candidate_messages)

    async def classify_runtime_failure(self, **kwargs):
        return await asyncio.to_thread(self.client.classify_runtime_failure, **kwargs)

    async def classify_envscaler_runtime_failure(self, **kwargs):
        return await asyncio.to_thread(
            self.client.classify_envscaler_runtime_failure,
            **kwargs,
        )

    def get_stats(self):
        return self.client.stats()
