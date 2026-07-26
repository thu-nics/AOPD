"""OpenRouter-backed oracle policy for Tau Bench VPR."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
import logging
import os
from pathlib import Path
import random
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import ray

from .actions import ParsedAction, deduplicate_actions, parse_action

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ORACLE_PROTOCOL_VERSION = 3
logger = logging.getLogger(__name__)


class OpenRouterOracleClient:
    """Thread-safe client with deterministic seeds and an append-only state cache."""

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-v4-flash",
        api_key_env: str = "OPENROUTER_API_KEY",
        samples: int = 3,
        reasoning_effort: str = "xhigh",
        max_tokens: int = 4096,
        cache_path: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
        max_concurrent_requests: int = 24,
    ):
        if samples <= 0:
            raise ValueError("oracle samples must be positive")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing required environment variable {api_key_env}")
        self.api_key = api_key
        self.model = model
        self.samples = int(samples)
        self.reasoning_effort = reasoning_effort
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "retries": 0,
            "failures": 0,
            "semantic_exact_matches": 0,
            "semantic_retries": 0,
            "semantic_failures": 0,
            "semantic_batch_requests": 0,
            "semantic_batch_failures": 0,
            "semantic_individual_requests": 0,
            "semantic_individual_failures": 0,
            "parallel_tool_calls_truncated": 0,
        }
        self._load_cache()

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
                    or int(record.get("samples", -1)) != self.samples
                    or record.get("reasoning_effort") != self.reasoning_effort
                    or int(record.get("max_tokens", -1)) != self.max_tokens
                ):
                    continue
                self._cache[str(record["state_fingerprint"])] = list(record["oracle_actions"])

    def _append_cache(self, state_fingerprint: str, actions: list[dict[str, Any]]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "protocol_version": ORACLE_PROTOCOL_VERSION,
            "state_fingerprint": state_fingerprint,
            "model": self.model,
            "samples": self.samples,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "oracle_actions": actions,
        }
        encoded = (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        with self.cache_path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell() > 0:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(encoded)
            handle.flush()

    @staticmethod
    def _seed(state_fingerprint: str, sample_index: int) -> int:
        digest = hashlib.sha256(f"{state_fingerprint}:{sample_index}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            OPENROUTER_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/verl-agent/verl-agent",
                "X-Title": "VPR Tau Bench oracle",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._request_slots:
                    with urlopen(request, timeout=self.timeout_seconds) as response:
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
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} attempts: {last_error}")

    def _response_action(self, response: dict[str, Any]) -> ParsedAction:
        choices = response.get("choices") or []
        if not choices:
            return ParsedAction(kind="invalid", error="oracle response has no choices")
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) > 1:
                # Some OpenRouter providers ignore parallel_tool_calls=False.
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
        return parse_action(message.get("content"))

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
            "max_tokens": self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort},
        }
        return self._response_action(self._post(payload))

    def sample_oracle_set(
        self,
        *,
        state_fingerprint: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._cache.get(state_fingerprint)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return list(cached)

        seeds = [self._seed(state_fingerprint, index) for index in range(self.samples)]
        with ThreadPoolExecutor(max_workers=self.samples) as pool:
            futures = [
                pool.submit(self._sample_once, messages=messages, tools=tools, seed=seed)
                for seed in seeds
            ]
            actions = [future.result() for future in futures]
        deduplicated = deduplicate_actions(actions)
        with self._lock:
            existing = self._cache.get(state_fingerprint)
            if existing is not None:
                return list(existing)
            self._cache[state_fingerprint] = deduplicated
            self._append_cache(state_fingerprint, deduplicated)
        return list(deduplicated)

    def match_messages(
        self,
        oracle_messages: list[str],
        candidate_messages: list[str],
    ) -> list[bool]:
        if not candidate_messages:
            return []
        if not oracle_messages:
            return [False] * len(candidate_messages)

        def normalize(value: str) -> str:
            return " ".join(value.split()).casefold()

        def parse_json_object(content: str) -> dict[str, Any]:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                start, end = content.find("{"), content.rfind("}")
                if start < 0 or end <= start:
                    raise
                parsed = json.loads(content[start : end + 1])
            if not isinstance(parsed, dict):
                raise TypeError("semantic matcher response must be a JSON object")
            return parsed

        def request_content(prompt: str, max_tokens: int) -> str:
            response = self._post(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "reasoning": {"enabled": False},
                }
            )
            return str(
                (response.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        oracle_normalized = {normalize(value) for value in oracle_messages}
        results = [normalize(value) in oracle_normalized for value in candidate_messages]
        with self._lock:
            self._stats["semantic_exact_matches"] += sum(results)

        unique_unresolved = []
        positions_by_message: dict[str, list[int]] = {}
        for index, (candidate, matched) in enumerate(zip(candidate_messages, results)):
            if matched:
                continue
            key = normalize(candidate)
            if key not in positions_by_message:
                unique_unresolved.append(candidate)
                positions_by_message[key] = []
            positions_by_message[key].append(index)
        if not unique_unresolved:
            return results

        batch_prompt = (
            "Judge whether each candidate message has the same immediate conversational "
            "intent and materially equivalent information as at least one oracle message. "
            f"Return JSON exactly as {{\"matches\":[true,...]}} with exactly "
            f"{len(unique_unresolved)} JSON boolean value(s), in candidate order.\n"
            + json.dumps(
                {
                    "oracle_messages": oracle_messages,
                    "candidate_messages": unique_unresolved,
                },
                ensure_ascii=False,
            )
        )
        batch_content = ""
        with self._lock:
            self._stats["semantic_batch_requests"] += 1
        try:
            batch_content = request_content(batch_prompt, max_tokens=512)
            parsed = parse_json_object(batch_content)
            values = parsed.get("matches")
            if not isinstance(values, list) or len(values) != len(unique_unresolved):
                raise ValueError(
                    "semantic matcher returned the wrong number of decisions"
                )
            if any(not isinstance(value, bool) for value in values):
                raise ValueError("semantic matcher decisions must be JSON booleans")
            for candidate, value in zip(unique_unresolved, values):
                for index in positions_by_message[normalize(candidate)]:
                    results[index] = value
            return results
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            with self._lock:
                self._stats["semantic_batch_failures"] += 1
            logger.warning(
                "Semantic batch matcher failed; retrying %d unresolved candidate "
                "message(s) independently. Error: %s. Response: %r",
                len(unique_unresolved),
                exc,
                batch_content[:512],
            )

        def match_one(item: tuple[int, str]):
            candidate_index, candidate = item
            prompt = (
                "Judge whether the candidate message has the same immediate "
                "conversational intent and materially equivalent information as at "
                "least one oracle message. Return JSON exactly as "
                '{"match":true} or {"match":false}.\n'
                + json.dumps(
                    {
                        "oracle_messages": oracle_messages,
                        "candidate_message": candidate,
                    },
                    ensure_ascii=False,
                )
            )
            content = ""
            try:
                content = request_content(prompt, max_tokens=128)
                parsed = parse_json_object(content)
                value = parsed.get("match")
                if not isinstance(value, bool):
                    raise ValueError(
                        "individual semantic matcher decision must be a JSON boolean"
                    )
                return candidate_index, value, None, content
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                return candidate_index, False, exc, content

        with ThreadPoolExecutor(max_workers=len(unique_unresolved)) as pool:
            individual_results = list(pool.map(match_one, enumerate(unique_unresolved)))

        failure_count = sum(error is not None for _, _, error, _ in individual_results)
        with self._lock:
            self._stats["semantic_retries"] += len(unique_unresolved)
            self._stats["semantic_individual_requests"] += len(unique_unresolved)
            self._stats["semantic_individual_failures"] += failure_count
            self._stats["semantic_failures"] += failure_count

        for candidate_index, value, error, content in individual_results:
            candidate = unique_unresolved[candidate_index]
            if error is not None:
                logger.warning(
                    "Individual semantic matcher failed for candidate %d; treating it "
                    "as a non-match. Error: %s. Response: %r",
                    candidate_index,
                    error,
                    content[:512],
                )
            for index in positions_by_message[normalize(candidate)]:
                results[index] = value
        return results

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


def build_expert_messages(
    *,
    policy: str,
    task: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    criteria = task.get("evaluation_criteria") or {}
    reference_actions = criteria.get("actions") or []
    hidden_context = {
        "task_id": task.get("id"),
        "user_scenario": task.get("user_scenario"),
        "reference_resolution_actions": reference_actions,
        "evaluation_criteria": criteria,
    }
    system = (
        "You are the oracle policy for a Tau Bench customer-service task. Choose exactly "
        "one best action for the current state. Return either one tool call or one message "
        "to the user; never return multiple actions. Follow the domain policy. The hidden "
        "task context and reference trajectory are privileged guidance, not a requirement "
        "to copy a unique path.\n\nDOMAIN POLICY:\n"
        f"{policy}\n\nHIDDEN TASK CONTEXT:\n"
        f"{json.dumps(hidden_context, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, *history]


@ray.remote(max_concurrency=32)
class OpenRouterOracleActor:
    """Central async Ray actor shared by all Tau environment workers."""

    def __init__(self, **kwargs):
        self.client = OpenRouterOracleClient(**kwargs)

    async def sample_oracle_set(self, **kwargs):
        return await asyncio.to_thread(self.client.sample_oracle_set, **kwargs)

    async def match_messages(self, oracle_messages, candidate_messages):
        return await asyncio.to_thread(
            self.client.match_messages, oracle_messages, candidate_messages
        )

    def get_stats(self):
        return self.client.stats()
