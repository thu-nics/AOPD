"""DeepSeek user simulator for EnvScaler conversation tasks."""

from __future__ import annotations

import json
import os
import random
import re
import time
from http.client import HTTPException
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://api.deepseek.com"
STOP = "###STOP###"

USER_SYSTEM_PROMPT = """You are a human user interacting with an assistant that can use tools.

Your private task goal is:
{task}

Reveal the task naturally and incrementally. Never invent requirements or facts that are
not present in the task. Preserve exact names, identifiers, dates, and constraints. Do not
repeat information unless the assistant asks for clarification. If the assistant offers
choices, select only an option consistent with the task.

When every part of the task has actually been completed, reply with exactly ###STOP### and
nothing else. Otherwise reply with only the next natural user message. Do not output
reasoning, analysis, labels, markdown headings, or a Thought/Reply wrapper."""


class DeepSeekUserSimulator:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 1.0,
        reasoning_enabled: bool = False,
        timeout_seconds: float = 300,
        max_retries: int = 3,
        request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ):
        self.model = str(model)
        self.url = str(api_base).rstrip("/") + "/chat/completions"
        self.api_key_env = str(api_key_env)
        self.temperature = float(temperature)
        self.reasoning_enabled = bool(reasoning_enabled)
        if self.temperature != 1.0:
            raise ValueError("EnvScaler requires user-simulator temperature=1")
        if self.reasoning_enabled:
            raise ValueError("EnvScaler requires disabled user-simulator reasoning")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        if self.max_retries <= 0:
            raise ValueError("EnvScaler user simulator max_retries must be positive")
        self.request_fn = request_fn
        self.messages: list[dict[str, str]] = []
        self.raw_messages: list[dict[str, str]] = []

    @staticmethod
    def parse_reply(content: str) -> str:
        value = str(content or "").strip()
        if STOP in value:
            return STOP
        legacy = re.search(r"#\s*Reply\s*:\s*(.*)", value, flags=re.DOTALL)
        if legacy:
            value = legacy.group(1).strip()
        if not value:
            raise ValueError("EnvScaler user simulator returned an empty reply")
        return value

    def _post(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(payload)
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing EnvScaler user simulator key {self.api_key_env}")
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode())
            except (
                OSError,
                TimeoutError,
                HTTPException,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                if isinstance(exc, HTTPError):
                    try:
                        body = exc.read().decode(errors="replace")
                    except Exception:
                        body = ""
                    last_error = RuntimeError(f"{exc}; response_body={body[:2048]!r}")
                else:
                    last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(8.0, 2.0**attempt) + random.random() * 0.1)
        raise RuntimeError(f"EnvScaler user simulator request failed after {self.max_retries} attempts: {last_error}")

    def _infer(self) -> str:
        response = self._post(
            {
                "model": self.model,
                "messages": list(self.messages),
                "thinking": {"type": "disabled"},
                "temperature": self.temperature,
                "stream": False,
            }
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("EnvScaler user simulator response has no choices")
        message = choices[0].get("message") or {}
        raw = str(message.get("content") or "")
        reply = self.parse_reply(raw)
        self.raw_messages.append(
            {
                "role": "assistant",
                "content": raw,
            }
        )
        self.messages.append({"role": "assistant", "content": raw})
        return reply

    def start(self, task: str) -> str:
        self.messages = [
            {
                "role": "system",
                "content": USER_SYSTEM_PROMPT.format(task=str(task)),
            },
            {
                "role": "user",
                "content": "[Agent] Hi! How can I help you today?",
            },
        ]
        self.raw_messages = list(self.messages)
        return self._infer()

    def reply(self, agent_message: str) -> str:
        content = f"[Agent] {agent_message}"
        self.messages.append({"role": "user", "content": content})
        self.raw_messages.append({"role": "user", "content": content})
        return self._infer()
