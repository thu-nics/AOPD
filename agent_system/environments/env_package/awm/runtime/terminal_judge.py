"""Provider-aware transport for AWM's official SQL-augmented judge."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .judge import runtime_judge_decoding_config

TERMINAL_JUDGE_PROTOCOL_VERSION = 2
TERMINAL_JUDGE_ENDPOINT = "/awm-terminal-judge"
DEFAULT_TERMINAL_JUDGE_PROVIDER = "deepseek"
DEFAULT_TERMINAL_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_TERMINAL_JUDGE_API_BASE = "https://api.deepseek.com"
SUPPORTED_TERMINAL_JUDGE_PROVIDERS = frozenset({"deepseek", "dashscope", "zai", "openai-compatible", "vllm"})


def terminal_judge_decoding_config(
    *,
    provider: str,
    reasoning_effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Use the same provider-native reasoning protocol as runtime judging."""
    config = runtime_judge_decoding_config(
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    config.pop("response_format", None)
    config.pop("stream", None)
    return config


def terminal_judge_protocol() -> dict[str, Any]:
    """Return the non-secret server-side judge identity."""
    provider = os.environ.get(
        "AWM_TERMINAL_JUDGE_PROVIDER",
        DEFAULT_TERMINAL_JUDGE_PROVIDER,
    ).lower()
    if provider not in SUPPORTED_TERMINAL_JUDGE_PROVIDERS:
        raise ValueError(f"unsupported AWM terminal judge provider: {provider!r}")
    reasoning_effort = os.environ.get(
        "AWM_TERMINAL_JUDGE_REASONING_EFFORT",
        "max",
    )
    max_tokens = int(os.environ.get("AWM_TERMINAL_JUDGE_MAX_TOKENS", "8192"))
    return {
        "protocol_version": TERMINAL_JUDGE_PROTOCOL_VERSION,
        "provider": provider,
        "model": os.environ.get("AWM_TERMINAL_JUDGE_MODEL", DEFAULT_TERMINAL_JUDGE_MODEL),
        "api_base": os.environ.get("AWM_TERMINAL_JUDGE_API_BASE", DEFAULT_TERMINAL_JUDGE_API_BASE),
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "decoding_config": terminal_judge_decoding_config(
            provider=provider,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        ),
        "timeout_seconds": float(os.environ.get("AWM_TERMINAL_JUDGE_TIMEOUT_SECONDS", "300")),
        "max_retries": int(os.environ.get("AWM_TERMINAL_JUDGE_MAX_RETRIES", "5")),
        "upstream_semantics": "OpenEnv AWM SQL evidence plus official LLM judge prompt",
    }


def fetch_terminal_judge_protocol(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch and minimally validate the AWM server's terminal-judge identity."""
    url = str(base_url).rstrip("/") + TERMINAL_JUDGE_ENDPOINT
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("protocol_version") != TERMINAL_JUDGE_PROTOCOL_VERSION:
        raise RuntimeError("AWM server terminal-judge protocol version mismatch")
    if payload.get("provider") not in SUPPORTED_TERMINAL_JUDGE_PROVIDERS:
        raise RuntimeError("AWM server returned an unsupported terminal-judge provider")
    return payload


def install_terminal_judge_transport() -> dict[str, Any]:
    """Patch only the OpenAI transport used by the pinned OpenEnv judge.

    OpenEnv remains responsible for SQL evidence construction, its official
    judge prompt, response parsing, and reward labels. The pinned transport
    uses generic OpenAI parameters that do not expose provider-native thinking
    controls, so this wrapper injects them without editing the external checkout.
    """
    from agent_world_model_env.server import verifier

    if getattr(verifier, "_verl_agent_terminal_judge_transport", False):
        return terminal_judge_protocol()

    original_client = verifier.AsyncOpenAI
    protocol = terminal_judge_protocol()

    class ProviderJudgeClient:
        def __init__(self, *, base_url: str, api_key: str, **kwargs: Any):
            self._client = original_client(
                base_url=base_url,
                api_key=api_key,
                timeout=protocol["timeout_seconds"],
                max_retries=0,
                **kwargs,
            )
            self.chat = self
            self.completions = self

        async def create(self, **kwargs: Any):
            # The official OpenEnv judge supplies max_completion_tokens=4096.
            # Provider-native endpoints use max_tokens and need a larger budget
            # when thinking is enabled.
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = protocol["max_tokens"]
            decoding_config = dict(protocol["decoding_config"])
            decoding_config.pop("max_tokens", None)
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            for name in ("temperature", "top_p"):
                if name in decoding_config:
                    kwargs[name] = decoding_config.pop(name)
            extra_body.update(decoding_config)
            kwargs["extra_body"] = extra_body
            return await self._client.chat.completions.create(**kwargs)

    original_judge = verifier.run_llm_judge

    async def retrying_judge(*args: Any, **kwargs: Any):
        attempts = max(1, int(protocol["max_retries"]))
        last = ("judge_error", {"error": "terminal judge was not attempted"})
        for attempt in range(1, attempts + 1):
            last = await original_judge(*args, **kwargs)
            if last[0] != "judge_error":
                return last
            if isinstance(last[1], dict):
                last[1]["verl_agent_judge_attempt"] = attempt
        return last

    verifier.AsyncOpenAI = ProviderJudgeClient
    verifier.run_llm_judge = retrying_judge
    verifier._verl_agent_terminal_judge_transport = True
    return protocol


# Historical import retained for downstream callers.
install_deepseek_terminal_judge_transport = install_terminal_judge_transport
