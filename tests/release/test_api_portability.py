"""API portability regressions; no real endpoint or credentials are used."""

import copy
import os
import re
import subprocess
from pathlib import Path

import pytest


def _teacher_payload():
    return {
        "model": "custom-teacher",
        "messages": [{"role": "user", "content": "Look up order 42."}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "seed": 123,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_generic_teacher_keeps_sampling_and_tools_without_vllm_extensions():
    from aopd.providers import adapt_chat_payload

    original = _teacher_payload()
    before = copy.deepcopy(original)
    result = adapt_chat_payload(original, "openai-compatible")

    expected = {key: value for key, value in before.items() if key not in {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}}
    assert result == expected
    assert original == before


def test_vllm_teacher_retains_requested_template_and_sampling():
    from aopd.providers import adapt_chat_payload

    original = _teacher_payload()
    assert adapt_chat_payload(original, "vllm") == original


@pytest.mark.parametrize(
    ("provider", "decoding"),
    [
        ("deepseek", {"thinking": {"type": "enabled"}, "reasoning_effort": "max", "max_tokens": 8192, "stream": False}),
        ("dashscope", {"enable_thinking": True, "thinking_budget": 4096, "temperature": 0.6, "top_p": 0.95, "max_tokens": 8192, "stream": False}),
        ("zai", {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "max", "temperature": 1.0, "top_p": 0.95, "max_tokens": 8192, "stream": False}),
    ],
)
def test_native_provider_decoding_is_not_rewritten(provider, decoding):
    from aopd.providers import adapt_chat_payload

    original = {"model": "custom-teacher", "messages": [{"role": "user", "content": "Hello"}], **decoding}
    assert adapt_chat_payload(original, provider) == original


@pytest.mark.parametrize("provider", ["deepseek", "dashscope", "zai"])
def test_qwen_thinking_request_becomes_provider_native(provider):
    from aopd.providers import adapt_chat_payload

    result = adapt_chat_payload(_teacher_payload(), provider)
    assert not {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}.intersection(result)
    if provider == "dashscope":
        assert result["enable_thinking"] is True
        assert "thinking" not in result
        assert result["temperature"] == 0.6
    else:
        assert result["thinking"]["type"] == "enabled"
        assert "enable_thinking" not in result
    if provider == "deepseek":
        assert "temperature" not in result
        assert "top_p" not in result
    assert result["tools"] == _teacher_payload()["tools"]


def test_generic_envscaler_user_can_complete_a_conversation():
    from agent_system.environments.env_package.envscaler.user_simulator import STOP, ProviderUserSimulator

    payloads = []
    replies = iter(["Please look up order 42.", STOP])

    def request(payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": next(replies), "reasoning_content": "private reasoning"}}]}

    simulator = ProviderUserSimulator(
        provider="openai-compatible",
        model="custom-user",
        api_base="https://user.example/v1",
        reasoning_enabled=False,
        request_fn=request,
    )
    assert simulator.start("Look up order 42") == "Please look up order 42."
    assert simulator.reply("The order has shipped.") == STOP
    for payload in payloads:
        assert payload["model"] == "custom-user"
        assert payload["temperature"] == 1.0
        assert not {"thinking", "enable_thinking", "chat_template_kwargs", "thinking_budget"}.intersection(payload)
        assert "private reasoning" not in str(payload)


def test_tau_remote_chat_api_does_not_require_model_listing():
    launcher = Path(__file__).parents[2] / "examples" / "tau_bench" / "train" / "run.sh"
    check = re.search(r"(?ms)^check_openai_endpoint\(\) \{\n.*?^\}", launcher.read_text())
    assert check is not None, "Keep the endpoint check independently executable"
    script = (
        """set -euo pipefail
curl() {
    for argument in "$@"; do
        case "$argument" in
            */models) return 22 ;;
        esac
    done
    return 0
}
"""
        + check.group()
        + "\ncheck_openai_endpoint 'remote user' 'https://user.example/v1' TEST_ONLY_API_KEY openai-compatible\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "TEST_ONLY_API_KEY": "test-only"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tau_teacher_generic_transport_and_provider_cache_isolation(monkeypatch, tmp_path):
    from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient

    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test-only")
    cache = str(tmp_path / "teacher.jsonl")
    client = TauTeacherClient(provider="openai-compatible", cache_path=cache, matcher_enabled=False)
    seen = []

    def post(payload):
        seen.append(payload)
        return {"choices": [{"message": {"content": "Hello"}}]}

    monkeypatch.setattr(client, "_post", post)
    client.sample_multiset(state_fingerprint="state", messages=[{"role": "user", "content": "Hello"}], tools=[])
    assert seen and all("chat_template_kwargs" not in p and "top_k" not in p for p in seen)
    other = TauTeacherClient(provider="vllm", cache_path=cache, matcher_enabled=False)
    assert other.stats()["cache_records_loaded"] == 0
    imported = TauTeacherClient(provider="vllm", teacher_cache_import_paths=[cache], matcher_enabled=False)
    assert imported._import_samples("state", [{"role": "user", "content": "Hello"}], [], "public") == []
