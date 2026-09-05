import asyncio

import pytest

from agent_system.environments.env_package.awm.runtime import terminal_judge


def test_deepseek_transport_injects_native_thinking_and_retries_judge_errors(
    monkeypatch,
):
    from agent_world_model_env.server import verifier

    original_client = verifier.AsyncOpenAI
    original_judge = verifier.run_llm_judge
    original_flag = getattr(verifier, "_verl_agent_terminal_judge_transport", None)
    client_kwargs = []
    request_kwargs = []
    judge_calls = 0

    class FakeClient:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            request_kwargs.append(kwargs)
            return object()

    async def fake_judge(*args, **kwargs):
        nonlocal judge_calls
        judge_calls += 1
        client = verifier.AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key="secret",
        )
        await client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[],
            max_completion_tokens=4096,
        )
        if judge_calls == 1:
            return "judge_error", {"error": "bad json"}
        return "incomplete", {"classification": "incomplete"}

    monkeypatch.setenv("AWM_TERMINAL_JUDGE_MAX_RETRIES", "3")
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_MAX_TOKENS", "8192")
    try:
        verifier.AsyncOpenAI = FakeClient
        verifier.run_llm_judge = fake_judge
        if hasattr(verifier, "_verl_agent_terminal_judge_transport"):
            del verifier._verl_agent_terminal_judge_transport

        protocol = terminal_judge.install_terminal_judge_transport()
        label, payload = asyncio.run(verifier.run_llm_judge())

        assert protocol["provider"] == "deepseek"
        assert protocol["decoding_config"]["thinking"] == {"type": "enabled"}
        assert label == "incomplete"
        assert payload == {"classification": "incomplete"}
        assert judge_calls == 2
        assert all(kwargs["max_retries"] == 0 for kwargs in client_kwargs)
        assert all(kwargs["timeout"] == 300 for kwargs in client_kwargs)
        assert all("max_completion_tokens" not in kwargs for kwargs in request_kwargs)
        assert all(kwargs["max_tokens"] == 8192 for kwargs in request_kwargs)
        assert all(kwargs["extra_body"] == {"thinking": {"type": "enabled"}, "reasoning_effort": "max"} for kwargs in request_kwargs)
    finally:
        verifier.AsyncOpenAI = original_client
        verifier.run_llm_judge = original_judge
        if original_flag is None:
            if hasattr(verifier, "_verl_agent_terminal_judge_transport"):
                del verifier._verl_agent_terminal_judge_transport
        else:
            verifier._verl_agent_terminal_judge_transport = original_flag


def test_terminal_judge_protocol_uses_dashscope_native_decoding(monkeypatch):
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_PROVIDER", "dashscope")
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_MODEL", "qwen3.7-flash")
    protocol = terminal_judge.terminal_judge_protocol()

    assert protocol["provider"] == "dashscope"
    assert protocol["model"] == "qwen3.7-flash"
    assert protocol["decoding_config"] == {
        "enable_thinking": True,
        "thinking_budget": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
    }


def test_terminal_judge_protocol_uses_zai_native_decoding(monkeypatch):
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_PROVIDER", "zai")
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_MODEL", "glm-5.3-flash")
    protocol = terminal_judge.terminal_judge_protocol()

    assert protocol["provider"] == "zai"
    assert protocol["model"] == "glm-5.3-flash"
    assert protocol["decoding_config"] == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "max",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    }


@pytest.mark.parametrize(
    ("provider", "model", "temperature", "extra_body"),
    [
        (
            "dashscope",
            "qwen3.7-flash",
            0.6,
            {"enable_thinking": True, "thinking_budget": 4096},
        ),
        (
            "zai",
            "glm-5.3-flash",
            1.0,
            {
                "thinking": {"type": "enabled", "clear_thinking": False},
                "reasoning_effort": "max",
            },
        ),
    ],
)
def test_non_deepseek_transport_injects_provider_native_controls(
    monkeypatch,
    provider,
    model,
    temperature,
    extra_body,
):
    from agent_world_model_env.server import verifier

    original_client = verifier.AsyncOpenAI
    original_judge = verifier.run_llm_judge
    original_flag = getattr(verifier, "_verl_agent_terminal_judge_transport", None)
    request_kwargs = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            request_kwargs.append(kwargs)
            return object()

    async def fake_judge(*args, **kwargs):
        client = verifier.AsyncOpenAI(base_url="https://example.test", api_key="secret")
        await client.chat.completions.create(
            model=model,
            messages=[],
            max_completion_tokens=4096,
        )
        return "complete", {"classification": "complete"}

    monkeypatch.setenv("AWM_TERMINAL_JUDGE_PROVIDER", provider)
    monkeypatch.setenv("AWM_TERMINAL_JUDGE_MODEL", model)
    try:
        verifier.AsyncOpenAI = FakeClient
        verifier.run_llm_judge = fake_judge
        if hasattr(verifier, "_verl_agent_terminal_judge_transport"):
            del verifier._verl_agent_terminal_judge_transport

        terminal_judge.install_terminal_judge_transport()
        label, _ = asyncio.run(verifier.run_llm_judge())

        assert label == "complete"
        assert request_kwargs == [
            {
                "model": model,
                "messages": [],
                "max_tokens": 8192,
                "temperature": temperature,
                "top_p": 0.95,
                "extra_body": extra_body,
            }
        ]
    finally:
        verifier.AsyncOpenAI = original_client
        verifier.run_llm_judge = original_judge
        if original_flag is None:
            if hasattr(verifier, "_verl_agent_terminal_judge_transport"):
                del verifier._verl_agent_terminal_judge_transport
        else:
            verifier._verl_agent_terminal_judge_transport = original_flag
