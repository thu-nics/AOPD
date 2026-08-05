import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_system.multi_turn_rollout import rollout_loop
from agent_system.multi_turn_rollout.rollout_loop import (
    AWMContextBudgetExceeded,
    TrajectoryCollector,
    _render_awm_prompt_with_budget,
    _render_tau_prompt_with_budget,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chat, **kwargs):
        return "|".join(str(message["content"]) for message in chat)

    @staticmethod
    def encode(prompt, add_special_tokens=False):
        return list(prompt)


def test_awm_budget_returns_exact_visible_chat_and_drops_whole_old_exchanges():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old-action"},
        {"role": "tool", "content": "old-result"},
        {"role": "assistant", "content": "new-action"},
        {"role": "tool", "content": "new-result"},
    ]
    max_tokens = len("system|task|new-action|new-result")

    prompt, visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        max_prompt_tokens=max_tokens,
        history_window=6,
    )

    assert prompt == "system|task|new-action|new-result"
    assert visible == [*chat[:2], *chat[-2:]]


def test_awm_renderer_honors_configured_history_window():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for index in range(8):
        chat.extend(
            [
                {"role": "assistant", "content": f"action-{index}"},
                {"role": "tool", "content": f"result-{index}"},
            ]
        )

    _, default_visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=10_000,
        history_window=6,
    )
    _, short_visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=10_000,
        history_window=2,
    )
    _, empty_visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=10_000,
        history_window=0,
    )

    assert default_visible == [*chat[:2], *chat[-12:]]
    assert short_visible == [*chat[:2], *chat[-4:]]
    assert empty_visible == chat[:2]


def test_awm_renderer_reports_complete_exchange_overflow_diagnostics():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "action",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "very-long-result"},
    ]

    with pytest.raises(AWMContextBudgetExceeded) as error:
        _render_awm_prompt_with_budget(
            tokenizer,
            chat,
            {},
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ],
            max_prompt_tokens=len("system|task|action"),
            history_window=6,
        )

    diagnostics = error.value.diagnostics
    assert diagnostics["context_overflow_component"] == "newest_complete_exchange"
    assert diagnostics["context_prompt_tokens"] > diagnostics["context_max_prompt_tokens"]
    assert diagnostics["context_excess_tokens"] > 0
    assert diagnostics["context_latest_tool_name"] == "lookup"
    assert diagnostics["context_retained_exchange_count"] == 1


def test_awm_teacher_preflight_isolates_only_oversized_rows():
    collector = object.__new__(TrajectoryCollector)

    def preprocess(*, item, gen_batch, obs):
        if item == 1:
            raise AWMContextBudgetExceeded(
                {
                    "context_prompt_tokens": 110,
                    "context_max_prompt_tokens": 100,
                    "context_excess_tokens": 10,
                    "context_overflow_component": "newest_complete_exchange",
                }
            )
        return {"awm_visible_chat": json.dumps([{"role": "system", "content": f"state-{item}"}])}

    collector.preprocess_single_sample = preprocess
    gen_batch = SimpleNamespace(batch={"input_ids": np.zeros((3, 1))})

    ready, chats, overflows = collector.preprocess_awm_teacher_preflight(
        gen_batch,
        {},
    )

    assert ready.tolist() == [0, 2]
    assert [chat[0]["content"] for chat in chats] == ["state-0", "state-2"]
    assert overflows == [
        (
            1,
            {
                "context_prompt_tokens": 110,
                "context_max_prompt_tokens": 100,
                "context_excess_tokens": 10,
                "context_overflow_component": "newest_complete_exchange",
            },
        )
    ]


def test_awm_preprocess_forwards_configured_history_window(monkeypatch):
    captured = {}

    def render_awm(tokenizer, chat, kwargs, *, tools, max_prompt_tokens, history_window):
        captured["history_window"] = history_window
        return "rendered", chat

    monkeypatch.setattr(rollout_loop, "_render_awm_prompt_with_budget", render_awm)
    monkeypatch.setattr(
        rollout_loop.verl_F,
        "tokenize_and_postprocess_data",
        lambda **kwargs: (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
        ),
    )

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    config = SimpleNamespace(
        data=AttrDict(
            apply_chat_template_kwargs={},
            max_prompt_length=100,
            truncation="error",
            return_raw_chat=False,
        ),
        env=SimpleNamespace(
            env_name="awm_semantic",
            agentic_eval=AttrDict(prompt_rendering="chatml"),
            awm=SimpleNamespace(history_window=6),
        ),
    )
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": np.asarray([[{"role": "user", "content": "task"}]], dtype=object),
            "data_source": np.asarray(["awm"]),
        }
    )
    obs = {
        "text": ["task"],
        "chat": [chat],
        "tools": [[]],
        "prompt_protocol": ["awm_semantic"],
    }

    row = TrajectoryCollector(config, FakeTokenizer()).preprocess_single_sample(
        0,
        gen_batch,
        obs,
    )

    assert captured["history_window"] == 6
    assert "awm_visible_chat" in row


def test_tau_renderer_contract_remains_a_string():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "result"},
    ]
    rendered = _render_tau_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=1000,
    )
    assert isinstance(rendered, str)
