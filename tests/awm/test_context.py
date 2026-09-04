import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_system.multi_turn_rollout import rollout_loop
from agent_system.multi_turn_rollout.rollout_loop import (
    AWMContextBudgetExceeded,
    TauContextBudgetExceeded,
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
        max_history_exchanges=None,
    )

    assert prompt == "system|task|new-action|new-result"
    assert visible == [*chat[:2], *chat[-2:]]


def test_awm_renderer_defaults_to_full_history_and_supports_optional_cap():
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
        max_history_exchanges=None,
    )
    _, short_visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=10_000,
        max_history_exchanges=2,
    )
    _, empty_visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=10_000,
        max_history_exchanges=0,
    )

    assert default_visible == chat
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
            max_history_exchanges=6,
        )

    diagnostics = error.value.diagnostics
    assert diagnostics["context_overflow_component"] == "newest_complete_exchange"
    assert diagnostics["context_prompt_tokens"] > diagnostics["context_max_prompt_tokens"]
    assert diagnostics["context_excess_tokens"] > 0
    assert diagnostics["context_latest_tool_name"] == "lookup"
    assert diagnostics["context_retained_exchange_count"] == 1


@pytest.mark.parametrize("overflow_type", [AWMContextBudgetExceeded, TauContextBudgetExceeded])
def test_agentic_teacher_preflight_isolates_only_oversized_rows(overflow_type):
    collector = object.__new__(TrajectoryCollector)

    def preprocess(*, item, gen_batch, obs):
        if item == 1:
            raise overflow_type(
                {
                    "context_prompt_tokens": 110,
                    "context_max_prompt_tokens": 100,
                    "context_excess_tokens": 10,
                    "context_overflow_component": "newest_complete_exchange",
                }
            )
        return {"teacher_visible_chat": json.dumps([{"role": "system", "content": f"state-{item}"}])}

    collector.preprocess_single_sample = preprocess
    gen_batch = SimpleNamespace(batch={"input_ids": np.zeros((3, 1))})

    ready, chats, overflows = collector.preprocess_teacher_preflight_states(
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


def test_awm_preprocess_forwards_configured_max_history_exchanges(monkeypatch):
    captured = {}

    def render_awm(tokenizer, chat, kwargs, *, tools, max_prompt_tokens, max_history_exchanges):
        captured["max_history_exchanges"] = max_history_exchanges
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
            env_name="awm_agentic_opd",
            agentic_eval=AttrDict(prompt_rendering="chatml"),
            context=SimpleNamespace(max_history_exchanges=None),
            awm=SimpleNamespace(),
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
        "prompt_protocol": ["awm_agentic_opd"],
    }

    row = TrajectoryCollector(config, FakeTokenizer()).preprocess_single_sample(
        0,
        gen_batch,
        obs,
    )

    assert captured["max_history_exchanges"] is None
    assert "teacher_visible_chat" in row


def test_tau_renderer_returns_prompt_and_teacher_visible_chat():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "result"},
    ]
    rendered, visible_chat = _render_tau_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=[],
        max_prompt_tokens=1000,
    )
    assert isinstance(rendered, str)
    assert visible_chat == chat
    assert rendered == "policy|task|action|result"


def test_tau_renderer_reports_complete_exchange_overflow_diagnostics():
    tokenizer = FakeTokenizer()
    chat = [
        {"role": "system", "content": "policy"},
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

    with pytest.raises(TauContextBudgetExceeded) as error:
        _render_tau_prompt_with_budget(
            tokenizer,
            chat,
            {},
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ],
            max_prompt_tokens=len("policy|task|action"),
        )

    diagnostics = error.value.diagnostics
    assert diagnostics["context_overflow_component"] == "newest_complete_exchange"
    assert diagnostics["context_prompt_tokens"] > diagnostics["context_max_prompt_tokens"]
    assert diagnostics["context_excess_tokens"] > 0
    assert diagnostics["context_latest_tool_name"] == "lookup"
    assert diagnostics["context_retained_exchange_count"] == 1


def test_tau_manager_protocol_preserves_teacher_visible_chat(monkeypatch):
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
            env_name="tau_agentic_opd",
            agentic_eval=AttrDict(prompt_rendering="chatml"),
        ),
    )
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
    ]
    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": np.asarray([[{"role": "user", "content": "task"}]], dtype=object),
            "data_source": np.asarray(["tau"]),
        }
    )
    obs = {
        "text": ["task"],
        "chat": [chat],
        "tools": [[]],
        "prompt_protocol": ["tau"],
    }

    row = TrajectoryCollector(config, FakeTokenizer()).preprocess_single_sample(0, gen_batch, obs)

    assert json.loads(row["teacher_visible_chat"]) == chat
