from agent_system.multi_turn_rollout.rollout_loop import (
    _render_awm_prompt_with_budget,
    _render_tau_prompt_with_budget,
)


class FakeTokenizer:
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
        {"role": "assistant", "content": "list_tools"},
        {"role": "user", "content": "tools"},
        {"role": "assistant", "content": "old-action"},
        {"role": "user", "content": "old-result"},
        {"role": "assistant", "content": "new-action"},
        {"role": "user", "content": "new-result"},
    ]
    max_tokens = len("system|task|list_tools|tools|new-action|new-result")

    prompt, visible = _render_awm_prompt_with_budget(
        tokenizer,
        chat,
        {},
        max_prompt_tokens=max_tokens,
    )

    assert prompt == "system|task|list_tools|tools|new-action|new-result"
    assert visible == [*chat[:4], *chat[-2:]]


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
