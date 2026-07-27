from types import SimpleNamespace

from pydantic import BaseModel
from tau2.data_model.message import UserMessage

from examples.tau_bench import training_compatible_agent as adapter


class Params(BaseModel):
    item_id: int


TOOL = SimpleNamespace(
    name="lookup",
    params=Params,
    openai_schema={
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": Params.model_json_schema(),
        },
    },
)


class FakeResponse:
    def __init__(self, content, *, reasoning_content=None, tool_calls=None):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
            )
        ]
        self.usage = {
            "completion_tokens": 7,
            "prompt_tokens": 11,
        }

    def get(self, key):
        return getattr(self, key, None)

    def to_dict(self):
        return {"fake": True}


def make_agent(*, decision_limit=5, invalid_action_limit=3):
    return adapter.TrainingCompatibleAgent(
        tools=[TOOL],
        domain_policy="Use lookup before replying.",
        llm="openai/test-model",
        llm_args={
            "seed": 41,
            "temperature": 0.6,
            "_training_decision_limit": decision_limit,
            "_training_invalid_action_limit": invalid_action_limit,
        },
    )


def test_invalid_output_is_resampled_from_same_history(monkeypatch):
    responses = iter(
        [
            FakeResponse("<think>unfinished"),
            FakeResponse(
                '<think>reason <tool_call>{"name":"lookup",'
                '"arguments":{"item_id":7}}</tool_call></think>'
            ),
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(adapter.litellm, "completion", fake_completion)
    agent = make_agent()
    state = agent.get_init_state()
    message, state = agent.generate_next_message(
        UserMessage(role="user", content="Find item 7"), state
    )

    assert message.content is None
    assert message.tool_calls[0].name == "lookup"
    assert message.tool_calls[0].arguments == {"item_id": 7}
    assert state.decision_count == 2
    assert state.invalid_action_count == 1
    assert len(state.messages) == 2
    assert calls[0]["messages"] == calls[1]["messages"]
    assert [call["seed"] for call in calls] == [41, 42]
    assert calls[0]["tools"] == [TOOL.openai_schema]
    assert calls[0]["tool_choice"] == "none"

    metadata = message.raw_data["training_compatible"]
    assert len(metadata["invalid_attempts_before_action"]) == 1
    assert metadata["invalid_attempts_before_action"][0]["error"] == (
        "unclosed reasoning tag"
    )


def test_invalid_limit_returns_agent_stop_without_polluting_history(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse("<think>unfinished")

    monkeypatch.setattr(adapter.litellm, "completion", fake_completion)
    agent = make_agent(decision_limit=9, invalid_action_limit=2)
    state = agent.get_init_state()
    message, state = agent.generate_next_message(
        UserMessage(role="user", content="Help"), state
    )

    assert agent.is_stop(message)
    assert state.decision_count == 2
    assert state.invalid_action_count == 2
    assert len(calls) == 2
    assert len(state.messages) == 2
    assert all("unfinished" not in str(item) for item in state.messages)
    assert message.raw_data["training_compatible"]["termination"] == (
        "invalid_or_decision_limit"
    )


def test_raw_response_reconstructs_reasoning_and_structured_tool_call():
    function = SimpleNamespace(
        name="lookup", arguments='{"item_id": 3}'
    )
    tool_call = SimpleNamespace(function=function)
    response = FakeResponse(
        "",
        reasoning_content="check first",
        tool_calls=[tool_call],
    )

    assert adapter.response_raw_text(response) == (
        '<think>check first</think><tool_call>{"name": "lookup", '
        '"arguments": {"item_id": 3}}</tool_call>'
    )
