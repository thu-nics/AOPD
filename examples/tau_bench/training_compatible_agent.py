"""Training-time Tau action parsing for the native Tau runner."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import litellm
from pydantic import BaseModel, Field

from agent_system.environments.env_package.tau_bench.actions import (
    ParsedAction,
    parse_action,
    tau_messages_to_openai,
    validate_tau_action,
)
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.registry import registry


AGENT_NAME = "training_compatible_agent"
STOP_TOKEN = "###TRAINING_COMPATIBLE_INVALID_LIMIT###"
SYSTEM_PROMPT = (
    "You are a customer-service agent. Follow the domain policy and use the "
    "available tools when needed. At each turn, produce exactly one current "
    "action: either one tool call or one message to the user.\n\nDOMAIN POLICY:\n"
    "{domain_policy}"
)


class TrainingCompatibleAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    decision_count: int = 0
    invalid_action_count: int = 0
    pending_invalid_attempts: list[dict[str, Any]] = Field(default_factory=list)


def _response_usage(response: Any) -> dict[str, int] | None:
    usage = response.get("usage") if hasattr(response, "get") else None
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    def _field(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    return {
        "completion_tokens": _field("completion_tokens"),
        "prompt_tokens": _field("prompt_tokens"),
    }


def response_raw_text(response: Any) -> str:
    """Reconstruct raw model text before Tau message validation."""
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        content = f"<think>{reasoning}</think>{content}"

    tool_texts = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        payload = {
            "name": tool_call.function.name,
            "arguments": arguments,
        }
        tool_texts.append(
            f"<tool_call>{json.dumps(payload, ensure_ascii=True)}</tool_call>"
        )
    return f"{content}{''.join(tool_texts)}"


class TrainingCompatibleAgent(
    LLMConfigMixin,
    HalfDuplexAgent[TrainingCompatibleAgentState],
):
    """Tau agent that applies the training-time raw action parser."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: dict[str, Any] | None = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.decision_limit = int(
            self.llm_args.pop("_training_decision_limit")
        )
        self.invalid_action_limit = int(
            self.llm_args.pop("_training_invalid_action_limit")
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy)

    def get_init_state(
        self, message_history: list[Message] | None = None
    ) -> TrainingCompatibleAgentState:
        history = list(message_history or [])
        if not all(is_valid_agent_history_message(msg) for msg in history):
            raise ValueError("Invalid training-compatible agent history")
        return TrainingCompatibleAgentState(
            system_messages=[
                SystemMessage(role="system", content=self.system_prompt)
            ],
            messages=history,
        )

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content and STOP_TOKEN in message.content)

    def _completion(
        self,
        messages: list[APICompatibleMessage],
        *,
        seed: int | None,
    ) -> tuple[Any, str, float]:
        kwargs = dict(self.llm_args)
        if seed is not None:
            kwargs["seed"] = seed
        started = time.perf_counter()
        response = litellm.completion(
            model=self.llm,
            messages=tau_messages_to_openai(messages),
            tools=[tool.openai_schema for tool in self.tools],
            # Tools remain in ChatML, while vLLM returns unparsed model text.
            tool_choice="none",
            **kwargs,
        )
        return response, response_raw_text(response), time.perf_counter() - started

    @staticmethod
    def _attempt_summary(
        raw_text: str,
        parsed: ParsedAction,
        response: Any,
    ) -> dict[str, Any]:
        return {
            "error": parsed.error or "invalid action",
            "raw_text_prefix": raw_text[:512],
            "raw_text_length": len(raw_text),
            "usage": _response_usage(response),
        }

    @staticmethod
    def _response_data(response: Any) -> dict[str, Any]:
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return {"response": str(response)}

    def _message_from_action(
        self,
        action: ParsedAction,
        *,
        response: Any,
        raw_text: str,
        generation_time: float,
        state: TrainingCompatibleAgentState,
    ) -> AssistantMessage:
        tool_calls = None
        content = None
        if action.kind == "tool":
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    name=action.name,
                    arguments=action.arguments or {},
                )
            ]
        else:
            content = action.content

        raw_data = self._response_data(response)
        raw_data["training_compatible"] = {
            "raw_text": raw_text,
            "decision_count": state.decision_count,
            "invalid_action_count": state.invalid_action_count,
            "invalid_attempts_before_action": list(
                state.pending_invalid_attempts
            ),
        }
        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=0.0,
            usage=_response_usage(response),
            raw_data=raw_data,
            generation_time_seconds=generation_time,
        )

    def _stop_message(
        self, state: TrainingCompatibleAgentState
    ) -> AssistantMessage:
        return AssistantMessage(
            role="assistant",
            content=STOP_TOKEN,
            cost=0.0,
            raw_data={
                "training_compatible": {
                    "decision_count": state.decision_count,
                    "invalid_action_count": state.invalid_action_count,
                    "termination": "invalid_or_decision_limit",
                    "invalid_attempts_before_action": list(
                        state.pending_invalid_attempts
                    ),
                }
            },
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: TrainingCompatibleAgentState,
    ) -> tuple[AssistantMessage, TrainingCompatibleAgentState]:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("Training-compatible agent is text-only")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        base_seed = self.llm_args.get("seed")
        while state.decision_count < self.decision_limit:
            attempt_seed = (
                int(base_seed) + state.decision_count
                if base_seed is not None
                else None
            )
            response, raw_text, generation_time = self._completion(
                state.system_messages + state.messages,
                seed=attempt_seed,
            )
            state.decision_count += 1
            action = validate_tau_action(parse_action(raw_text), self.tools)
            if action.kind != "invalid":
                assistant_message = self._message_from_action(
                    action,
                    response=response,
                    raw_text=raw_text,
                    generation_time=generation_time,
                    state=state,
                )
                state.pending_invalid_attempts.clear()
                state.messages.append(assistant_message)
                return assistant_message, state

            state.invalid_action_count += 1
            state.pending_invalid_attempts.append(
                self._attempt_summary(raw_text, action, response)
            )
            if state.invalid_action_count >= self.invalid_action_limit:
                break

        assistant_message = self._stop_message(state)
        state.messages.append(assistant_message)
        return assistant_message, state


def create_training_compatible_agent(
    tools, domain_policy: str, **kwargs: Any
) -> TrainingCompatibleAgent:
    return TrainingCompatibleAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def register_training_compatible_agent() -> None:
    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(
            create_training_compatible_agent,
            AGENT_NAME,
        )
