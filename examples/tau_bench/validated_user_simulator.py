"""Tau user simulator adapter with strict local-generation validation."""

from __future__ import annotations

from typing import Any

from tau2.data_model.message import UserMessage
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

USER_NAME = "validated_user_simulator"


def _finish_reason(raw_data: dict[str, Any] | None) -> str | None:
    choices = (raw_data or {}).get("choices") or []
    if not choices:
        return None
    first = choices[0]
    if isinstance(first, dict):
        value = first.get("finish_reason")
    else:
        value = getattr(first, "finish_reason", None)
    return str(value) if value is not None else None


def validate_user_generation(message: UserMessage) -> None:
    """Reject truncated or empty local-user generations as infrastructure errors."""
    finish_reason = _finish_reason(message.raw_data)
    if finish_reason == "length":
        raise RuntimeError("Local user simulator output was truncated")
    if not message.has_content() and not message.is_tool_call():
        raise RuntimeError("Local user simulator returned no final content or tool call")


class ValidatedUserSimulator(UserSimulator):
    """Native Tau user simulator that validates the provider response."""

    def _generate_next_message(self, message, state) -> UserMessage:
        user_message = super()._generate_next_message(message, state)
        validate_user_generation(user_message)
        return user_message


def register_validated_user_simulator() -> None:
    if USER_NAME not in registry.get_users():
        registry.register_user(ValidatedUserSimulator, USER_NAME)
