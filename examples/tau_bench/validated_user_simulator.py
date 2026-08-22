"""Tau user simulator adapter with strict local-generation validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tau2.data_model.message import UserMessage
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

USER_NAME = "validated_user_simulator"
_RETRY_SEED_STRIDE = 104_729


class LocalUserGenerationError(RuntimeError):
    """A completed local generation that cannot be used as a user turn."""


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
        raise LocalUserGenerationError("Local user simulator output was truncated")
    if not message.has_content() and not message.is_tool_call():
        raise LocalUserGenerationError("Local user simulator returned no final content or tool call")


class ValidatedUserSimulator(UserSimulator):
    """Native Tau user simulator with bounded invalid-generation retries."""

    def __init__(self, *args, llm_args=None, **kwargs):
        clean_llm_args = dict(llm_args or {})
        self.validation_retries = int(clean_llm_args.pop("_validation_retries", 0))
        if self.validation_retries < 0:
            raise ValueError("_validation_retries must be nonnegative")
        super().__init__(*args, llm_args=clean_llm_args, **kwargs)

    def generate_next_message(self, message, state):
        # UserSimulator._generate_next_message appends the incoming message before
        # generation. Retry against a copy so an invalid response cannot duplicate
        # that turn in the retained state.
        base_seed = self.llm_args.get("seed")
        try:
            for attempt in range(self.validation_retries + 1):
                if attempt > 0 and base_seed is not None:
                    self.llm_args["seed"] = base_seed + attempt * _RETRY_SEED_STRIDE
                trial_state = deepcopy(state)
                try:
                    user_message, updated_state = super().generate_next_message(message, trial_state)
                    validate_user_generation(user_message)
                    return user_message, updated_state
                except LocalUserGenerationError:
                    if attempt == self.validation_retries:
                        raise
        finally:
            if base_seed is None:
                self.llm_args.pop("seed", None)
            else:
                self.llm_args["seed"] = base_seed


def register_validated_user_simulator() -> None:
    if USER_NAME not in registry.get_users():
        registry.register_user(ValidatedUserSimulator, USER_NAME)
