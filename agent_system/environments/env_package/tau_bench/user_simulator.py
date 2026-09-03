"""Validated Tau user simulator for OpenAI-compatible local or remote models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tau2.data_model.message import UserMessage
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator

USER_NAME = "validated_user_simulator"
_RETRY_SEED_STRIDE = 104_729


class UserGenerationError(RuntimeError):
    """A completed user generation that cannot be used as a Tau user turn."""


def _finish_reason(raw_data: dict[str, Any] | None) -> str | None:
    choices = (raw_data or {}).get("choices") or []
    if not choices:
        return None
    first = choices[0]
    value = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
    return str(value) if value is not None else None


def validate_user_generation(message: UserMessage) -> None:
    """Reject truncated or empty generations as retryable infrastructure errors."""
    if _finish_reason(message.raw_data) == "length":
        raise UserGenerationError("Tau user simulator output was truncated")
    if not message.has_content() and not message.is_tool_call():
        raise UserGenerationError("Tau user simulator returned no final content or tool call")


class ValidatedUserSimulator(UserSimulator):
    """Native Tau user simulator with bounded invalid-generation retries."""

    def __init__(self, *args, llm_args=None, **kwargs):
        clean_llm_args = dict(llm_args or {})
        self.validation_retries = int(clean_llm_args.pop("_validation_retries", 0))
        if self.validation_retries < 0:
            raise ValueError("_validation_retries must be nonnegative")
        super().__init__(*args, llm_args=clean_llm_args, **kwargs)

    def generate_next_message(self, message, state):
        # The native simulator appends the incoming message before generation.
        # Retry against a copy so a rejected response cannot duplicate that turn.
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
                except UserGenerationError:
                    if attempt == self.validation_retries:
                        raise
        finally:
            if base_seed is None:
                self.llm_args.pop("seed", None)
            else:
                self.llm_args["seed"] = base_seed


class ValidatedUserAgentGymEnv(AgentGymEnv):
    """Pinned Tau AgentGymEnv using the validated user implementation."""

    def _get_user(self) -> UserSimulator:
        environment = self._get_environment()
        task = self._get_task()
        try:
            user_tools = environment.get_user_tools(include=task.user_tools) or None
        except ValueError:
            user_tools = None
        if self.solo_mode:
            return DummyUser()
        return ValidatedUserSimulator(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
        )


# Compatibility alias for historical imports and persisted exception names.
LocalUserGenerationError = UserGenerationError


def register_validated_user_simulator() -> None:
    if USER_NAME not in registry.get_users():
        registry.register_user(ValidatedUserSimulator, USER_NAME)
