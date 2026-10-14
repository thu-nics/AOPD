"""Prompt builders for native-function-calling agentic environments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

TAU_PROMPT_PROTOCOL = "tau2-native-llm-agent-v1"
ENVSCALER_PROMPT_PROTOCOL = "envscaler-conversation-v2"

TAU_SYSTEM_PROMPT_TEMPLATE = """<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
<policy>
{domain_policy}
</policy>"""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def tau_system_prompt(domain_policy: str) -> str:
    return TAU_SYSTEM_PROMPT_TEMPLATE.format(domain_policy=str(domain_policy))


def envscaler_system_prompt(environment: Mapping[str, Any]) -> str:
    introduction = str(environment.get("environment_introduction") or "")
    raw_rules = environment.get("constraints_rules") or []
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        raise TypeError("EnvScaler constraints_rules must be a sequence")
    rules = "\n".join(f"- {str(rule)}" for rule in raw_rules)
    interaction_guidance = (
        "Use tools for information they can provide. If essential information is "
        "unavailable from tools, ask the user a specific question instead of guessing. "
        "Adapt to tool results; do not repeat an identical call after the same result "
        "unless retrying or polling is justified. After completing the current request, "
        "report the result and ask for follow-up."
    )
    return f"""<instructions>
You are a helpful assistant operating in an interactive environment through native function calling. Fulfill the user's requests and communicate when needed.

In each turn, take exactly one action:
- Send one ordinary message to the user.
- Make one available tool call.
Do not do both or make multiple tool calls in one turn.

{interaction_guidance}
</instructions>
<environment>
{introduction}
</environment>
<rules>
{rules}
</rules>"""
