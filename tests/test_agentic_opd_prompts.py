from agent_system.environments.env_package.awm.runtime.actions import (
    native_system_prompt,
)
from agent_system.environments.prompts.agentic_opd import (
    ENVSCALER_PROMPT_PROTOCOL,
    TAU_PROMPT_PROTOCOL,
    envscaler_system_prompt,
    prompt_hash,
    tau_system_prompt,
)


def test_awm_native_prompt_remains_unchanged():
    assert native_system_prompt() == (
        "You are operating in an interactive environment. Use the available functions "
        "to complete the user's task. The functions are supplied through the model's "
        "native tool-calling interface.\n\n"
        "At each decision, take exactly one action: either call exactly one available "
        "function or send one ordinary assistant message. Never combine a function call "
        "with a message, and never call multiple functions in one decision. You are "
        "already logged in, and your user id is 1 if required.\n\n"
        "At the final step, directly output the answer or summary without a function call."
    )


def test_tau_prompt_is_native_compatible_and_single_action():
    prompt = tau_system_prompt("Always verify before updating.")

    assert TAU_PROMPT_PROTOCOL == "tau-native-compatible-v2"
    assert (
        prompt
        == """<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn, take exactly one action:
- Send one message to the user.
- Make one tool call.
You cannot do both at the same time, and you cannot make multiple tool calls in one turn.

Try to be helpful and always follow the policy. Tool calls are provided through the model's native function-calling interface.
</instructions>
<policy>
Always verify before updating.
</policy>"""
    )
    assert len(prompt_hash(prompt)) == 64


def test_envscaler_prompt_is_concise_and_progress_oriented():
    prompt = envscaler_system_prompt(
        {
            "environment_introduction": "Manage a test workspace.",
            "constraints_rules": ["Never delete records.", "Confirm updates."],
        }
    )

    assert ENVSCALER_PROMPT_PROTOCOL == "envscaler-conversation-v2"
    guidance = (
        "Use tools for information they can provide. If essential information is "
        "unavailable from tools, ask the user a specific question instead of guessing. "
        "Adapt to tool results; do not repeat an identical call after the same result "
        "unless retrying or polling is justified. After completing the current request, "
        "report the result and ask for follow-up."
    )
    assert (
        prompt
        == f"""<instructions>
You are a helpful assistant operating in an interactive environment through native function calling. Fulfill the user's requests and communicate when needed.

In each turn, take exactly one action:
- Send one ordinary message to the user.
- Make one available tool call.
Do not do both or make multiple tool calls in one turn.

{guidance}
</instructions>
<environment>
Manage a test workspace.
</environment>
<rules>
- Never delete records.
- Confirm updates.
</rules>"""
    )
