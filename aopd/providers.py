"""Explicit Chat Completions capabilities; provider extensions never leak."""

from copy import deepcopy

PROVIDERS = frozenset({"openai-compatible", "vllm", "deepseek", "dashscope", "zai"})


def adapt_chat_payload(payload, provider):
    if provider not in PROVIDERS:
        raise ValueError(f"unsupported provider: {provider}")
    result = deepcopy(payload)
    if provider == "vllm":
        if "enable_thinking" in result:
            result.setdefault("chat_template_kwargs", {})["enable_thinking"] = result.pop("enable_thinking")
        return result
    enabled = result.pop("chat_template_kwargs", {}).get("enable_thinking")
    for key in ("top_k", "min_p", "repetition_penalty"):
        result.pop(key, None)
    if provider == "openai-compatible":
        for key in ("enable_thinking", "thinking", "thinking_budget"):
            result.pop(key, None)
        for message in result.get("messages", []):
            message.pop("reasoning_content", None)
        return result
    if enabled is not None:
        if provider == "dashscope":
            result["enable_thinking"] = enabled
        else:
            result["thinking"] = {"type": "enabled" if enabled else "disabled"}
            if provider == "zai":
                result["thinking"]["clear_thinking"] = False
            if provider == "deepseek" and enabled:
                result.pop("temperature", None)
                result.pop("top_p", None)
    return result


def adapt_litellm_kwargs(arguments, provider):
    """LiteLLM transports nonstandard provider fields through extra_body."""
    value = deepcopy(arguments)
    extra = value.pop("extra_body", {})
    value = adapt_chat_payload({**value, **extra}, provider)
    extensions = {key: value.pop(key) for key in ("chat_template_kwargs", "top_k", "min_p", "repetition_penalty", "enable_thinking", "thinking", "thinking_budget") if key in value}
    if extensions:
        value["extra_body"] = extensions
    return value
