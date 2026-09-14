"""Small provider-decoding and pair-scheduling helpers; no transport or cache I/O."""

import json
from concurrent.futures import ThreadPoolExecutor


def matcher_decoding_config(provider, *, enable_thinking=None, reasoning_effort=None, max_tokens=None):
    """Preserve non-DeepSeek defaults; DeepSeek uses the audited low preset."""
    defaults = {
        "deepseek": (True, "low", 32768),
        "openai-compatible": (True, None, 8192),
        "dashscope": (False, None, 128),
        "zai": (True, "max", 8192),
    }
    if provider not in defaults:
        raise ValueError(f"unsupported matcher provider: {provider!r}")
    default_enabled, default_effort, default_budget = defaults[provider]
    enabled = default_enabled if enable_thinking is None else enable_thinking
    if type(enabled) is not bool:
        raise ValueError("matcher_enable_thinking must be a boolean or null")
    budget = default_budget if max_tokens is None else max_tokens
    if type(budget) is not int or budget <= 0:
        raise ValueError("matcher_max_tokens must be a positive integer")
    effort = default_effort if reasoning_effort is None else reasoning_effort
    if reasoning_effort is not None and (not enabled or provider not in {"deepseek", "zai"}):
        raise ValueError("matcher_reasoning_effort requires a thinking DeepSeek/ZAI matcher")
    if effort not in {None, "low", "medium", "high", "max"}:
        raise ValueError("invalid matcher_reasoning_effort")
    config = {"max_tokens": budget, "stream": False}
    if provider in {"deepseek", "zai"}:
        config["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if provider == "zai":
            config["thinking"]["clear_thinking"] = False
        if enabled:
            config["reasoning_effort"] = effort
    else:
        config["enable_thinking"] = enabled
    if provider == "deepseek" and enabled:
        pass  # Thinking sampling is provider-controlled; omit temperature/top_p.
    elif provider == "zai":
        config.update(temperature=1.0, top_p=0.95)
    else:
        config.update(temperature=0.0, top_p=1.0)
    if provider != "openai-compatible":
        config["response_format"] = {"type": "json_object"}
    return config


def matcher_boolean(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") not in {None, "stop"}:
        raise ValueError(f"incomplete matcher response: {choice.get('finish_reason')}")
    content = choice["message"].get("content") or ""
    obj = json.loads(content[content.find("{") : content.rfind("}") + 1])
    value = obj.get("equivalent") if isinstance(obj, dict) else None
    if type(value) is not bool:
        raise ValueError("matcher decision must be a JSON equivalent boolean")
    return value


def match_message_matrix(teachers, candidates, match_pair, *, normalize=str.strip, max_workers=32):
    """Deduplicate work only, then restore every original K vote position."""
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("matcher concurrency must be a positive integer")
    if not teachers:
        return {"counts": [0] * len(candidates), "matrix": [[] for _ in candidates]}
    keys, unique = [], {}
    for candidate in candidates:
        for teacher in teachers:
            key = (normalize(str(teacher)), normalize(str(candidate)))
            keys.append(key)
            unique.setdefault(key, (teacher, candidate))
    if not unique:
        return {"counts": [], "matrix": []}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique))) as pool:
        values = dict(zip(unique, pool.map(lambda pair: match_pair(*pair), unique.values()), strict=True))
    if any(type(value) is not bool for value in values.values()):
        raise ValueError("matcher decisions must be booleans")
    flat = [values[key] for key in keys]
    matrix = [flat[i : i + len(teachers)] for i in range(0, len(flat), len(teachers))]
    return {"counts": [sum(row) for row in matrix], "matrix": matrix}
