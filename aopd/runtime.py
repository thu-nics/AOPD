"""Validate device allocation independently of scientific recipes."""

import copy
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml

from aopd.providers import PROVIDERS


def active_runtime(runtime, names):
    result = copy.deepcopy(runtime)
    result["roles"] = {k: v for k, v in result.get("roles", {}).items() if k in names}
    services = {v["service"] for v in result["roles"].values()}
    result["services"] = {k: v for k, v in result.get("services", {}).items() if k in services}
    return result


def validate_generation(roles, *, tau=False, evaluation=False):
    matcher = {"enable_thinking", "max_tokens", "reasoning_effort", "max_concurrent_requests"}
    allowed = {
        "teacher": {"enable_thinking", "temperature", "top_p", "max_tokens", "reasoning_effort", "thinking_budget"} | ({"top_k", "min_p", "max_concurrent_requests"} if tau else {"presence_penalty"}),
        "matcher": matcher,
        "user": {"enable_thinking", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens"} if tau or evaluation else {"enable_thinking", "temperature"},
        "runtime_judge": {"max_tokens", "reasoning_effort", "max_format_retries"},
        "terminal_judge": {"max_tokens", "reasoning_effort", "timeout_seconds", "max_retries"},
    }
    for name, role in roles.items():
        if name not in allowed or (extra := set(role["generation"]) - allowed[name]):
            raise ValueError(f"{name}: unsupported generation parameters: {sorted(extra) if name in allowed else name}")


def load_yaml(path):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _gpus(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}.gpus must be a nonempty list")
    ids = [str(v) for v in value]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate GPUs in {label}")
    if any(not (v.isdigit() or v.startswith("GPU-")) for v in ids):
        raise ValueError(f"{label}: use physical GPU indices or UUIDs")
    return set(ids)


def validate_runtime(runtime):
    if not runtime.get("model"):
        raise ValueError("model must name a local HF model or checkpoint")
    student = runtime.get("student", {})
    used = _gpus(student.get("gpus"), "student")
    for kind in ("tp", "sp"):
        size = int(student.get(kind, 1))
        if size < 1 or len(used) % size:
            raise ValueError(f"student {kind} must divide GPU count")
    ports = set()
    for name, service in runtime.get("services", {}).items():
        if service.get("provider", "vllm" if service.get("mode") == "local" else "openai-compatible") not in PROVIDERS:
            raise ValueError(f"{name}: unsupported provider")
        if "api_key" in service:
            raise ValueError(f"{name}: use api_key_env, never literal secrets")
        if not service.get("model"):
            raise ValueError(f"{name}: missing model identity")
        mode = service.get("mode")
        if mode == "local":
            devices = _gpus(service.get("gpus"), name)
            if devices & used:
                raise ValueError(f"GPU overlap for service {name}: {devices & used}")
            used |= devices
            tp, dp = int(service.get("tp", 1)), int(service.get("dp", 1))
            if min(tp, dp) < 1 or tp * dp != len(devices):
                raise ValueError(f"{name}: tp * dp must equal GPU count")
            if not service.get("model_path"):
                raise ValueError(f"{name}: local service needs model_path")
            port = int(service.get("port", 0))
            if not 1024 <= port <= 65535 or port in ports:
                raise ValueError(f"invalid or duplicate port for {name}")
            ports.add(port)
        elif mode == "api":
            if service.get("gpus"):
                raise ValueError(f"{name}: API services must not allocate GPUs")
            parsed = urlparse(str(service.get("base_url", "")))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
                raise ValueError(f"{name}: invalid base_url (no embedded credentials)")
        else:
            raise ValueError(f"{name}: mode must be local or api")
    resolve_roles(runtime)


def resolve_roles(runtime):
    roles = {}
    for name, role in runtime.get("roles", {}).items():
        service_name = role.get("service")
        if service_name not in runtime.get("services", {}):
            raise ValueError(f"{name}: missing service {service_name}")
        service = copy.deepcopy(runtime["services"][service_name])
        service["generation"] = dict(role.get("generation", {}))
        if service["mode"] == "local":
            service["base_url"] = f"http://127.0.0.1:{service['port']}/v1"
            service.setdefault("provider", "vllm")
        service.setdefault("api_key_env", "AOPD_LOCAL_API_KEY")
        service.setdefault("provider", "openai-compatible")
        roles[name] = service
    return roles


def expand_runtime(runtime):
    """Explicit ${VAR} expansion; missing paths/identities never get guessed."""
    import re

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):

            def replace(match):
                key = match[1]
                if key not in os.environ:
                    raise ValueError(f"runtime requires environment variable {key}")
                return os.environ[key]

            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
        return value

    return expand(runtime)
