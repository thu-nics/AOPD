"""Validate provider-returned names without rewriting request/cache identities."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit


def response_model_matches(*, provider: str, api_base: str, requested_model: str, returned_model: str) -> bool:
    if not isinstance(requested_model, str) or not requested_model or not isinstance(returned_model, str) or not returned_model:
        return False
    if returned_model == requested_model:
        return True
    # Observed on the official service, not a documented model-version promise.
    # This is deliberately one-way and endpoint-scoped. Never normalize cache
    # keys, accept arbitrary prefixes, or extend it to third-party gateways.
    if (provider, requested_model, returned_model) != ("deepseek", "deepseek-v4-flash", "deepseek-flash"):
        return False
    try:
        endpoint = urlsplit(api_base)
        return (
            endpoint.scheme == "https"
            and endpoint.hostname == "api.deepseek.com"
            and endpoint.port in (None, 443)
            and endpoint.username is None
            and endpoint.password is None
            and endpoint.path.rstrip("/") in {"", "/v1", "/chat/completions", "/v1/chat/completions"}
            and not endpoint.query
            and not endpoint.fragment
        )
    except ValueError:
        return False


def checked_provider_identity(response: Mapping[str, Any], *, provider: str, api_base: str, requested_model: str, role: str) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RuntimeError(f"{role} returned no model identity")
    returned = str(response.get("model") or "")
    if response.get("requested_model", requested_model) != requested_model or not response_model_matches(provider=provider, api_base=api_base, requested_model=requested_model, returned_model=returned):
        raise RuntimeError(f"{role} returned model {returned!r}, expected {requested_model!r}")
    return {
        "requested_model": requested_model,
        "model": returned,
        "system_fingerprint": str(response["system_fingerprint"]) if response.get("system_fingerprint") is not None else None,
    }


def deepseek_identity_preflight(*, api_base: str, model: str, api_key: str, session=None) -> dict[str, Any]:
    """Check the model list and both thinking modes before loading GPUs.

    Checks API/model identity, not task success or full tool-calling quality.
    No credentials or generated text are included in the returned report.
    """
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    owns_session = session is None
    if owns_session:
        import requests

        session = requests.Session()
    session.trust_env = False
    headers = {"Authorization": f"Bearer {api_key}"}
    report: dict[str, Any] = {"api_base": api_base, "requested_model": model, "checks": []}
    try:
        response = session.get(f"{api_base.rstrip('/')}/models", headers=headers, timeout=60)
        response.raise_for_status()
        report["advertised_models"] = [item["id"] for item in response.json()["data"]]
        report["request_name_advertised"] = model in report["advertised_models"]
        for mode in ("disabled", "enabled"):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply OK."}],
                "thinking": {"type": mode},
                "max_tokens": 128 if mode == "enabled" else 32,
                "stream": False,
            }
            if mode == "enabled":
                payload["reasoning_effort"] = "max"
            response = session.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            identity = checked_provider_identity(data, provider="deepseek", api_base=api_base, requested_model=model, role=f"preflight/{mode}")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping) or not isinstance(choices[0].get("message"), Mapping):
                raise RuntimeError(f"preflight/{mode} returned no chat completion")
            report["checks"].append(
                {
                    **identity,
                    "thinking": mode,
                    "response_name_alias": identity["model"] != model,
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": data.get("usage"),
                }
            )
        report["status"] = "passed"
        return report
    finally:
        if owns_session:
            session.close()
