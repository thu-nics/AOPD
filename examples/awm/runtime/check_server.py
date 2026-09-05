#!/usr/bin/env python3
"""Require a healthy AWM server with the pinned logical-time identity."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from agent_system.environments.env_package.awm.runtime.logical_time import require_server_protocol
from agent_system.environments.env_package.awm.runtime.terminal_judge import (
    TERMINAL_JUDGE_PROTOCOL_VERSION,
    terminal_judge_decoding_config,
)


def require_server_run_id(base_url: str, expected_run_id: str, timeout: float) -> None:
    url = str(base_url).rstrip("/") + "/awm-run-identity"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    actual_run_id = payload.get("run_id")
    if actual_run_id != expected_run_id:
        raise RuntimeError(f"AWM server run identity mismatch: expected {expected_run_id!r}, got {actual_run_id!r}")


def require_terminal_judge_protocol(
    base_url: str,
    *,
    expected_provider: str,
    expected_model: str,
    expected_reasoning_effort: str,
    minimum_max_tokens: int,
    timeout: float,
) -> dict:
    url = str(base_url).rstrip("/") + "/awm-terminal-judge"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    expected = {
        "protocol_version": TERMINAL_JUDGE_PROTOCOL_VERSION,
        "provider": expected_provider,
        "model": expected_model,
        "reasoning_effort": expected_reasoning_effort,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"AWM terminal judge {field} mismatch: expected {value!r}, got {payload.get(field)!r}")
    if int(payload.get("max_tokens", 0)) < int(minimum_max_tokens):
        raise RuntimeError("AWM terminal judge max_tokens is below the required budget")
    expected_decoding = terminal_judge_decoding_config(
        provider=expected_provider,
        reasoning_effort=expected_reasoning_effort,
        max_tokens=int(payload["max_tokens"]),
    )
    if payload.get("decoding_config") != expected_decoding:
        raise RuntimeError("AWM terminal judge decoding protocol mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expected-run-id")
    parser.add_argument("--expected-terminal-provider", default="deepseek")
    parser.add_argument("--expected-terminal-model")
    parser.add_argument("--expected-terminal-reasoning-effort", default="max")
    parser.add_argument("--minimum-terminal-max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    protocol = require_server_protocol(args.base_url, args.data_dir, timeout=args.timeout)
    if args.expected_run_id is not None:
        require_server_run_id(args.base_url, args.expected_run_id, args.timeout)
    terminal_judge = None
    if args.expected_terminal_model is not None:
        terminal_judge = require_terminal_judge_protocol(
            args.base_url,
            expected_provider=args.expected_terminal_provider,
            expected_model=args.expected_terminal_model,
            expected_reasoning_effort=args.expected_terminal_reasoning_effort,
            minimum_max_tokens=args.minimum_terminal_max_tokens,
            timeout=args.timeout,
        )
    payload = protocol
    if terminal_judge is not None:
        payload = {"logical_time": protocol, "terminal_judge": terminal_judge}
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
