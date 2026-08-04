#!/usr/bin/env python3
"""Require a healthy AWM server with the pinned logical-time identity."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from agent_system.environments.env_package.awm.logical_time import require_server_protocol


def require_server_run_id(base_url: str, expected_run_id: str, timeout: float) -> None:
    url = str(base_url).rstrip("/") + "/awm-run-identity"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    actual_run_id = payload.get("run_id")
    if actual_run_id != expected_run_id:
        raise RuntimeError(f"AWM server run identity mismatch: expected {expected_run_id!r}, got {actual_run_id!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expected-run-id")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    protocol = require_server_protocol(args.base_url, args.data_dir, timeout=args.timeout)
    if args.expected_run_id is not None:
        require_server_run_id(args.base_url, args.expected_run_id, args.timeout)
    print(json.dumps(protocol, sort_keys=True))


if __name__ == "__main__":
    main()
