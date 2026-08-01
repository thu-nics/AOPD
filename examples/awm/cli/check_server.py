#!/usr/bin/env python3
"""Require a healthy AWM server with the pinned logical-time identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_system.environments.env_package.awm.logical_time import require_server_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    protocol = require_server_protocol(args.base_url, args.data_dir, timeout=args.timeout)
    print(json.dumps(protocol, sort_keys=True))


if __name__ == "__main__":
    main()
