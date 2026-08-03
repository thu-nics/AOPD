#!/usr/bin/env python3
"""Verify the current AWM deterministic training pool."""

import argparse
import json
from pathlib import Path

from agent_system.environments.env_package.awm.verification import (
    verify_training_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify_training_pool(args.data, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
