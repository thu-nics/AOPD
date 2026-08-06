#!/usr/bin/env python3
"""Materialize or verify a deterministic ordered prefix of an AWM pool."""

import argparse
import json
from pathlib import Path

from agent_system.environments.env_package.awm.data.pools import (
    materialize_training_slice,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-count", type=int)
    selection.add_argument("--fraction")
    args = parser.parse_args()
    result = materialize_training_slice(
        source_data=args.data,
        source_manifest_path=args.manifest,
        output_data=args.output_data,
        output_manifest_path=args.output_manifest,
        task_count=args.task_count,
        fraction=args.fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
