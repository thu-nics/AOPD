#!/usr/bin/env python3
"""Materialize or verify an exact-length deterministic AWM training schedule."""

import argparse
import json
from pathlib import Path

from agent_system.environments.env_package.awm.data.pools import (
    materialize_training_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    args = parser.parse_args()
    result = materialize_training_schedule(
        source_data=args.data,
        source_manifest_path=args.manifest,
        output_data=args.output_data,
        output_manifest_path=args.output_manifest,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
