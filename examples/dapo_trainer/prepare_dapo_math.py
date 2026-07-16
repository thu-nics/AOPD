"""Deduplicate locally materialized DAPO and AIME parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_dapo_vpr_mixed import load_unique_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=17917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_unique_rows(
        args.train_source,
        expected_unique=args.expected_train,
    )
    val_rows = load_unique_rows(args.val_source)

    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "dapo-math-17k-unique.parquet"
    val_path = args.output_dir / "aime-2024-unique.parquet"
    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(val_rows).to_parquet(val_path)
    manifest = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_file": str(train_path),
        "validation_file": str(val_path),
    }
    (args.output_dir / "math_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
