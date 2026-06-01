"""Generate minimal parquet data files for VPR smoke tests.

VPR environments generate their own observations from game state, so the data
file only provides dummy prompts to bootstrap the rollout loop. Each row
corresponds to one episode in the training batch.
"""

import argparse
import os
from pathlib import Path


def make_rows(env_name: str, split: str, count: int) -> dict:
    return {
        "data_source": [env_name] * count,
        "prompt": [
            [{"role": "user", "content": f"Play a {env_name} game."}]
            for _ in range(count)
        ],
        "ability": ["agent"] * count,
        "reward_model": [
            {"style": "rule", "ground_truth": ""}
            for _ in range(count)
        ],
        "extra_info": [
            {"split": split, "index": i}
            for i in range(count)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="vpr_tictactoe")
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / "data" / args.env_name)

    from datasets import Dataset
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = Dataset.from_dict(make_rows(args.env_name, "train", args.train_size))
    val = Dataset.from_dict(make_rows(args.env_name, "test", args.val_size))
    train.to_parquet(out / "train.parquet")
    val.to_parquet(out / "test.parquet")

    print(f"Wrote {len(train)} train rows and {len(val)} val rows to {out}")


if __name__ == "__main__":
    main()
