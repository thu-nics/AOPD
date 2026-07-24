"""Build deterministic per-step batches for unified DAPO and VPR training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from agent_system.environments.env_package.vpr_games.mixed.envs import interleave_counts


def load_unique_rows(path: Path, expected_unique: int | None = None) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            extra_info = row.get("extra_info") or {}
            source_id = str(extra_info.get("index", ""))
            if not source_id:
                source_id = json.dumps(
                    [row.get("prompt"), row.get("reward_model")],
                    sort_keys=True,
                    ensure_ascii=True,
                )
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append(row)
            if expected_unique is not None and len(rows) == expected_unique:
                return rows
    if expected_unique is not None and len(rows) != expected_unique:
        raise ValueError(f"expected {expected_unique} unique rows, found {len(rows)}")
    return rows


def question_from_prompt(prompt: list[dict[str, str]]) -> str:
    for message in reversed(prompt):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("math prompt has no user message")


def make_math_row(
    sources: list[dict[str, Any]],
    split: str,
    index: int,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("math source pool must be non-empty")
    source = sources[0]
    prompt = [dict(message) for message in source["prompt"]]
    reward_model = dict(source["reward_model"])
    ground_truth = str(reward_model["ground_truth"])
    data_source = str(source.get("data_source", "math_dapo"))
    questions = [question_from_prompt(row["prompt"]) for row in sources]
    ground_truths = [str(row["reward_model"]["ground_truth"]) for row in sources]
    data_sources = [str(row.get("data_source", "math_dapo")) for row in sources]
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": str(source.get("ability", "math")),
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": index, "task": "math"},
        "env_kwargs": {
            "task": "math",
            "question": questions[0],
            "ground_truth": ground_truths[0],
            "data_source": data_sources[0],
            "question_pool": questions,
            "ground_truth_pool": ground_truths,
            "data_source_pool": data_sources,
        },
    }


def make_game_row(task: str, split: str, index: int) -> dict[str, Any]:
    data_source = f"vpr_{task}"
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": f"Play a {task} game."}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": split, "index": index, "task": task},
        "env_kwargs": {
            "task": task,
            "question": "",
            "ground_truth": "",
            "data_source": data_source,
            "question_pool": [],
            "ground_truth_pool": [],
            "data_source_pool": [],
        },
    }


def build_rows(
    math_rows: list[dict[str, Any]],
    *,
    split: str,
    num_batches: int,
    counts: dict[str, int],
    math_pool_size: int = 1,
) -> list[dict[str, Any]]:
    if not math_rows and counts.get("math", 0):
        raise ValueError("math rows are required when the math count is positive")
    if math_pool_size <= 0:
        raise ValueError("math_pool_size must be positive")
    labels = interleave_counts(counts)
    output = []
    math_index = 0
    row_index = 0
    for _ in range(num_batches):
        for task in labels:
            if task == "math":
                sources = [
                    math_rows[(math_index + offset) % len(math_rows)]
                    for offset in range(math_pool_size)
                ]
                output.append(make_math_row(sources, split, row_index))
                math_index += math_pool_size
            else:
                output.append(make_game_row(task, split, row_index))
            row_index += 1
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapo-train", type=Path, required=True)
    parser.add_argument("--dapo-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--expected-unique-train", type=int, default=17917)
    parser.add_argument("--math", type=int, default=64)
    parser.add_argument("--sokoban", type=int, default=6)
    parser.add_argument("--sudoku", type=int, default=8)
    parser.add_argument("--minesweeper", type=int, default=18)
    parser.add_argument("--val-math", type=int, default=30)
    parser.add_argument("--val-per-game", type=int, default=32)
    parser.add_argument("--math-pool-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    train_math = load_unique_rows(
        args.dapo_train, expected_unique=args.expected_unique_train
    )
    val_math = load_unique_rows(args.dapo_val)
    train_counts = {
        "math": args.math,
        "sokoban": args.sokoban,
        "sudoku": args.sudoku,
        "minesweeper": args.minesweeper,
    }
    validation_counts = {
        "math": args.val_math,
        "sokoban": args.val_per_game,
        "sudoku": args.val_per_game,
        "minesweeper": args.val_per_game,
    }

    train_rows = build_rows(
        train_math,
        split="train",
        num_batches=args.train_steps,
        counts=train_counts,
        math_pool_size=args.math_pool_size,
    )
    val_rows = build_rows(
        val_math,
        split="validation",
        num_batches=1,
        counts=validation_counts,
        math_pool_size=1,
    )

    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(args.output_dir / "train.parquet")
    Dataset.from_list(val_rows).to_parquet(args.output_dir / "validation.parquet")
    manifest = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_steps": args.train_steps,
        "train_counts": train_counts,
        "math_pool_size": args.math_pool_size,
        "validation_counts": validation_counts,
        "unique_train_math": len(train_math),
        "unique_validation_math": len(val_math),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
