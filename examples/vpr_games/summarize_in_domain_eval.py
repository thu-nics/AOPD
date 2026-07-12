#!/usr/bin/env python3
"""Aggregate completed in-domain evaluation metrics into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_done(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def parse_seed(name: str) -> int | str:
    value = name.removeprefix("seed_")
    try:
        return int(value)
    except ValueError:
        return value


def load_records(run_dir: Path) -> list[dict]:
    records = []
    for done_path in sorted(run_dir.glob("*/*/seed_*/.done")):
        seed_dir = done_path.parent
        metric_paths = sorted(
            (seed_dir / "raw").glob("*.metrics.json"),
            key=lambda path: path.stat().st_mtime,
        )
        raw_paths = sorted(
            path
            for path in (seed_dir / "raw").glob("*.jsonl")
            if not path.name.endswith(".metrics.json")
        )
        if not metric_paths or not raw_paths:
            continue
        metrics_path = metric_paths[-1]
        metrics = json.loads(metrics_path.read_text())
        done = parse_done(done_path)
        records.append(
            {
                "task": seed_dir.parents[1].name,
                "model": seed_dir.parent.name,
                "seed": parse_seed(seed_dir.name),
                "checkpoint": done.get("checkpoint", ""),
                "protocol_sha256": done.get("protocol_sha256", ""),
                "metrics_file": str(metrics_path.relative_to(run_dir)),
                "raw_file": str(raw_paths[-1].relative_to(run_dir)),
                "metrics": metrics,
            }
        )
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["model"])].append(record)

    aggregate = []
    for (task, model), group in sorted(grouped.items()):
        success = [float(row["metrics"].get("val/env/success_rate", 0.0)) for row in group]
        completion = [float(row["metrics"].get("val/env/completion_rate", 0.0)) for row in group]
        valid = [float(row["metrics"].get("val/env/valid_action_rate", 0.0)) for row in group]
        aggregate.append(
            {
                "task": task,
                "model": model,
                "completed_seeds": len(group),
                "seeds": [row["seed"] for row in group],
                "success_rate_mean": statistics.mean(success),
                "success_rate_std": statistics.stdev(success) if len(success) > 1 else 0.0,
                "completion_rate_mean": statistics.mean(completion),
                "valid_action_rate_mean": statistics.mean(valid),
            }
        )
    return aggregate


def write_json(run_dir: Path, records: list[dict], aggregate: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "records": records,
        "aggregate": aggregate,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def write_csv(run_dir: Path, records: list[dict]) -> None:
    metric_keys = sorted({key for record in records for key in record["metrics"]})
    fields = [
        "task",
        "model",
        "seed",
        "checkpoint",
        "protocol_sha256",
        "metrics_file",
        "raw_file",
    ]
    with (run_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields + metric_keys)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields}
            row.update(record["metrics"])
            writer.writerow(row)


def write_markdown(run_dir: Path, aggregate: list[dict]) -> None:
    lines = [
        "# In-domain evaluation summary",
        "",
        "| Task | Model | Seeds | Success mean | Success std | Completion | Valid action |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['completed_seeds']} "
            f"| {row['success_rate_mean']:.4f} | {row['success_rate_std']:.4f} "
            f"| {row['completion_rate_mean']:.4f} | {row['valid_action_rate_mean']:.4f} |"
        )
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines))


def summarize(run_dir: Path) -> tuple[list[dict], list[dict]]:
    records = load_records(run_dir)
    aggregate = aggregate_records(records)
    write_json(run_dir, records, aggregate)
    write_csv(run_dir, records)
    write_markdown(run_dir, aggregate)
    return records, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")
    records, aggregate = summarize(args.run_dir)
    print(f"Summarized {len(records)} completed evaluations across {len(aggregate)} models")


if __name__ == "__main__":
    main()
