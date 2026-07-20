"""Summarize ALFWorld and WebShop validation-only metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def find_metric(metrics: dict[str, object], exact: tuple[str, ...], contains: str) -> float:
    for key in exact:
        if key in metrics:
            return float(metrics[key])
    matches = [
        float(value)
        for key, value in metrics.items()
        if contains in key and isinstance(value, (int, float))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one metric containing {contains!r}, found {len(matches)}"
        )
    return matches[0]


def mean_stats(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    return mean, std, 1.96 * std / math.sqrt(len(values))


def collect_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        return rows

    for metrics_path in sorted(results_dir.glob("*/*/seed_*/raw/*.metrics.json")):
        seed_dir = metrics_path.parents[1]
        benchmark_dir = seed_dir.parent
        model_dir = benchmark_dir.parent
        done_file = seed_dir / ".done"
        if not done_file.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        success = find_metric(
            metrics,
            ("val/success_rate", "val/env/success_rate"),
            "success_rate",
        )
        task_score = None
        if benchmark_dir.name == "webshop":
            task_score = find_metric(
                metrics,
                (
                    "val/webshop_task_score (not success_rate)",
                    "val/env/webshop_task_score (not success_rate)",
                ),
                "webshop_task_score",
            )
        task_rates = {
            key.removeprefix("val/"): float(value)
            for key, value in metrics.items()
            if key.endswith("_success_rate")
            and key not in {"val/success_rate", "val/env/success_rate"}
            and isinstance(value, (int, float))
        }
        rows.append(
            {
                "model_id": model_dir.name,
                "benchmark": benchmark_dir.name,
                "seed": seed_dir.name.removeprefix("seed_"),
                "success_rate": success,
                "task_score": task_score,
                "task_rates": task_rates,
                "metrics_path": str(metrics_path.relative_to(run_dir)),
            }
        )
    return rows


def aggregate(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_id"]), str(row["benchmark"]))].append(row)

    summaries = []
    for (model_id, benchmark), group in sorted(groups.items()):
        success_values = [float(row["success_rate"]) for row in group]
        success_mean, success_std, success_ci95 = mean_stats(success_values)
        task_values = [
            float(row["task_score"])
            for row in group
            if row["task_score"] is not None
        ]
        task_mean, task_std, task_ci95 = (
            mean_stats(task_values) if task_values else (None, None, None)
        )
        summaries.append(
            {
                "model_id": model_id,
                "benchmark": benchmark,
                "n_seeds": len(group),
                "success_rate_mean": success_mean,
                "success_rate_std": success_std,
                "success_rate_ci95": success_ci95,
                "task_score_mean": task_mean,
                "task_score_std": task_std,
                "task_score_ci95": task_ci95,
            }
        )

    task_type_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["benchmark"] != "alfworld":
            continue
        for task_type, value in dict(row["task_rates"]).items():
            task_type_groups[(str(row["model_id"]), task_type)].append(float(value))

    task_type_summaries = []
    for (model_id, task_type), values in sorted(task_type_groups.items()):
        mean, std, ci95 = mean_stats(values)
        task_type_summaries.append(
            {
                "model_id": model_id,
                "task_type": task_type,
                "n_seeds": len(values),
                "success_rate_mean": mean,
                "success_rate_std": std,
                "success_rate_ci95": ci95,
            }
        )
    return summaries, task_type_summaries


def write_outputs(run_dir: Path, rows: list[dict[str, object]]) -> None:
    summaries, task_type_summaries = aggregate(rows)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "summary": summaries,
                "alfworld_task_types": task_type_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with (run_dir / "raw_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "benchmark",
                "seed",
                "success_rate",
                "task_score",
                "metrics_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Agentic OOD Evaluation",
        "",
        "| Model | Benchmark | Seeds | Success rate | Run std | Task score | Task-score std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        success = f"{100 * float(item['success_rate_mean']):.2f}"
        success_std = f"{100 * float(item['success_rate_std']):.2f}"
        task_score = "-"
        task_std = "-"
        if item["task_score_mean"] is not None:
            task_score = f"{100 * float(item['task_score_mean']):.2f}"
            task_std = f"{100 * float(item['task_score_std']):.2f}"
        lines.append(
            f"| {item['model_id']} | {item['benchmark']} | "
            f"{item['n_seeds']} | {success} | {success_std} | "
            f"{task_score} | {task_std} |"
        )
    if task_type_summaries:
        lines.extend(
            [
                "",
                "## ALFWorld Task Types",
                "",
                "| Model | Task type | Seeds | Success rate | Run std |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in task_type_summaries:
            lines.append(
                f"| {item['model_id']} | {item['task_type']} | "
                f"{item['n_seeds']} | "
                f"{100 * float(item['success_rate_mean']):.2f} | "
                f"{100 * float(item['success_rate_std']):.2f} |"
            )
    lines.extend(
        [
            "",
            "Values are means and sample standard deviations across sampling "
            "seeds. JSON output also includes normal-approximation 95% confidence "
            "half-widths. WebShop task score is the original partial-credit "
            "reward; success is exact task completion.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = collect_rows(args.run_dir)
    write_outputs(args.run_dir, rows)
    print(f"Summarized {len(rows)} completed evaluations in {args.run_dir}")


if __name__ == "__main__":
    main()
