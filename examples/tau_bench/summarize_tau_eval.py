"""Aggregate Tau evaluation metrics across models and seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev

METRICS = {
    "success_rate": "val/env/success_rate",
    "airline_success_rate": "val/env/airline/success_rate",
    "retail_success_rate": "val/env/retail/success_rate",
    "valid_action_rate": "val/env/valid_action_rate",
    "airline_valid_action_rate": "val/env/airline/valid_action_rate",
    "retail_valid_action_rate": "val/env/retail/valid_action_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(results_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(results_dir.glob("*/seed_*/raw/*.metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "model_id": path.parents[2].name,
            "seed": int(path.parents[1].name.removeprefix("seed_")),
            "metrics_path": str(path.resolve()),
        }
        for output_name, metric_name in METRICS.items():
            row[output_name] = float(metrics[metric_name]) if metric_name in metrics else None
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    models = sorted({str(row["model_id"]) for row in rows})
    output = []
    for model_id in models:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        summary: dict[str, object] = {"model_id": model_id, "num_seeds": len(model_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in model_rows if row.get(metric) is not None]
            summary[f"{metric}_mean"] = fmean(values) if values else None
            summary[f"{metric}_std"] = pstdev(values) if values else None
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"no Tau metrics found under {args.results_dir}")
    summary = summarize(rows)
    payload = {"runs": rows, "summary": summary}
    output_path = args.results_dir.parent / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.results_dir.parent / "summary.csv", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
