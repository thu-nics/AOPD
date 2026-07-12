import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "examples" / "vpr_games" / "summarize_in_domain_eval.py"


def write_result(run_dir: Path, seed: int, success_rate: float) -> None:
    seed_dir = run_dir / "sokoban" / "vpr_sokoban" / f"seed_{seed}"
    raw_dir = seed_dir / "raw"
    raw_dir.mkdir(parents=True)
    (seed_dir / ".done").write_text(
        "checkpoint=/checkpoint\nprotocol_sha256=protocol-hash\n"
    )
    (raw_dir / "100.jsonl").write_text('{"terminal_success": true}\n')
    (raw_dir / "100.metrics.json").write_text(
        json.dumps(
            {
                "val/env/success_rate": success_rate,
                "val/env/completion_rate": success_rate + 0.1,
                "val/env/valid_action_rate": 0.9,
            }
        )
    )


def test_summary_cli_writes_raw_and_aggregate_tables(tmp_path):
    write_result(tmp_path, 0, 0.2)
    write_result(tmp_path, 100, 0.4)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Summarized 2 completed evaluations across 1 models" in result.stdout
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert len(payload["records"]) == 2
    aggregate = payload["aggregate"][0]
    assert aggregate["task"] == "sokoban"
    assert aggregate["model"] == "vpr_sokoban"
    assert aggregate["completed_seeds"] == 2
    assert aggregate["seeds"] == [0, 100]
    assert aggregate["success_rate_mean"] == pytest.approx(0.3)
    assert aggregate["success_rate_std"] == pytest.approx(0.14142135623730953)
    assert aggregate["completion_rate_mean"] == pytest.approx(0.4)
    assert aggregate["valid_action_rate_mean"] == pytest.approx(0.9)

    with (tmp_path / "summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["seed"] for row in rows] == ["0", "100"]
    assert rows[0]["val/env/success_rate"] == "0.2"

    markdown = (tmp_path / "summary.md").read_text()
    assert "| sokoban | vpr_sokoban | 2 | 0.3000 |" in markdown
