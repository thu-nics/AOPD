#!/usr/bin/env python3
"""Prepare the complete AWM task dataset and its hash-stable split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "Snowflake/AgentWorldModel-1K"
DATASET_REVISION = "dde80a0283fe781bdc51656bce57063dc5650213"
PROTOCOL_VERSION = 1
EXPECTED_ENVIRONMENTS = 1000
EXPECTED_TASKS_PER_ENVIRONMENT = 10
EXPECTED_SOURCE_SHA256 = {
    "gen_db.jsonl": "ae8acb3c23765ca4866b35799ffb980fbb15831240fdc35c046e8a7d27a2c0e8",
    "gen_envs.jsonl": "2c7749c1710303f0f663bbe14aea689ade77c19282e2dd0ef0c54e2a95f5e7d8",
    "gen_sample.jsonl": "39c40969ad76d52a3ea51384752a639cf55dce15bc1a8f0f022cfb6bbd25db3c",
    "gen_scenario.jsonl": "6362e31af6e39bc914c6606b32f62b3d5f571f9653e86a6a9c29e507ae0e647e",
    "gen_tasks.jsonl": "0537871c8824cd56d23cc51118294ae3c3070d990cb9a3f766f9dc690f91bf45",
    "gen_verifier.jsonl": "269a97a085dce103afed6e5c48d001b8d47a9e7e2b142f2edc0ddaf6da08a348",
    "gen_verifier.pure_code.jsonl": "2de0b668bd0c6b37a033dda7d697bcd8ddc3bf5b0c9bf83fbd691fbd2c3827f7",
}
DATA_FILES = (
    "gen_scenario.jsonl",
    "gen_tasks.jsonl",
    "gen_db.jsonl",
    "gen_sample.jsonl",
    "gen_envs.jsonl",
    "gen_verifier.jsonl",
    "gen_verifier.pure_code.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_hashes(source_hashes: dict[str, str]) -> None:
    if source_hashes != EXPECTED_SOURCE_SHA256:
        mismatched = {
            filename: {
                "expected": EXPECTED_SOURCE_SHA256.get(filename),
                "actual": source_hashes.get(filename),
            }
            for filename in sorted(set(EXPECTED_SOURCE_SHA256) | set(source_hashes))
            if EXPECTED_SOURCE_SHA256.get(filename) != source_hashes.get(filename)
        }
        raise RuntimeError("AWM source files do not match the pinned dataset revision: " + json.dumps(mismatched, sort_keys=True))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _download(data_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename in DATA_FILES:
        paths[filename] = Path(
            hf_hub_download(
                repo_id=DATASET_NAME,
                repo_type="dataset",
                revision=DATASET_REVISION,
                filename=filename,
                local_dir=data_dir,
            )
        )
    return paths["gen_tasks.jsonl"], paths["gen_verifier.pure_code.jsonl"]


def _stable_rank(value: str) -> tuple[str, str]:
    return hashlib.sha256(value.encode()).hexdigest(), value


def _validate_and_expand(
    task_records: list[dict[str, Any]],
    verifier_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if len(task_records) != EXPECTED_ENVIRONMENTS:
        raise RuntimeError(f"AWM revision must contain {EXPECTED_ENVIRONMENTS} environments; found {len(task_records)}")
    scenarios = [str(record.get("scenario") or "") for record in task_records]
    if any(not scenario for scenario in scenarios) or len(set(scenarios)) != len(scenarios):
        raise RuntimeError("AWM scenarios must be non-empty and unique")

    # Kept as an optional argument so existing callers can rebuild the same
    # hash-stable base parquet. Pool membership is independent of both verifier
    # families; SQL verifier health is audited later by health.py.
    del verifier_records
    rows = []
    seen_task_text = set()
    for record in task_records:
        scenario = str(record["scenario"])
        tasks = record.get("tasks") or []
        if len(tasks) != EXPECTED_TASKS_PER_ENVIRONMENT:
            raise RuntimeError(f"AWM scenario {scenario!r} has {len(tasks)} tasks instead of 10")
        for task_idx, task in enumerate(tasks):
            task = str(task)
            task_id = f"{scenario}:{task_idx}"
            if task in seen_task_text:
                raise RuntimeError(f"duplicate AWM task text at {task_id}")
            seen_task_text.add(task)
            rows.append(
                {
                    "task_id": task_id,
                    "scenario": scenario,
                    "task_idx": task_idx,
                    "task": task,
                }
            )
    expected_tasks = EXPECTED_ENVIRONMENTS * EXPECTED_TASKS_PER_ENVIRONMENT
    if len(rows) != expected_tasks:
        raise RuntimeError(f"AWM revision has {len(rows)} tasks instead of {expected_tasks}")
    return rows


def _select_splits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    ranked_scenarios = sorted(by_scenario, key=_stable_rank)
    dev_scenarios = ranked_scenarios[:32]
    dev = []
    for scenario in dev_scenarios:
        ranked_tasks = sorted(by_scenario[scenario], key=lambda row: _stable_rank(row["task_id"]))
        dev.extend(ranked_tasks[:8])
    smoke = []
    for scenario in dev_scenarios[:4]:
        ranked_tasks = sorted(by_scenario[scenario], key=lambda row: _stable_rank(row["task_id"]))
        smoke.extend(ranked_tasks[:2])
    splits = {"all": list(rows), "dev": dev, "smoke": smoke}
    expected = {
        "all": (1000, 10000),
        "dev": (32, 256),
        "smoke": (4, 8),
    }
    for name, selected in splits.items():
        actual = (len({row["scenario"] for row in selected}), len(selected))
        if actual != expected[name]:
            raise AssertionError(f"AWM {name} split expected {expected[name]}, got {actual}")
    return splits


def _training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": "awm",
        "prompt": [{"role": "user", "content": row["task"]}],
        "ability": "agentic_tool_use",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "task_id": row["task_id"],
            "scenario": row["scenario"],
            "task_idx": row["task_idx"],
            "task": row["task"],
        },
        "env_kwargs": {
            "scenario": row["scenario"],
            "task_idx": row["task_idx"],
        },
    }


def _build_manifest(
    splits: dict[str, list[dict[str, Any]]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": source_hashes,
        "selection": ("sha256(task_or_scenario_id), independent of expert/verifier/student outcomes"),
        "counts": {
            name: {
                "environments": len({row["scenario"] for row in selected}),
                "tasks": len(selected),
            }
            for name, selected in splits.items()
        },
        "split_task_ids": {name: [row["task_id"] for row in selected] for name, selected in splits.items()},
    }


def _verify_prepared_artifacts(
    output_dir: Path,
    expected_manifest: dict[str, Any],
) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing prepared AWM manifest: {manifest_path}")
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise RuntimeError("prepared AWM manifest does not match the pinned source and ID-only splits")
    for split in ("all",):
        expected_ids = expected_manifest["split_task_ids"][split]
        parquet_path = output_dir / f"awm_{split}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing prepared AWM split: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        actual_ids = [str(extra_info["task_id"]) for extra_info in frame["extra_info"].tolist()]
        if actual_ids != expected_ids:
            raise RuntimeError(f"prepared AWM split {split!r} does not match its manifest task IDs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AWM_DATA_DIR", "~/.cache/openenv/awm")).expanduser(),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/awm"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use already downloaded pinned files from --data-dir without network access.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify source hashes, manifest, and prepared split IDs without rewriting.",
    )
    args = parser.parse_args()

    if args.local_files_only:
        missing = [str(args.data_dir / filename) for filename in DATA_FILES if not (args.data_dir / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"missing local AWM data file(s): {missing}")
        tasks_path = args.data_dir / "gen_tasks.jsonl"
        verifier_path = args.data_dir / "gen_verifier.pure_code.jsonl"
    else:
        tasks_path, verifier_path = _download(args.data_dir)

    source_hashes = {filename: _sha256(args.data_dir / filename) for filename in DATA_FILES}
    _validate_source_hashes(source_hashes)

    rows = _validate_and_expand(
        _load_jsonl(tasks_path),
        _load_jsonl(verifier_path),
    )
    splits = _select_splits(rows)
    manifest = _build_manifest(splits, source_hashes)
    if args.verify_only:
        _verify_prepared_artifacts(args.output_dir, manifest)
        print(
            json.dumps(
                {"verified_output_dir": str(args.output_dir), **manifest["counts"]},
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": source_hashes,
    }
    (args.data_dir / "dataset_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("all",):
        selected = splits[name]
        pd.DataFrame([_training_row(row) for row in selected]).to_parquet(
            args.output_dir / f"awm_{name}.parquet",
            index=False,
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
