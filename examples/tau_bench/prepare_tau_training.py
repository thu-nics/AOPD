"""Build deterministic Tau Airline/Retail training batches from a qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_system.environments.env_package.tau_bench.envs import (
    DOMAIN_ORDER,
    interleave_domains,
    load_qualification_manifest,
)


def make_row(domain: str, task_id: str, split: str, index: int) -> dict[str, Any]:
    return {
        "data_source": f"tau_{domain}",
        "prompt": [{"role": "user", "content": f"Handle Tau {domain} task {task_id}."}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "domain": domain,
            "task_id": task_id,
        },
        "env_kwargs": {"domain": domain, "task_id": task_id},
    }


def build_rows(
    task_pools: dict[str, list[str]],
    *,
    counts: dict[str, int],
    num_batches: int,
    split: str,
) -> list[dict[str, Any]]:
    labels = interleave_domains(counts)
    positions = {domain: 0 for domain in DOMAIN_ORDER}
    output = []
    for _ in range(num_batches):
        for domain in labels:
            pool = task_pools.get(domain) or []
            if not pool:
                raise ValueError(f"no {split} tasks available for Tau {domain}")
            position = positions[domain]
            task_id = pool[position % len(pool)]
            output.append(make_row(domain, task_id, split, len(output)))
            positions[domain] += 1
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--airline", type=int, default=4)
    parser.add_argument("--retail", type=int, default=4)
    parser.add_argument("--val-airline", type=int, default=4)
    parser.add_argument("--val-retail", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    qualification = load_qualification_manifest(args.qualification_manifest)
    stable = {
        domain: list((qualification.get("stable_tasks") or {}).get(domain) or [])
        for domain in DOMAIN_ORDER
    }
    test_tasks = qualification.get("test_tasks") or {}
    validation = {
        domain: list(test_tasks.get(domain) or stable[domain])
        for domain in DOMAIN_ORDER
    }
    train_counts = {"airline": args.airline, "retail": args.retail}
    validation_counts = {"airline": args.val_airline, "retail": args.val_retail}
    train_rows = build_rows(
        stable,
        counts=train_counts,
        num_batches=args.train_steps,
        split="train",
    )
    validation_rows = build_rows(
        validation,
        counts=validation_counts,
        num_batches=1,
        split="test",
    )

    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(args.output_dir / "train.parquet")
    Dataset.from_list(validation_rows).to_parquet(args.output_dir / "validation.parquet")
    manifest = {
        "qualification_manifest": str(args.qualification_manifest.resolve()),
        "train_steps": args.train_steps,
        "train_counts": train_counts,
        "validation_counts": validation_counts,
        "stable_task_counts": {domain: len(stable[domain]) for domain in DOMAIN_ORDER},
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
