"""Build deterministic Tau datasets from official train/test task splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_system.environments.env_package.tau_bench.envs import (
    DOMAIN_ORDER,
    OFFICIAL_TASK_COUNTS,
    TASK_MANIFEST_PROTOCOL_VERSION,
    interleave_domains,
    validate_tau_source,
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _task_payload(task: Any) -> Any:
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json")
    if isinstance(task, dict):
        return task
    raise TypeError(f"unsupported Tau task object: {type(task)!r}")


def _task_id(task: Any) -> str:
    value = task.get("id") if isinstance(task, dict) else getattr(task, "id", None)
    if value is None:
        raise ValueError("Tau task is missing id")
    return str(value)


def deterministic_seed(domain: str, task_id: str, trial: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{domain}:{task_id}:{trial}:{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def make_row(
    domain: str,
    task_id: str,
    split: str,
    index: int,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {"domain": domain, "task_id": task_id}
    if seed is not None:
        env_kwargs["seed"] = int(seed)
    return {
        "data_source": f"tau_{domain}",
        "prompt": [
            {
                "role": "user",
                "content": f"Handle Tau {domain} task {task_id}.",
            }
        ],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "domain": domain,
            "task_id": task_id,
            **({"seed": int(seed)} if seed is not None else {}),
        },
        "env_kwargs": env_kwargs,
    }


def build_train_rows(
    task_pools: dict[str, list[Any]],
    *,
    counts: dict[str, int],
    num_batches: int,
) -> list[dict[str, Any]]:
    labels = interleave_domains(counts)
    positions = {domain: 0 for domain in DOMAIN_ORDER}
    output: list[dict[str, Any]] = []
    for _ in range(num_batches):
        for domain in labels:
            pool = task_pools[domain]
            position = positions[domain]
            task_id = _task_id(pool[position % len(pool)])
            output.append(make_row(domain, task_id, "train", len(output)))
            positions[domain] += 1
    return output


def allocate_validation_counts(
    pool_sizes: dict[str, int],
    *,
    domains: list[str],
    batch_size: int,
) -> dict[str, int]:
    """Allocate one fixed worker-domain template proportional to available rows."""
    if batch_size < len(domains):
        raise ValueError("validation batch size must be at least the number of domains")
    if any(int(pool_sizes.get(domain, 0)) <= 0 for domain in domains):
        raise ValueError("each Tau validation domain must contain tasks")
    total = sum(int(pool_sizes[domain]) for domain in domains)
    raw = {domain: batch_size * int(pool_sizes[domain]) / total for domain in domains}
    counts = {domain: max(1, int(raw[domain])) for domain in domains}
    while sum(counts.values()) < batch_size:
        domain = max(
            domains,
            key=lambda value: (raw[value] - counts[value], -domains.index(value)),
        )
        counts[domain] += 1
    while sum(counts.values()) > batch_size:
        candidates = [domain for domain in domains if counts[domain] > 1]
        if not candidates:
            raise ValueError("cannot allocate a positive fixed Tau domain template")
        domain = max(
            candidates,
            key=lambda value: (counts[value] - raw[value], domains.index(value)),
        )
        counts[domain] -= 1
    return {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER}


def build_validation_rows(
    task_pools: dict[str, list[Any]],
    *,
    domains: list[str],
    trials: int,
    base_seed: int,
    num_tasks: int | None,
    batch_size: int,
    split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: dict[str, list[tuple[Any, int]]] = {}
    for domain in domains:
        tasks = task_pools[domain]
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        expanded[domain] = [(task, trial) for task in tasks for trial in range(trials)]
    counts = allocate_validation_counts(
        {domain: len(expanded[domain]) for domain in domains},
        domains=domains,
        batch_size=batch_size,
    )
    num_batches = min(len(expanded[domain]) // counts[domain] for domain in domains if counts[domain] > 0)
    if num_batches <= 0:
        raise ValueError("Tau validation selection cannot form one complete batch")

    labels = interleave_domains(counts)
    positions = {domain: 0 for domain in DOMAIN_ORDER}
    output: list[dict[str, Any]] = []
    for _ in range(num_batches):
        for domain in labels:
            task, trial = expanded[domain][positions[domain]]
            positions[domain] += 1
            task_id = _task_id(task)
            output.append(
                make_row(
                    domain,
                    task_id,
                    split,
                    len(output),
                    seed=deterministic_seed(domain, task_id, trial, base_seed),
                )
            )
    available = {domain: len(expanded.get(domain, [])) for domain in DOMAIN_ORDER}
    evaluated = {domain: num_batches * counts.get(domain, 0) for domain in DOMAIN_ORDER}
    plan = {
        "mode": "fixed_domain_complete_batches",
        "batch_size": batch_size,
        "counts": counts,
        "batches": num_batches,
        "available_rows": available,
        "evaluated_rows": evaluated,
        "dropped_rows": {domain: available[domain] - evaluated[domain] for domain in DOMAIN_ORDER},
    }
    return output, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/public2/yuanhuining/repos/tau2-bench"),
    )
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--airline", type=int, default=5)
    parser.add_argument("--retail", type=int, default=11)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument(
        "--validation-domains",
        default="airline",
        help="Comma-separated official validation domains: airline,retail",
    )
    parser.add_argument("--validation-split", choices=["test", "base"], default="test")
    parser.add_argument("--validation-trials", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=300)
    parser.add_argument(
        "--validation-num-tasks",
        type=int,
        default=None,
        help="Optional per-domain prefix for smoke tests; formal runs leave unset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if args.validation_trials <= 0:
        raise ValueError("validation_trials must be positive")
    if args.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    if args.validation_num_tasks is not None and args.validation_num_tasks <= 0:
        raise ValueError("validation_num_tasks must be positive when set")
    validation_domains = [value.strip().lower() for value in args.validation_domains.split(",") if value.strip()]
    if not validation_domains or any(domain not in DOMAIN_ORDER for domain in validation_domains):
        raise ValueError("validation_domains must be a comma-separated subset of airline,retail")
    if len(validation_domains) != len(set(validation_domains)):
        raise ValueError("validation_domains contains duplicates")

    source = validate_tau_source(args.source_root)
    from tau2.runner.helpers import load_tasks

    train_tasks = {domain: list(load_tasks(domain, "train")) for domain in DOMAIN_ORDER}
    validation_tasks = {domain: list(load_tasks(domain, args.validation_split)) for domain in validation_domains}
    for domain in DOMAIN_ORDER:
        expected = OFFICIAL_TASK_COUNTS["train"][domain]
        if len(train_tasks[domain]) != expected:
            raise RuntimeError(f"official Tau train count mismatch for {domain}: expected {expected}, got {len(train_tasks[domain])}")
    for domain in validation_domains:
        expected = OFFICIAL_TASK_COUNTS[args.validation_split][domain]
        if len(validation_tasks[domain]) != expected:
            raise RuntimeError(f"official Tau validation count mismatch for {domain}: expected {expected}, got {len(validation_tasks[domain])}")

    train_counts = {"airline": args.airline, "retail": args.retail}
    if any(value < 0 for value in train_counts.values()) or not sum(train_counts.values()):
        raise ValueError("Tau train counts must be nonnegative and sum to a positive batch")

    train_rows = build_train_rows(
        train_tasks,
        counts=train_counts,
        num_batches=args.train_steps,
    )
    validation_rows, validation_plan = build_validation_rows(
        validation_tasks,
        split=args.validation_split,
        batch_size=args.validation_batch_size,
        domains=validation_domains,
        trials=args.validation_trials,
        base_seed=args.validation_seed,
        num_tasks=args.validation_num_tasks,
    )

    from datasets import Dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(validation_rows).to_parquet(validation_path)
    manifest = {
        "protocol_version": TASK_MANIFEST_PROTOCOL_VERSION,
        **source,
        "train_split": "train",
        "validation_split": args.validation_split,
        "train_steps": args.train_steps,
        "validation_plan": validation_plan,
        "train_counts": train_counts,
        "validation_domains": validation_domains,
        "validation_trials": args.validation_trials,
        "validation_seed": args.validation_seed,
        "validation_num_tasks": args.validation_num_tasks,
        "official_task_counts": OFFICIAL_TASK_COUNTS,
        "task_ids": {
            "train": {domain: [_task_id(task) for task in train_tasks[domain]] for domain in DOMAIN_ORDER},
            args.validation_split: {domain: [_task_id(task) for task in validation_tasks[domain]] for domain in validation_domains},
        },
        "task_content_sha256": {
            "train": {domain: _sha256_json([_task_payload(task) for task in train_tasks[domain]]) for domain in DOMAIN_ORDER},
            args.validation_split: {domain: _sha256_json([_task_payload(task) for task in validation_tasks[domain]]) for domain in validation_domains},
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
