#!/usr/bin/env python3
"""Run and summarize native tau2 text evaluations.

The agent is served by a local OpenAI-compatible endpoint. The user simulator
can use either another local endpoint or a configured remote LiteLLM model.
Before Tau's native EvaluationType.ALL is called, experimental NL assertions
are removed from a copy of the task reward basis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import litellm
from tau2.data_model.simulation import Results, TerminationReason, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import compute_metrics, is_successful
from tau2.runner import get_tasks, run_tasks

if __package__:
    from .deterministic_evaluator import (
        EVALUATION_PROTOCOL,
        install_deterministic_evaluator,
    )
    from .training_compatible_agent import (
        AGENT_NAME as TRAINING_COMPATIBLE_AGENT,
    )
    from .training_compatible_agent import register_training_compatible_agent
    from .validated_user_simulator import (
        USER_NAME as VALIDATED_USER_SIMULATOR,
    )
    from .validated_user_simulator import register_validated_user_simulator
else:
    from deterministic_evaluator import (
        EVALUATION_PROTOCOL,
        install_deterministic_evaluator,
    )
    from training_compatible_agent import (
        AGENT_NAME as TRAINING_COMPATIBLE_AGENT,
    )
    from training_compatible_agent import register_training_compatible_agent
    from validated_user_simulator import (
        USER_NAME as VALIDATED_USER_SIMULATOR,
    )
    from validated_user_simulator import register_validated_user_simulator

LEGACY_EVALUATION_PROTOCOL = "tau_all_without_nl_assertions_v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_or_validate_domain_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
        return

    existing = json.loads(manifest_path.read_text())
    if "evaluation_protocol" not in existing:
        if existing.get("evaluation_type") != EvaluationType.ALL.value:
            raise RuntimeError(f"Cannot migrate unknown evaluation protocol in {manifest_path}")
        # A manifest without this field predates the replay/user-generation
        # repairs. Record its actual legacy protocol first; upgrading it must
        # still go through the explicit, compatibility-checked migration below.
        existing["evaluation_protocol"] = LEGACY_EVALUATION_PROTOCOL
        _write_json(manifest_path, existing)
    if existing == manifest:
        return

    legacy = deepcopy(manifest)
    legacy["evaluation_protocol"] = LEGACY_EVALUATION_PROTOCOL
    if legacy.get("user_simulator_mode") == "local":
        legacy_user_sampling = legacy["user_sampling"]
        legacy_user_sampling["max_tokens"] = 4096
        legacy_user_sampling.pop("generation_retries", None)
    allow_upgrade = os.environ.get("ALLOW_INFRASTRUCTURE_PROTOCOL_UPGRADE") == "1"
    if allow_upgrade and existing == legacy:
        backup = manifest_path.with_name("domain_manifest.pre_infrastructure_repair_v1.json")
        if backup.exists():
            if json.loads(backup.read_text()) != existing:
                raise RuntimeError(f"Domain protocol migration backup differs: {backup}")
        else:
            _write_json(backup, existing)
        _write_json(manifest_path, manifest)
        return

    raise RuntimeError(f"Domain protocol changed for {manifest_path.parent}; use a new RUN_DIR")


def _load_domain_results(domain_dir: Path) -> tuple[Results | None, int]:
    result_paths = sorted(domain_dir.glob("shard_*/results.json"))
    if not result_paths:
        return None, 0

    first = Results.load(result_paths[0])
    tasks_by_id = {task.id: task for task in first.tasks}
    simulations_by_key = {(sim.trial, sim.task_id, sim.seed): sim for sim in first.simulations}
    for result_path in result_paths[1:]:
        shard = Results.load(result_path)
        tasks_by_id.update({task.id: task for task in shard.tasks})
        for simulation in shard.simulations:
            simulations_by_key[(simulation.trial, simulation.task_id, simulation.seed)] = simulation

    combined = Results(
        info=first.info,
        tasks=list(tasks_by_id.values()),
        simulations=list(simulations_by_key.values()),
    )
    return combined, len(result_paths)


def _domain_summary(domain_dir: Path) -> dict[str, Any]:
    manifest_path = domain_dir / "domain_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing domain manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    results, completed_shards = _load_domain_results(domain_dir)

    expected_tasks = int(manifest["num_tasks"])
    num_trials = int(manifest["num_trials"])
    expected_simulations = expected_tasks * num_trials
    if results is None:
        metrics: dict[str, Any] = {
            "avg_reward": 0.0,
            "pass_hat_ks": {},
            "avg_agent_cost": 0.0,
            "total_simulations": 0,
            "total_tasks": 0,
            "infra_error_count": 0,
        }
        successful_simulations = 0
        total_duration_seconds = 0.0
    else:
        metrics = _json_safe(compute_metrics(results).model_dump(mode="json"))
        evaluated = [sim for sim in results.simulations if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR]
        successful_simulations = sum(1 for sim in evaluated if sim.reward_info is not None and is_successful(sim.reward_info.reward))
        total_duration_seconds = sum(sim.duration for sim in evaluated)

    completed_simulations = int(metrics["total_simulations"])
    summary = {
        "model_id": manifest["model_id"],
        "domain": manifest["domain"],
        "task_split": manifest["task_split"],
        "expected_tasks": expected_tasks,
        "expected_simulations": expected_simulations,
        "discovered_shards": completed_shards,
        "expected_shards": int(manifest["num_shards"]),
        "completed_simulations": completed_simulations,
        "successful_simulations": successful_simulations,
        "success_rate": (successful_simulations / completed_simulations if completed_simulations else 0.0),
        "progress": (completed_simulations / expected_simulations if expected_simulations else 0.0),
        "total_duration_seconds": total_duration_seconds,
        "complete": (completed_simulations == expected_simulations and int(metrics["infra_error_count"]) == 0),
        "metrics": metrics,
    }
    _write_json(domain_dir / "summary.json", summary)
    return summary


def _write_run_summary(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    results_root = run_dir / "results"
    if results_root.is_dir():
        for domain_manifest in sorted(results_root.glob("*/*/domain_manifest.json")):
            summaries.append(_domain_summary(domain_manifest.parent))

    _write_json(
        run_dir / "summary.json",
        {
            "domains": summaries,
            "complete": bool(summaries) and all(summary["complete"] for summary in summaries),
        },
    )

    csv_path = run_dir / "summary.csv"
    fieldnames = [
        "model_id",
        "domain",
        "expected_tasks",
        "completed_simulations",
        "expected_simulations",
        "successful_simulations",
        "success_rate",
        "avg_reward",
        "pass_hat_1",
        "pass_hat_2",
        "pass_hat_3",
        "infra_error_count",
        "progress",
        "complete",
    ]
    temporary = csv_path.with_suffix(f".csv.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            pass_hat = metrics.get("pass_hat_ks", {})
            writer.writerow(
                {
                    "model_id": summary["model_id"],
                    "domain": summary["domain"],
                    "expected_tasks": summary["expected_tasks"],
                    "completed_simulations": summary["completed_simulations"],
                    "expected_simulations": summary["expected_simulations"],
                    "successful_simulations": summary["successful_simulations"],
                    "success_rate": summary["success_rate"],
                    "avg_reward": metrics["avg_reward"],
                    "pass_hat_1": pass_hat.get("1", pass_hat.get(1, "")),
                    "pass_hat_2": pass_hat.get("2", pass_hat.get(2, "")),
                    "pass_hat_3": pass_hat.get("3", pass_hat.get(3, "")),
                    "infra_error_count": metrics["infra_error_count"],
                    "progress": summary["progress"],
                    "complete": summary["complete"],
                }
            )
    temporary.replace(csv_path)
    return summaries


def _agent_args(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "api_base": args.agent_base_url,
        "api_key": args.agent_api_key,
        "temperature": args.agent_temperature,
        "top_p": args.agent_top_p,
        "max_tokens": args.agent_max_tokens,
        "num_retries": args.llm_retries,
        "extra_body": {
            "top_k": args.agent_top_k,
            "min_p": args.agent_min_p,
            "chat_template_kwargs": {
                "enable_thinking": args.agent_enable_thinking,
            },
        },
    }
    if args.agent_protocol == "training_compatible":
        values["_training_decision_limit"] = args.training_decision_limit
        values["_training_invalid_action_limit"] = args.training_invalid_action_limit
    return values


def _user_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.user_simulator_mode == "local":
        if not args.user_base_url:
            raise RuntimeError("--user-base-url is required in local user mode")
        return {
            "api_base": args.user_base_url,
            "api_key": args.user_api_key,
            "temperature": args.user_temperature,
            "top_p": args.user_top_p,
            "presence_penalty": args.user_presence_penalty,
            "max_tokens": args.user_max_tokens,
            "_validation_retries": args.user_generation_retries,
            "num_retries": args.llm_retries,
            "extra_body": {
                "top_k": args.user_top_k,
                "min_p": args.user_min_p,
                "repetition_penalty": args.user_repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }

    from agent_system.environments.env_package.tau_bench.envs import (
        tau_user_simulator_llm_args,
    )

    return {
        **tau_user_simulator_llm_args(
            args.user_model,
            temperature=1.0,
            reasoning_enabled=False,
        ),
        "num_retries": args.llm_retries,
    }


def _user_sampling_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.user_simulator_mode == "local":
        return {
            "temperature": args.user_temperature,
            "top_p": args.user_top_p,
            "top_k": args.user_top_k,
            "min_p": args.user_min_p,
            "presence_penalty": args.user_presence_penalty,
            "repetition_penalty": args.user_repetition_penalty,
            "generation_retries": args.user_generation_retries,
            "max_tokens": args.user_max_tokens,
            "enable_thinking": True,
        }
    return {"temperature": 1.0, "reasoning_enabled": False}


def _required_user_api_key(user_model: str) -> str | None:
    model = str(user_model).strip().lower()
    if model == "deepseek" or model.startswith("deepseek/"):
        return "DEEPSEEK_API_KEY"
    if model.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    return None


def _run_domain(args: argparse.Namespace) -> None:
    required_api_key = _required_user_api_key(args.user_model) if args.user_simulator_mode == "remote" else None
    if required_api_key and not os.environ.get(required_api_key):
        raise RuntimeError(f"{required_api_key} is required for user model {args.user_model}")
    install_deterministic_evaluator()
    if args.user_simulator_mode == "local":
        register_validated_user_simulator()
    if args.agent_protocol == "training_compatible":
        register_training_compatible_agent()

    all_tasks = get_tasks(
        args.domain,
        task_split_name=args.task_split,
        num_tasks=args.num_tasks,
    )
    if not all_tasks:
        raise RuntimeError(f"No tasks found for {args.domain}/{args.task_split}")

    domain_dir = args.run_dir / "results" / args.model_id / args.domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(len(all_tasks) / args.tasks_per_shard)
    manifest = {
        "model_id": args.model_id,
        "domain": args.domain,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "agent_protocol": args.agent_protocol,
        "training_decision_limit": args.training_decision_limit,
        "training_invalid_action_limit": args.training_invalid_action_limit,
        "task_split": args.task_split,
        "num_tasks": len(all_tasks),
        "num_trials": args.num_trials,
        "tasks_per_shard": args.tasks_per_shard,
        "num_shards": num_shards,
        "evaluation_type": EvaluationType.ALL.value,
        "agent_model": f"openai/{args.model_id}",
        "agent_sampling": {
            "temperature": args.agent_temperature,
            "top_p": args.agent_top_p,
            "top_k": args.agent_top_k,
            "min_p": args.agent_min_p,
            "max_tokens": args.agent_max_tokens,
            "enable_thinking": args.agent_enable_thinking,
        },
        "user_model": args.user_model,
        "user_simulator_mode": args.user_simulator_mode,
        "user_sampling": _user_sampling_manifest(args),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
    }
    manifest_path = domain_dir / "domain_manifest.json"
    _write_or_validate_domain_manifest(manifest_path, manifest)

    config = TextRunConfig(
        domain=args.domain,
        task_set_name=args.domain,
        task_split_name=args.task_split,
        agent=(TRAINING_COMPATIBLE_AGENT if args.agent_protocol == "training_compatible" else "llm_agent"),
        llm_agent=f"openai/{args.model_id}",
        llm_args_agent=_agent_args(args),
        user=(VALIDATED_USER_SIMULATOR if args.user_simulator_mode == "local" else "user_simulator"),
        llm_user=args.user_model,
        llm_args_user=_user_args(args),
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        timeout=args.simulation_timeout,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        verbose_logs=args.verbose_logs,
        max_retries=args.task_retries,
        retry_delay=args.retry_delay,
        auto_resume=True,
        auto_review=False,
        hallucination_retries=0,
    )

    # The pinned tau2 package defaults to a ten-connection global LiteLLM
    # pool. Increase it to match native batch concurrency.
    litellm.client_session = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=args.max_concurrency * 2,
            max_connections=args.max_concurrency * 2,
        )
    )

    for shard_index in range(num_shards):
        start = shard_index * args.tasks_per_shard
        end = min(start + args.tasks_per_shard, len(all_tasks))
        shard_tasks = all_tasks[start:end]
        shard_dir = domain_dir / f"shard_{shard_index:04d}_{start:05d}_{end:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Running {args.model_id}/{args.domain} shard {shard_index + 1}/{num_shards}: tasks [{start}, {end})",
            flush=True,
        )
        run_tasks(
            config,
            shard_tasks,
            save_path=shard_dir / "results.json",
            save_dir=shard_dir / "simulations",
            evaluation_type=EvaluationType.ALL,
            console_display=False,
        )
        summary = _domain_summary(domain_dir)
        print(
            f"Progress {args.model_id}/{args.domain}: {summary['completed_simulations']}/{summary['expected_simulations']} simulations, success_rate={summary['success_rate']:.4f}",
            flush=True,
        )
        _write_run_summary(args.run_dir)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom", "telecom-workflow"],
        required=True,
    )
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--num-tasks", type=_positive_int)
    parser.add_argument("--num-trials", type=_positive_int, default=3)
    parser.add_argument("--tasks-per-shard", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-steps", type=_positive_int, default=200)
    parser.add_argument("--max-errors", type=_positive_int, default=10)
    parser.add_argument("--max-concurrency", type=_positive_int, default=32)
    parser.add_argument("--simulation-timeout", type=float, default=1800.0)
    parser.add_argument("--task-retries", type=_nonnegative_int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--llm-retries", type=_nonnegative_int, default=6)
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument("--agent-base-url", required=True)
    parser.add_argument(
        "--agent-protocol",
        choices=["strict_native", "training_compatible"],
        default="strict_native",
    )
    parser.add_argument("--training-decision-limit", type=_positive_int, default=200)
    parser.add_argument("--training-invalid-action-limit", type=_positive_int, default=10)
    parser.add_argument("--agent-api-key", default="local-tau-eval")
    parser.add_argument("--agent-temperature", type=float, default=0.6)
    parser.add_argument("--agent-top-p", type=float, default=0.95)
    parser.add_argument("--agent-top-k", type=int, default=20)
    parser.add_argument("--agent-min-p", type=float, default=0.0)
    parser.add_argument("--agent-max-tokens", type=_positive_int, default=4096)
    parser.add_argument(
        "--agent-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--user-model", default="openrouter/qwen/qwen3.6-27b")
    parser.add_argument(
        "--user-simulator-mode",
        choices=["local", "remote"],
        default="local",
    )
    parser.add_argument("--user-base-url")
    parser.add_argument("--user-api-key", default="local-tau-user")
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-top-k", type=int, default=20)
    parser.add_argument("--user-min-p", type=float, default=0.0)
    parser.add_argument("--user-presence-penalty", type=float, default=1.5)
    parser.add_argument("--user-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--user-generation-retries", type=_nonnegative_int, default=2)
    parser.add_argument("--user-max-tokens", type=_positive_int, default=8192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-domain")
    _add_run_arguments(run_parser)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run-domain":
        _run_domain(args)
        _write_run_summary(args.run_dir)
    else:
        summaries = _write_run_summary(args.run_dir)
        for summary in summaries:
            print(f"{summary['model_id']}/{summary['domain']}: {summary['completed_simulations']}/{summary['expected_simulations']} success_rate={summary['success_rate']:.4f} complete={summary['complete']}")


if __name__ == "__main__":
    main()
