"""Qualify a DeepSeek oracle policy on all Tau Airline/Retail training tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version as package_version
import hashlib
import json
from pathlib import Path
import random
import subprocess
import threading
from typing import Any

from agent_system.environments.env_package.tau_bench.actions import (
    ParsedAction,
    canonical_action,
    state_fingerprint,
    tau_messages_to_openai,
    to_tau_action,
    validate_tau_action,
)
from agent_system.environments.env_package.tau_bench.envs import (
    QUALIFICATION_PROTOCOL_VERSION,
    TAU2_COMMIT,
    TERMINAL_REWARD_PROTOCOL,
    compatibility_patch_sha256,
    make_tau_agent_gym_env,
)
from agent_system.environments.env_package.tau_bench.oracle import (
    ORACLE_PROTOCOL_VERSION,
    OpenRouterOracleClient,
    build_expert_messages,
)

DOMAINS = ("airline", "retail")
ORACLE_REASONING_EFFORT = "xhigh"
ORACLE_MAX_TOKENS = 4096
TRIAL_MAX_ATTEMPTS = 3


def qualification_protocol(args: argparse.Namespace, tau_commit: str) -> dict[str, Any]:
    return {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "terminal_reward_protocol": TERMINAL_REWARD_PROTOCOL,
        "tau2_commit": tau_commit,
        "tau2_compatibility_patch_sha256": compatibility_patch_sha256(),
        "expert_model": args.model,
        "user_llm": args.user_llm,
        "user_temperature": 0.0,
        "user_reasoning_enabled": False,
        "oracle_samples_per_state": args.oracle_samples,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "oracle_reasoning_effort": ORACLE_REASONING_EFFORT,
        "oracle_max_tokens": ORACLE_MAX_TOKENS,
        "trials_per_task": args.trials,
        "max_agent_steps": args.max_steps,
        "base_seed": args.seed,
        "trial_max_attempts": TRIAL_MAX_ATTEMPTS,
    }


def ensure_resume_protocol(path: Path, protocol: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            differing = sorted(
                key
                for key in set(existing) | set(protocol)
                if existing.get(key) != protocol.get(key)
            )
            raise RuntimeError(
                "qualification output contains records from a different protocol; "
                f"use a new OUTPUT_DIR or restore these fields: {differing}"
            )
        return
    path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def validate_action(action: ParsedAction, tools) -> ParsedAction:
    return validate_tau_action(action, tools)


def deterministic_seed(domain: str, task_id: str, trial: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{domain}:{task_id}:{trial}:{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_trial(
    *,
    domain: str,
    task_id: str,
    trial: int,
    base_seed: int,
    max_steps: int,
    user_llm: str,
    oracle: OpenRouterOracleClient,
    user_reasoning_enabled: bool = False,
) -> dict[str, Any]:
    seed = deterministic_seed(domain, task_id, trial, base_seed)
    rng = random.Random(seed)
    env = make_tau_agent_gym_env(
        domain=domain,
        task_id=task_id,
        # Tau counts message transitions internally; qualification caps agent decisions.
        max_steps=max_steps * 3 + 4,
        user_llm=user_llm,
        user_llm_args={
            "temperature": 0.0,
            "reasoning": {"enabled": user_reasoning_enabled},
        },
        all_messages_as_observation=False,
    )
    decisions = []
    illegal = False
    reward = 0.0
    terminated = False
    final_info: dict[str, Any] = {}
    try:
        _, info = env.reset(seed=seed)
        for step in range(max_steps):
            history = tau_messages_to_openai(env._agent.observation)
            tools = env._get_tools()
            tool_schemas = [tool.openai_schema for tool in tools]
            fingerprint = state_fingerprint(domain, task_id, history, tool_schemas)
            messages = build_expert_messages(
                policy=env._get_policy(),
                task=env._get_task().model_dump(mode="json"),
                history=history,
            )
            sampled = oracle.sample_oracle_set(
                state_fingerprint=fingerprint,
                messages=messages,
                tools=tool_schemas,
            )
            valid = [validate_action(ParsedAction(**action), tools) for action in sampled]
            valid = [action for action in valid if action.kind != "invalid"]
            deduplicated = {canonical_action(action): action for action in valid}
            valid = list(deduplicated.values())
            if not valid:
                illegal = True
                decisions.append(
                    {"step": step, "state_fingerprint": fingerprint, "oracle_set": [], "selected": None}
                )
                break
            selected = rng.choice(valid)
            decisions.append(
                {
                    "step": step,
                    "state_fingerprint": fingerprint,
                    "oracle_set": [action.to_dict() for action in valid],
                    "selected": selected.to_dict(),
                }
            )
            _, reward, terminated, _, final_info = env.step(to_tau_action(selected))
            if terminated:
                break
        if not terminated:
            try:
                _, reward, terminated, _, final_info = env.step(
                    json.dumps({"name": "done", "arguments": {}})
                )
            except Exception:
                pass
        simulation_run = final_info.get("simulation_run", "{}") if final_info else "{}"
        try:
            simulation_run = json.loads(simulation_run)
        except (TypeError, json.JSONDecodeError):
            simulation_run = {}
        return {
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "seed": seed,
            "success": bool(float(reward) > 0),
            "protocol_reward": float(reward),
            "illegal_tool_call": illegal,
            "steps": len(decisions),
            "decisions": decisions,
            "simulation_run": simulation_run if float(reward) > 0 else None,
        }
    except Exception as exc:
        return {
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "seed": seed,
            "success": False,
            "protocol_reward": 0.0,
            "illegal_tool_call": True,
            "steps": len(decisions),
            "decisions": decisions,
            "simulation_run": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_trial_with_retries(**kwargs) -> dict[str, Any]:
    record: dict[str, Any] | None = None
    for attempt in range(1, TRIAL_MAX_ATTEMPTS + 1):
        record = run_trial(**kwargs)
        record["attempt"] = attempt
        if "error" not in record:
            return record
    assert record is not None
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--user-llm", default="openrouter/qwen/qwen3.6-27b")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--oracle-samples", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-airline", type=int, default=20)
    parser.add_argument("--minimum-retail", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials != 4:
        raise ValueError("the formal qualification protocol requires exactly 4 trials per task")
    if args.oracle_samples != 3:
        raise ValueError("the formal qualification protocol requires exactly 3 oracle samples per state")
    from tau2.runner.helpers import load_tasks

    train_tasks = {domain: load_tasks(domain, "train") for domain in DOMAINS}
    test_tasks = {domain: load_tasks(domain, "test") for domain in DOMAINS}
    print(
        json.dumps(
            {
                "train": {domain: len(tasks) for domain, tasks in train_tasks.items()},
                "test": {domain: len(tasks) for domain, tasks in test_tasks.items()},
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tau_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__import__("tau2").__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:
        tau_commit = "unknown"
    if tau_commit != TAU2_COMMIT:
        raise RuntimeError(
            f"Tau source must be pinned to {TAU2_COMMIT}, got {tau_commit}"
        )
    protocol = qualification_protocol(args, tau_commit)
    ensure_resume_protocol(args.output_dir / "qualification_protocol.json", protocol)
    trial_path = args.output_dir / "qualification_trials.jsonl"
    cache_path = args.output_dir / "oracle_state_cache.jsonl"
    completed: dict[tuple[str, str, int], dict[str, Any]] = {}
    if trial_path.exists():
        for line in trial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("error"):
                continue
            completed[(record["domain"], record["task_id"], int(record["trial"]))] = record

    oracle = OpenRouterOracleClient(
        model=args.model,
        samples=args.oracle_samples,
        reasoning_effort=ORACLE_REASONING_EFFORT,
        max_tokens=ORACLE_MAX_TOKENS,
        cache_path=str(cache_path),
    )
    pending = [
        (domain, task.id, trial)
        for domain in DOMAINS
        for task in train_tasks[domain]
        for trial in range(args.trials)
        if (domain, task.id, trial) not in completed
    ]
    write_lock = threading.Lock()
    if trial_path.exists() and trial_path.stat().st_size > 0:
        with trial_path.open("rb+") as boundary_handle:
            boundary_handle.seek(-1, 2)
            if boundary_handle.read(1) != b"\n":
                boundary_handle.seek(0, 2)
                boundary_handle.write(b"\n")
    with trial_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_trial_with_retries,
                    domain=domain,
                    task_id=task_id,
                    trial=trial,
                    base_seed=args.seed,
                    max_steps=args.max_steps,
                    user_llm=args.user_llm,
                    oracle=oracle,
                    user_reasoning_enabled=False,
                ): (domain, task_id, trial)
                for domain, task_id, trial in pending
            }
            for future in as_completed(futures):
                record = future.result()
                key = (record["domain"], record["task_id"], int(record["trial"]))
                completed[key] = record
                with write_lock:
                    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                    handle.flush()
                print(
                    f"[{len(completed)}/{sum(len(v) for v in train_tasks.values()) * args.trials}] "
                    f"{key[0]} {key[1]} trial={key[2]} success={record['success']}"
                )

    stable_tasks = {domain: [] for domain in DOMAINS}
    task_results = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        for task in train_tasks[domain]:
            records = [completed[(domain, task.id, trial)] for trial in range(args.trials)]
            successes = sum(bool(record["success"]) for record in records)
            successful_illegal = any(
                bool(record["illegal_tool_call"]) for record in records if record["success"]
            )
            stable = successes == args.trials and not successful_illegal
            task_results[domain][task.id] = {
                "successes": successes,
                "trials": args.trials,
                "successful_illegal_tool_call": successful_illegal,
                "stable": stable,
            }
            if stable:
                stable_tasks[domain].append(task.id)

    manifest = {
        **protocol,
        "tau2_version": package_version("tau2"),
        "stable_rule": "protocol_success=4/4 and no successful trial contains an illegal tool call",
        "stable_tasks": stable_tasks,
        "test_tasks": {
            domain: [task.id for task in test_tasks[domain]] for domain in DOMAINS
        },
        "task_results": task_results,
        "oracle_cache": str(cache_path.resolve()),
        "oracle_stats": oracle.stats(),
    }
    manifest_path = args.output_dir / "qualification_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    counts = {domain: len(stable_tasks[domain]) for domain in DOMAINS}
    print(json.dumps({"stable": counts, "manifest": str(manifest_path)}, sort_keys=True))
    if counts["airline"] < args.minimum_airline or counts["retail"] < args.minimum_retail:
        raise SystemExit(
            "qualification threshold not met: "
            f"Airline={counts['airline']}/{args.minimum_airline}, "
            f"Retail={counts['retail']}/{args.minimum_retail}"
        )


if __name__ == "__main__":
    main()
