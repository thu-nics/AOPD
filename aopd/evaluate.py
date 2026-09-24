"""Native Tau evaluation plans with the same explicit service allocation as training."""

import sys
from pathlib import Path

from aopd.launch import ROOT
from aopd.runtime import active_runtime, resolve_roles, validate_generation, validate_runtime


def build_eval_plan(runtime, run_dir, *, smoke=False):
    runtime = active_runtime(runtime, {"user"})
    if "evaluated-agent" in runtime.get("services", {}):
        raise ValueError("evaluated-agent is a reserved local service name")
    validate_runtime(runtime)
    roles = resolve_roles(runtime)
    if "user" not in roles:
        raise ValueError("Tau evaluation requires a user role")
    validate_generation(roles, evaluation=True)
    user, student = roles["user"], runtime["student"]
    config = runtime.get("evaluation", {})
    if extra := set(config) - {"domains", "model_id", "split", "trials", "response_tokens", "max_steps", "concurrency", "seed"}:
        raise ValueError(f"unsupported evaluation settings: {sorted(extra)}")
    domains = config.get("domains", ["airline", "retail", "telecom"])
    if not domains or any(d not in {"airline", "retail", "telecom"} for d in domains):
        raise ValueError("evaluation.domains must select airline/retail/telecom")
    model_id = config.get("model_id", "student")
    if not model_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in model_id):
        raise ValueError("model_id must be a simple identifier, not a path")
    python = runtime.get("python", sys.executable)
    tau_root = runtime.get("sources", {}).get("tau", str(ROOT.parent / "tau2-bench"))
    port = int(student.get("port", 8100))
    if any(s.get("port") == port for s in runtime.get("services", {}).values() if s["mode"] == "local"):
        raise ValueError("student port conflicts with a local service")
    model_path = str(Path(run_dir) / "models/student") if runtime.get("checkpoint") else runtime["model"]
    agent = {
        "model": model_id,
        "mode": "local",
        "model_path": model_path,
        "gpus": student["gpus"],
        "tp": student.get("tp", 1),
        "dp": len(student["gpus"]) // student.get("tp", 1),
        "port": port,
        "python": student.get("python", python),
        "max_model_len": student.get("max_model_len", 40960),
        "memory_utilization": student.get("memory_utilization", 0.8),
    }
    commands = []
    generation = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_tokens": 8192, "enable_thinking": True, **user["generation"]}
    evaluation = {
        "domains": domains,
        "model_id": model_id,
        "split": config.get("split", "base"),
        "trials": 1 if smoke else config.get("trials", 4),
        "seed": config.get("seed", 300),
        "max_steps": 4 if smoke else config.get("max_steps", 200),
        "num_tasks": 1 if smoke else None,
        "simulation_timeout": 600 if smoke else 1800,
        "max_model_len": agent["max_model_len"],
        "agent_generation": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_tokens": config.get("response_tokens", 4096), "enable_thinking": True},
        "user_generation": generation,
    }
    for domain in domains:
        command = [
            python,
            str(ROOT / "examples/tau_bench/eval/native_eval.py"),
            "run-domain",
            "--run-dir",
            str(run_dir),
            "--model-id",
            model_id,
            "--domain",
            domain,
            "--task-split",
            evaluation["split"],
            "--num-trials",
            str(evaluation["trials"]),
            "--seed",
            str(evaluation["seed"]),
            "--agent-base-url",
            f"http://127.0.0.1:{port}/v1",
            "--max-steps",
            str(evaluation["max_steps"]),
            "--simulation-timeout",
            str(evaluation["simulation_timeout"]),
            "--max-concurrency",
            str(config.get("concurrency", 32)),
            "--user-simulator-mode",
            "remote",
            "--user-model",
            "openai/" + user["model"].removeprefix("openai/"),
            "--user-base-url",
            user["base_url"],
            "--user-provider",
            user["provider"],
        ]
        for role in ("agent", "user"):
            for key, value in evaluation[role + "_generation"].items():
                if key == "enable_thinking":
                    command.append(f"--{role}-enable-thinking" if value else f"--no-{role}-enable-thinking")
                else:
                    command.extend(["--" + role + "-" + key.replace("_", "-"), str(value)])
        if evaluation["num_tasks"] is not None:
            command.extend(["--num-tasks", str(evaluation["num_tasks"])])
        commands.append(command)
    return {
        "recipe": "tau-eval",
        "active_roles": ["user"],
        "commands": commands,
        "agent_service": agent,
        "evaluation": evaluation,
        "env": {"PYTHON": python, "TAU2_ROOT": tau_root, "TAU2_DATA_DIR": str(Path(tau_root) / "data"), "PYTHONPATH": str(ROOT) + ":" + str(Path(tau_root) / "src")},
        "user_key_env": user["api_key_env"],
        "source_root": tau_root,
    }
