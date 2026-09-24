"""Small release interface; the environment adapters retain algorithm semantics."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from aopd.launch import RECIPES, ROOT, build_plan, launch_environment
from aopd.runtime import active_runtime, expand_runtime, load_yaml, resolve_roles
from aopd.services import local_services, owned_process, probe_roles, require_free_gpus, stop_owned


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run_plan(plan, runtime, run_dir):
    runtime = active_runtime(runtime, plan["active_roles"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from aopd.resume import check_resume

        identity = check_resume(plan, runtime, run_dir)
        environment = launch_environment(plan, resolve_roles(runtime))
        environment.setdefault("AOPD_LOCAL_API_KEY", "EMPTY")
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["TENSORBOARD_DIR"] = str(run_dir / "tensorboard")
        # Do not accidentally attach an experiment to another user's Ray cluster.
        environment.pop("RAY_ADDRESS", None)
        require_free_gpus(runtime)
        if plan.get("agent_service") and runtime.get("checkpoint"):
            from aopd.export import export_checkpoint

            export_checkpoint(runtime["checkpoint"], plan["agent_service"]["model_path"], python=environment["PYTHON"])
        service_runtime = deepcopy(runtime)
        if plan.get("agent_service"):
            service_runtime["services"]["evaluated-agent"] = plan["agent_service"]
            environment["TAU_USER_API_KEY"] = environment.get(plan["user_key_env"], "")
        (run_dir / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
        (run_dir / "runtime.yaml").write_text(__import__("yaml").safe_dump(runtime))
        (run_dir / "protocol.json").write_text(json.dumps(identity, indent=2) + "\n")
        previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with local_services(service_runtime, run_dir):
                probe_roles(resolve_roles(runtime), environment)
                with (run_dir / "launcher.log").open("ab", buffering=0) as log:
                    print(f"Launching {plan['recipe']}; log: {run_dir / 'launcher.log'}", flush=True)
                    for command in plan.get("commands", [plan.get("command")]):
                        process = owned_process(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            status = process.wait()
                        finally:
                            stop_owned(process)
                        if status:
                            break
                (run_dir / "exit_code").write_text(str(status) + "\n")
                if status:
                    raise SystemExit(status)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description="AOPD training and native Tau evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("recipe", choices=RECIPES)
    train.add_argument("--runtime", type=Path, required=True)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--check", action="store_true", help="Print resolved launch plan without API/GPU/file side effects")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", type=Path)
    evaluate = commands.add_parser("eval-tau")
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path)
    evaluate.add_argument("--check", action="store_true")
    evaluate.add_argument("--smoke", action="store_true")
    evaluate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        from aopd.export import export_checkpoint

        print(export_checkpoint(args.checkpoint, args.output))
        return
    runtime = expand_runtime(load_yaml(args.runtime))
    recipe = getattr(args, "recipe", "tau-eval")
    run_dir = (args.run_dir or ROOT / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + recipe)).resolve()
    if args.command == "train":
        plan = build_plan(args.recipe, runtime, run_dir, smoke=args.smoke, resume=args.resume)
    else:
        from aopd.evaluate import build_eval_plan

        plan = build_eval_plan(runtime, run_dir, smoke=args.smoke)
    if args.check:
        print(json.dumps(plan, indent=2))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        parser.error("run directory is not empty; use a new directory or explicit --resume")
    run_plan(plan, runtime, run_dir)


if __name__ == "__main__":
    main()
