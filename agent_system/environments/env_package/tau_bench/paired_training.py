"""Own two isolated Tau experiments and their shared inference services.

Whole-machine cleanup requires an explicit flag; no implicit resume or shared writable caches.
Runtime identities come from environment variables, not cluster-specific paths.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import psutil

from .matcher_profiles import MESSAGE_CONCISE, TOOL_CONCISE

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = "tau-three-domain-pairs-v1"
GPU_LAYOUTS = {"full-a1": {"full": [2, 3, 4, 5], "a1": [6, 7]}, "a4-a5": {"a4": [4, 5], "a5": [6, 7]}, "self-s1-s2": {"s1": [2, 3], "s2": [4, 5]}}


def assigned_gpus(plan):
    return {gpu for service in plan["services"] for gpu in service["gpus"]} | {gpu for settings in plan["experiments"].values() for gpu in settings["CUDA_VISIBLE_DEVICES"].split(",")}


def is_compute_service(name, command):
    """Recognize detached control processes, including ones no longer on a GPU."""
    if name.startswith(("VLLM::", "ray::")) or name in {"raylet", "gcs_server"}:
        return True
    modules = {"vllm.entrypoints.openai.api_server", "ray.scripts.scripts", "verl.trainer.main_ppo", "agent_system.environments.env_package.tau_bench.paired_training"}
    if any(arg in modules for arg in command):
        return True
    for arg in command:
        if Path(arg).name in {"vllm", "ray", "torchrun"} and not arg.startswith("-"):
            return True
        if arg.endswith("/examples/tau_bench/train/run.sh") or any(part in arg for part in ("/ray/dashboard/", "/ray/_private/workers/", "/ray/autoscaler/")):
            return True
    return False


def cleanup_machine_compute(*, timeout=15):
    """Explicit dedicated-machine mode: stop ALL GPU jobs and Ray/vLLM services.

    Does not depend on old ownership tags or manifests. Protect this launcher's
    ancestry and login/session infrastructure; never delete run data or caches.
    """
    protected = {1, os.getpid(), *(p.pid for p in psutil.Process().parents())}
    print("Dedicated-machine cleanup: stopping ALL existing GPU jobs and Ray/vLLM/training services on this machine", flush=True)

    def targets():
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
        gpu_pids = {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}
        selected = {}
        for process in psutil.process_iter():
            try:
                if process.pid in protected or process.name().startswith(("sshd", "tmux")) or process.status() == psutil.STATUS_ZOMBIE:
                    continue
                if process.pid in gpu_pids or is_compute_service(process.name(), process.cmdline()):
                    selected[process.pid] = process
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if process.pid in gpu_pids:
                    raise RuntimeError(f"cannot inspect GPU process {process.pid}; run cleanup with appropriate privileges") from None
        # Capture descendants before terminating their parent, including CPU
        # helpers that would otherwise be reparented when a server exits.
        for process in list(selected.values()):
            try:
                for child in process.children(recursive=True):
                    if child.pid not in protected and not child.name().startswith(("sshd", "tmux")):
                        selected[child.pid] = child
            except psutil.NoSuchProcess:
                pass
        return list(selected.values())

    remaining = targets()
    for process in remaining:
        try:
            print(f"Cleanup TERM: PID {process.pid} ({process.name()})", flush=True)
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(remaining, timeout=timeout)
    # Include newly spawned workers as well as TERM-resistant initial targets.
    remaining = {p.pid: p for p in [*alive, *targets()]}
    for process in remaining.values():
        try:
            print(f"Cleanup KILL: PID {process.pid}", flush=True)
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(list(remaining.values()), timeout=5)
    # CUDA contexts may take a moment to release after the process exits.
    deadline = time.monotonic() + 30
    while True:
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=15)
        if all(int(line) <= 512 for line in used.splitlines()):
            print("Dedicated-machine cleanup complete: all GPUs released", flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU memory still occupied after whole-machine cleanup (MiB per GPU: {used.split()}); inspect nvidia-smi for inaccessible processes")
        time.sleep(1)


def cleanup_stale_runs(gpu_ids, *, timeout=15):
    """Reap local, tagged leftovers only when their owning run is unlocked.

    Older launchers used RUN_DIR rather than TAU_PAIR_RUN_DIR. Both are
    checked against a paired-run manifest under this repository's runs/.
    Never infer ownership from a port, executable name, or GPU usage alone.
    """
    groups = {}
    for process in psutil.process_iter():
        try:
            env = process.environ()
            token = env.get("TAU_PAIR_OWNER_TOKEN", "")
            raw_dir = env.get("TAU_PAIR_RUN_DIR") or env.get("RUN_DIR")
            if process.pid == os.getpid() or process.uids().real != os.getuid() or len(token) != 32 or any(c not in "0123456789abcdef" for c in token) or not raw_dir:
                continue
            path = (ROOT / raw_dir).resolve()
            for dest in (path, path.parent):
                if not dest.is_relative_to((ROOT / "runs").resolve()):
                    continue
                try:
                    plan = json.loads((dest / "protocol.json").read_text())
                    if plan["protocol"] != PROTOCOL or plan["pair"] not in GPU_LAYOUTS:
                        continue
                    if path != dest and str(path) not in {s["RUN_DIR"] for s in plan["experiments"].values()}:
                        continue
                    # Don't stop a run that also owns GPUs outside this launch.
                    if not assigned_gpus(plan) or not assigned_gpus(plan) <= set(gpu_ids):
                        continue
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    continue
                groups.setdefault(dest, set()).add(token)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    cleaned = 0
    for dest, tokens in groups.items():
        with (dest / ".launcher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"Cleanup skipped active paired run: {dest}", flush=True)
                continue

            # Rescan under the run lock to include reparented/late Ray workers.
            def members(tokens=tokens):
                result = []
                for process in psutil.process_iter():
                    try:
                        if process.pid != os.getpid() and process.uids().real == os.getuid() and process.environ().get("TAU_PAIR_OWNER_TOKEN") in tokens and process.status() != psutil.STATUS_ZOMBIE:
                            result.append(process)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                return result

            remaining = members()
            print(f"Cleaning {len(remaining)} leftover processes from stopped run: {dest}", flush=True)
            for process in remaining:
                try:
                    process.terminate()  # psutil checks birth time against PID reuse.
                    cleaned += 1
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(remaining, timeout=timeout)
            remaining = members()
            for process in remaining:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(remaining, timeout=5)
            if members():
                raise RuntimeError(f"paired-run leftovers could not be stopped: {dest}")
    print(f"Startup cleanup complete: {cleaned} stale processes signalled; unrelated processes left untouched", flush=True)


@contextmanager
def uninterrupted_cleanup():
    """A second Ctrl+C must not abandon the remaining owned children."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def server_command(python, model_path, model_name, port, *, shared):
    return [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "65536" if shared else "40960",
        "--max-num-seqs",
        "32" if shared else "64",
        "--max-num-batched-tokens",
        "32768" if shared else "65536",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml" if shared else "hermes",
        "--reasoning-parser",
        "qwen3",
        *(["--language-model-only"] if shared else []),
    ]


def build_plan(args, environ):
    self_teacher = args.pair == "self-s1-s2"
    if self_teacher and getattr(args, "local_teacher", False):
        raise ValueError("self teacher shares student weights; do not deploy a separate teacher")
    local_teacher = args.pair == "a4-a5" or getattr(args, "local_teacher", False)
    layout = GPU_LAYOUTS[args.pair]
    if args.pair == "full-a1" and local_teacher:
        layout = {"full": [4, 5], "a1": [6, 7]}
    required = ["MODEL_PATH", "TAU_SHARED_MODEL_PATH", "TAU2_ROOT"]
    if not self_teacher:
        required += ["TAU_LOCAL_TEACHER_PATH"] if local_teacher else ["TAU_TEACHER_API_BASE"]
    for key in required:
        if not environ.get(key):
            raise ValueError(f"set {key}")
    gpu_ids = environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError("paired training requires eight distinct CUDA_VISIBLE_DEVICES")
    run_dir = Path(args.run_dir).resolve()
    python = environ.get("PYTHON", "python")
    server_python = environ.get("TAU_SERVER_PYTHON", "python")
    shared_model = environ.get("TAU_SHARED_MODEL", "qwen3.8-27b")
    teacher_model = environ.get("TAU_TEACHER_MODEL", "qwen3-32b")
    teacher_concurrency = int(environ.get("TAU_TEACHER_MAX_CONCURRENT_REQUESTS", "32"))
    if teacher_concurrency < 1:
        raise ValueError("TAU_TEACHER_MAX_CONCURRENT_REQUESTS must be positive")
    shared_base = f"http://127.0.0.1:{args.shared_port}/v1"
    teacher_base = "self://rollout" if self_teacher else f"http://127.0.0.1:{args.teacher_port}/v1" if local_teacher else environ["TAU_TEACHER_API_BASE"]
    imports = json.loads(environ.get("TAU_TEACHER_CACHE_IMPORT_PATHS", "[]"))
    if not isinstance(imports, list) or any(not isinstance(p, str) for p in imports):
        raise ValueError("TAU_TEACHER_CACHE_IMPORT_PATHS must be a JSON list of paths")
    if self_teacher and imports:
        raise ValueError("self teacher cannot import external teacher cache votes")
    for path in imports:
        if not Path(path).is_file():
            raise ValueError(f"missing cache import: {path}")
    common = {
        "METHOD": "agentic_opd",
        "TAU_TEACHER_SOURCE": "external",
        "MODEL_PATH": str(Path(environ["MODEL_PATH"]).resolve()),
        "TAU2_ROOT": str(Path(environ["TAU2_ROOT"]).resolve()),
        "TAU2_DATA_DIR": str(Path(environ["TAU2_ROOT"]).resolve() / "data"),
        "PYTHON": python,
        "TAU_USER_MODEL": f"openai/{shared_model}",
        "TAU_USER_API_BASE": shared_base,
        "TAU_USER_API_KEY_ENV": "TAU_PAIR_LOCAL_API_KEY",
        "TAU_USER_REASONING_ENABLED": "false",
        "TAU_USER_TEMPERATURE": "0.7",
        "TAU_USER_TOP_P": "0.8",
        "TAU_USER_MAX_TOKENS": "8192",
        "TAU_MATCHER_PROVIDER": "openai-compatible",
        "TAU_MATCHER_PROFILE": "qwen38_concise",
        "TAU_MATCHER_MODEL": shared_model,
        "TAU_MATCHER_API_BASE": shared_base,
        "TAU_MATCHER_API_KEY_ENV": "TAU_PAIR_LOCAL_API_KEY",
        "TAU_MATCHER_ENABLE_THINKING": "false",
        "TAU_MATCHER_MAX_TOKENS": "32768",
        "TAU_MATCHER_REASONING_EFFORT": "null",
        "TAU_MATCHER_MAX_CONCURRENT_REQUESTS": "8",
        "TAU_TEACHER_MODEL": teacher_model,
        "TAU_TEACHER_API_BASE": teacher_base,
        "TAU_TEACHER_API_KEY_ENV": "TAU_PAIR_LOCAL_API_KEY" if local_teacher else environ.get("TAU_TEACHER_API_KEY_ENV", "TAU_TEACHER_API_KEY"),
        # Two experiments share the local teacher's 64-sequence engine.
        "TAU_TEACHER_MAX_CONCURRENT_REQUESTS": str(teacher_concurrency),
        "TAU_TEACHER_TEMPERATURE": "0.6",
        "TAU_TEACHER_TOP_P": "0.95",
        "TAU_TEACHER_TOP_K": "20",
        "TAU_TEACHER_MIN_P": "0.0",
        "TAU_TEACHER_MAX_TOKENS": "8192",
        "TAU_TEACHER_VALIDITY_MAX_RETRIES": "2",
        "TAU_TEACHER_CACHE_IMPORT_PATHS": json.dumps(imports),
        "TAU_TRANSFER_REWARD_GUARD": "true",
        "TAU_MASK_MATCHER_REQUIRED_GROUPS": "false",
        "TAU_USE_PRIVILEGED_TEACHER_CONTEXT": "false",
        "AIRLINE_TRAJ": "8",
        "RETAIL_TRAJ": "8",
        "TELECOM_TRAJ": "8",
        "TRAIN_STEPS": str(args.steps),
        "TRAIN_MAX_STEPS": "20",
        "TAU_INTERNAL_MAX_STEPS": "200",
        "ROLLOUT_N": "4",
        "PPO_MINI_BATCH": "32",
        "LR": "1e-6",
        "WARMUP_STEPS": "0",
        "TP_SIZE": "1",
        "SP_SIZE": "2",
        "PPO_MAX_TOKENS_PER_GPU": "16384",
        "LOGPROB_MAX_TOKENS_PER_GPU": "16384",
        "PPO_MICRO": "1",
        "LOGPROB_MICRO": "1",
        # Leave headroom for colocated actor/optimizer allocations after the
        # first update; a standalone-server KV budget is too aggressive here.
        "GPU_MEM_UTIL": environ.get("GPU_MEM_UTIL", "0.65"),
        "MAX_PROMPT": "24576",
        "MAX_RESPONSE": "4096",
        "MAX_MODEL_LEN": "32768",
        "STATE_GROUP_ADVANTAGE_MODE": "mean_then_batch_whiten",
        "COMPACT_STATE_GROUP_ROWS": "true",
        "MIN_EFFECTIVE_STATE_GROUPS": "1",
        "MAX_GEN_BATCHES": "1",
        "FREQUENCY_BONUS_SCALE": "0.5",
        "SAVE_FREQ": "10",
        "TEST_FREQ": "-1",
        "VAL_BEFORE_TRAIN": "false",
        "MAX_CKPTS": "null",
        "VAL_BATCH": "16",
        "VALIDATION_DOMAINS": "airline,retail,telecom",
        "VALIDATION_SPLIT": "test",
        "VALIDATION_TRIALS": "1",
        "VALIDATION_NUM_TASKS": "",
        "RAY_CPUS": str(args.cpus_per_run),
        # Bound student prefill activations, not the context/task/candidate
        # budget. 65k permitted 42k+ prefill steps that OOMed all four variants.
        "MAX_NUM_BATCHED_TOKENS": environ.get("MAX_NUM_BATCHED_TOKENS", "49152"),
        "ENABLE_THINKING": "True",
        "DATA_TRUNCATION": "error",
        "SMOKE": "1" if args.smoke else "0",
        # Keep the same balanced smoke batch for all four variants. In
        # state-group mode task count itself (not task count * N) must divide
        # across both the 4-GPU Full run and the 2-GPU ablations.
        "SMOKE_AIRLINE_TRAJ": "4",
        "SMOKE_RETAIL_TRAJ": "4",
        "SMOKE_TELECOM_TRAJ": "4",
        "SMOKE_PPO_MINI_BATCH": "32",
        "SMOKE_MAX_STEPS": "3",
    }
    if args.smoke:
        common.update(VALIDATION_DOMAINS="airline,retail")
    if self_teacher:
        common.update(
            TAU_TEACHER_SOURCE="self",
            TAU_TEACHER_MODEL="self-rollout-policy",
            TAU_TEACHER_API_KEY_ENV="TAU_PAIR_LOCAL_API_KEY",
            TAU_TEACHER_MAX_TOKENS="4096",
            TAU_SELF_EXTRA_PROMPT_TOKENS="4096",
            TAU_SELF_PRIVILEGE_MODE=environ.get("TAU_SELF_PRIVILEGE_MODE", "answer_conditioned"),
            TAU_SELF_CUSTOMER_BRIEFS=environ.get("TAU_SELF_CUSTOMER_BRIEFS", ""),
            SMOKE_TRAIN_STEPS="2",
            SMOKE_SAVE_FREQ="1",
            SMOKE_MAX_PROMPT="24576",
            SMOKE_MAX_RESPONSE="4096",
        )
    experiments = {}
    for variant, slots in layout.items():
        dest = run_dir / variant
        settings = dict(
            common,
            TAU_ABLATION="full" if self_teacher else variant,
            N_GPUS=str(len(slots)),
            CUDA_VISIBLE_DEVICES=",".join(gpu_ids[i] for i in slots),
            RUN_DIR=str(dest),
            # Ray starts BEFORE run.sh exports this variable. Its workers
            # inherit the head's environment, not the later shell exports.
            TENSORBOARD_DIR=str(dest / "tensorboard"),
            DATA_DIR=str(dest / "data"),
            ORACLE_CACHE=str(dest / "cache/teacher.jsonl"),
            ORACLE_MATCHER_CACHE=str(dest / "cache/matcher.jsonl"),
            TEACHER_REWARD_MODE="appearance" if variant == "a4" else "frequency_weighted",
        )
        if self_teacher:
            settings["TAU_USE_PRIVILEGED_TEACHER_CONTEXT"] = "true" if variant == "s2" else "false"
        experiments[variant] = settings
    services = [{"name": "shared", "gpus": gpu_ids[:2], "api_base": shared_base, "model": shared_model, "command": server_command(server_python, environ["TAU_SHARED_MODEL_PATH"], shared_model, args.shared_port, shared=True)}]
    if local_teacher:
        services.append({"name": "teacher", "gpus": gpu_ids[2:4], "api_base": teacher_base, "model": teacher_model, "command": server_command(server_python, environ["TAU_LOCAL_TEACHER_PATH"], teacher_model, args.teacher_port, shared=False)})
    return {
        "protocol": PROTOCOL,
        "pair": args.pair,
        "services": services,
        "experiments": experiments,
        "matcher_prompt_sha256": hashlib.sha256((MESSAGE_CONCISE + TOOL_CONCISE).encode()).hexdigest(),
        "loss_normalization": "global-minibatch-v1",
        "schedule": "official-train-8-8-8-ordered-seed0",
        "final_eval": "manual official test airline20/retail40/telecom40; 4 trials",
    }


def api_request(base, route, key, payload=None):
    req = Request(base.rstrip("/") + route, data=None if payload is None else json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=180 if payload else 5) as response:
        return json.load(response)


def check_model(base, model, key):
    available = [item["id"] for item in api_request(base, "/models", key)["data"]]
    if model not in available:
        raise RuntimeError(f"{base} serves {available}, not {model}")


def preflight(base, model, key, *, thinking):
    check_model(base, model, key)
    result = api_request(base, "/chat/completions", key, {"model": model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 256, "temperature": 0.6 if thinking else 0.7, "chat_template_kwargs": {"enable_thinking": thinking}})
    choice = result["choices"][0]
    msg = choice["message"]
    # This short probe checks generation, not a complete teacher action. A
    # thinking model can spend its entire probe budget in reasoning; newer
    # vLLM returns that text as `reasoning`, older servers as `reasoning_content`.
    fields = ("content", "reasoning", "reasoning_content") if thinking else ("content",)
    output_field = next((field for field in fields if isinstance(msg.get(field), str) and msg[field].strip()), None)
    if output_field is None:
        raise RuntimeError(f"empty preflight response from {base}: model={model}, finish_reason={choice.get('finish_reason')}, message_fields={sorted(msg)}")
    print(f"Preflight OK: {model}, output={output_field}, finish_reason={choice.get('finish_reason')}", flush=True)


class OwnedProcess:
    def __init__(self, command, env, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log.open("a")
        self.owner_token = uuid.uuid4().hex
        self.proc = subprocess.Popen(command, env=dict(env, TAU_PAIR_OWNER_TOKEN=self.owner_token), cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children = {}
        self.closed = False

    def track(self):
        try:
            for child in psutil.Process(self.proc.pid).children(recursive=True):
                self.children[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            pass

    def stop(self):
        if self.closed:
            return
        self.track()
        # Catch detached Ray daemons even if they reparented between polls.
        # Read ownership locally; never log complete process environments.
        for child in psutil.process_iter():
            try:
                if child.pid != self.proc.pid and child.environ().get("TAU_PAIR_OWNER_TOKEN") == self.owner_token:
                    self.children[child.pid] = child.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Only our own newly-created process group. The leader may already be
        # gone while Ray/vLLM children are still alive.
        for sig, delay in ((signal.SIGTERM, 15), (signal.SIGKILL, 0)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            if delay:
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    try:
                        os.killpg(self.proc.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.5)
        self.proc.wait()
        # Ray daemons can create a new session. Restrict cleanup to descendants
        # actually observed under this launcher and verify PID birth time.
        remaining = []
        for pid, born in self.children.items():
            try:
                child = psutil.Process(pid)
                if child.create_time() == born and child.is_running():
                    child.terminate()
                    remaining.append(child)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=10)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        self.log.close()
        self.closed = True


def start_ray_head(python, env, temp, log, *, num_cpus, num_gpus, object_store_memory=None):
    """Use Ray's CLI to isolate the agent HTTP port, unavailable via ray.init.

    Workers bind port 0 so the OS allocates each live listener atomically,
    instead of two Ray heads independently allocating the same default range.
    Only our own fresh temp directory may supply the address; never connect
    via the global 'auto' file.
    """
    with socket.socket() as sock, socket.socket() as metrics_sock:
        sock.bind(("0.0.0.0", 0))
        metrics_sock.bind(("0.0.0.0", 0))
        agent_port = sock.getsockname()[1]
        dashboard_metrics_port = metrics_sock.getsockname()[1]
    command = [
        python,
        "-m",
        "ray.scripts.scripts",
        "start",
        "--head",
        "--block",
        "--port=0",
        "--min-worker-port=0",
        "--max-worker-port=0",
        # These are fixed-size single-node clusters. The unused autoscaler
        # monitor otherwise competes for its fixed Prometheus port 44217.
        "--no-monitor",
        "--include-dashboard=false",
        "--disable-usage-stats",
        f"--dashboard-agent-listen-port={agent_port}",
        f"--num-cpus={num_cpus}",
        f"--num-gpus={num_gpus}",
        f"--temp-dir={temp}",
    ]
    if object_store_memory is not None:
        command.append(f"--object-store-memory={object_store_memory}")
    # Even --include-dashboard=false retains a minimal dashboard process whose
    # Prometheus endpoint defaults to 44227. Give each head its own endpoint.
    process = OwnedProcess(command, dict(env, DASHBOARD_METRIC_PORT=str(dashboard_metrics_port)), log)
    address_file = Path(temp) / "ray_current_cluster"
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            process.track()
            if process.proc.poll() is not None:
                raise RuntimeError(f"isolated Ray head exited; inspect {log}")
            if address_file.is_file():
                address = address_file.read_text().strip()
                if address:
                    return process, address
            time.sleep(0.5)
        raise TimeoutError(f"isolated Ray head startup timed out; inspect {log}")
    except BaseException:
        process.stop()
        raise


def training_command(address):
    # The head owns resources. ray.init rejects CPU/GPU counts on connection.
    # Keep TensorBoard enabled in paired smoke runs too, so they exercise the
    # same worker-side logging path as the full experiments.
    return ["bash", str(ROOT / "examples/tau_bench/train/run.sh"), f"+ray_init.address={address}", "ray_init.num_cpus=null", "actor_rollout_ref.actor.loss_normalization=global-minibatch-v1", 'trainer.logger=["console","tensorboard"]']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=GPU_LAYOUTS, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cpus-per-run", type=int, default=48)
    parser.add_argument("--shared-port", type=int, default=8180)
    parser.add_argument("--teacher-port", type=int, default=8181)
    parser.add_argument("--local-teacher", action="store_true", help="Deploy the Full/A1 teacher on GPUs 2–3; Full uses GPUs 4–5. A4/A5 already use a local teacher.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--cleanup-stale", action="store_true", help="Before GPU allocation, clean tagged leftovers from unlocked paired runs on these GPUs. Never stop active or unrelated jobs.")
    cleanup.add_argument("--cleanup-machine", action="store_true", help="Dedicated machine only: stop ALL existing GPU jobs, Ray/vLLM servers and training launchers, even untagged or active ones. Preserve SSH/tmux and run files.")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.cpus_per_run < 1 or args.shared_port == args.teacher_port:
        parser.error("positive steps/CPUs and distinct ports are required")
    plan = build_plan(args, os.environ)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # Pin the exact task content/schedule and selected implementation on resume.
    # Running this before starting GPUs also catches source/data installation drift.
    from .envs import validate_tau_source

    sample = next(iter(plan["experiments"].values()))
    os.environ["TAU2_DATA_DIR"] = sample["TAU2_DATA_DIR"]
    from tau2.runner.helpers import load_tasks

    plan["source"] = validate_tau_source(sample["TAU2_ROOT"])
    plan["train_tasks_sha256"] = hashlib.sha256(json.dumps({domain: [t.model_dump(mode="json") for t in load_tasks(domain, "train")] for domain in ("airline", "retail", "telecom")}, sort_keys=True).encode()).hexdigest()
    if args.pair == "self-s1-s2":
        from .customer_briefs import load_briefs
        from .self_teacher import audit_task_budgets

        plan["customer_briefs_sha256"] = load_briefs(sample["TAU_SELF_CUSTOMER_BRIEFS"]).fingerprint
        plan["self_context_audit"] = audit_task_budgets(sample["TAU2_ROOT"], sample["MODEL_PATH"], briefs_path=sample["TAU_SELF_CUSTOMER_BRIEFS"], privilege_mode=sample["TAU_SELF_PRIVILEGE_MODE"])
    files = [
        "examples/tau_bench/train/run.sh",
        "examples/tau_bench/train/prepare_data.py",
        "verl/trainer/config/tau_agentic_opd.yaml",
        "agent_system/environments/env_package/tau_bench/envs.py",
        "agent_system/environments/env_package/tau_bench/oracle.py",
        "agent_system/environments/env_package/tau_bench/ablations.py",
        "agent_system/environments/env_package/tau_bench/paired_training.py",
        "agent_system/environments/env_package/tau_bench/matcher_profiles.py",
    ]
    if args.pair == "self-s1-s2":
        files += [
            "agent_system/environments/env_package/tau_bench/self_teacher.py",
            "agent_system/environments/env_package/tau_bench/self_teacher_context.py",
            "agent_system/environments/env_package/tau_bench/self_teacher_privilege.py",
            "agent_system/environments/env_package/tau_bench/self_teacher_answers.py",
            "agent_system/environments/env_package/tau_bench/customer_briefs.py",
            "agent_system/environments/env_package/tau_bench/self_teacher_telecom_manual.md",
            "agent_system/multi_turn_rollout/rollout_loop.py",
            "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
        ]
    plan["implementation_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files}
    if args.cleanup_machine:
        cleanup_machine_compute()
    elif args.cleanup_stale:
        cleanup_stale_runs(assigned_gpus(plan))
    dest = Path(args.run_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / ".launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dest / "protocol.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != plan:
            raise RuntimeError("existing run: explicit --resume with identical protocol is required")
    elif args.resume:
        raise RuntimeError("cannot resume without protocol.json")
    else:
        if any((dest / name).exists() for name in plan["experiments"]):
            raise RuntimeError("refusing to overwrite existing experiment directories")
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    # No global proxy/port/Ray values may leak from another launcher.
    base_env = dict(os.environ)
    for key in ("RAY_ADDRESS", "RAY_NAMESPACE", "VLLM_PORT", "VLLM_HOST_IP", "MASTER_ADDR", "MASTER_PORT"):
        base_env.pop(key, None)
    base_env["TAU_PAIR_LOCAL_API_KEY"] = "EMPTY"
    base_env["TAU_PAIR_RUN_DIR"] = str(dest)
    base_env.setdefault("TAU_TEACHER_API_KEY", "EMPTY")
    owned, temporary = [], []
    statuses = {}
    status_path = dest / "status.json"
    if args.resume and status_path.exists():
        statuses = json.loads(status_path.read_text())

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Refuse busy GPUs, including unrelated servers. Never pkill them.
        ids = sorted(assigned_gpus(plan))
        used = subprocess.check_output(["nvidia-smi", "-i", ",".join(ids), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if any(int(line) > 512 for line in used.splitlines()):
            raise RuntimeError("assigned GPUs are occupied; stop their owning jobs first")
        services = []
        for item in plan["services"]:
            port = int(item["api_base"].split(":")[-1].split("/")[0])
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            env = dict(base_env, CUDA_VISIBLE_DEVICES=",".join(item["gpus"]))
            process = OwnedProcess(item["command"], env, dest / "services" / f"{item['name']}.log")
            owned.append(process)
            services.append(process)
            deadline = time.monotonic() + 1200
            while True:
                process.track()
                if process.proc.poll() is not None:
                    raise RuntimeError(f"{item['name']} service exited; inspect its log")
                try:
                    check_model(item["api_base"], item["model"], "EMPTY")
                    break
                except Exception as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{item['name']} startup timeout") from exc
                    time.sleep(5)
        sample = next(iter(plan["experiments"].values()))
        preflight(sample["TAU_USER_API_BASE"], sample["TAU_MATCHER_MODEL"], "EMPTY", thinking=False)
        if args.pair != "self-s1-s2":
            preflight(sample["TAU_TEACHER_API_BASE"], sample["TAU_TEACHER_MODEL"], base_env[sample["TAU_TEACHER_API_KEY_ENV"]], thinking=True)
        experiments, ray_heads = {}, {}
        for name, settings in plan["experiments"].items():
            if args.resume and statuses.get(name) == 0:
                continue
            temp = tempfile.mkdtemp(prefix=f"tau-{name}-")
            temporary.append(temp)
            env = dict(base_env, **settings, RESUME_MODE="auto" if args.resume else "disable", RESUME_FROM_PATH="")
            head, address = start_ray_head(settings["PYTHON"], env, temp, dest / name / "ray_head.log", num_cpus=args.cpus_per_run, num_gpus=int(settings["N_GPUS"]))
            owned.append(head)
            ray_heads[name] = head
            process = OwnedProcess(training_command(address), env, dest / name / "launcher.log")
            owned.append(process)
            experiments[name] = process
            print(f"Started {name}: GPUs {settings['CUDA_VISIBLE_DEVICES']}, PID {process.proc.pid}, Ray {address}", flush=True)
        while experiments:
            for process in owned:
                if not process.closed:
                    process.track()
            if any(p.proc.poll() is not None for p in services):
                raise RuntimeError("shared service died; stopping owned training jobs")
            for name, process in list(experiments.items()):
                code = process.proc.poll()
                if code is None and ray_heads[name].proc.poll() is not None:
                    code = 1
                    print(f"{name} Ray head exited; stopping only its training job", flush=True)
                if code is not None:
                    statuses[name] = code
                    status_path.write_text(json.dumps(statuses, indent=2) + "\n")
                    print(f"{name} exited {code}; sibling is left running", flush=True)
                    process.stop()
                    ray_heads[name].stop()
                    del experiments[name]
            time.sleep(5)
        if any(code != 0 for code in statuses.values()):
            raise RuntimeError(f"one or more experiments failed: {statuses}")
    except Exception:
        # Preserve the initiating failure, not just shutdown errors in vLLM's
        # service log when finally cleans up the owned processes.
        with (dest / "launcher_error.log").open("a") as error_log:
            error_log.write(traceback.format_exc())
        raise
    finally:
        with uninterrupted_cleanup():
            for process in reversed(owned):
                try:
                    process.stop()
                except Exception as exc:
                    # One failed cleanup must not abandon all other processes.
                    print(f"Cleanup failed for owned PID {process.proc.pid}: {exc}", flush=True)
            for temp in temporary:
                # mkdtemp-owned Ray sockets only; training artifacts live in runs/.
                shutil.rmtree(temp, ignore_errors=True)
            lock.close()


if __name__ == "__main__":
    main()
