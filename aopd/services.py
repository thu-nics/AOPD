"""Launch local services and stop only process groups owned by this launcher."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import psutil


def local_command(name, service, run_dir):
    extra = service.get("extra_args", [])
    if not isinstance(extra, list) or any(not isinstance(v, str) for v in extra):
        raise ValueError(f"{name}.extra_args must be an argv list")
    flags = {"--enable-prefix-caching", "--enable-chunked-prefill", "--enforce-eager", "--language-model-only", "--disable-log-requests"}
    valued = {"--max-num-seqs", "--max-num-batched-tokens", "--dtype"}
    index = 0
    while index < len(extra):
        option, _, inline = extra[index].partition("=")
        if option in valued:
            if not inline:
                index += 1
                if index >= len(extra) or extra[index].startswith("-"):
                    raise ValueError(f"{name}.extra_args: missing value for {option}")
        elif option not in flags or inline:
            raise ValueError(f"{name}.extra_args: unsupported or reserved flag {option}")
        index += 1
    command = [
        service.get("python", sys.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        service["model_path"],
        "--served-model-name",
        service["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(service["port"]),
        "--tensor-parallel-size",
        str(service.get("tp", 1)),
        "--data-parallel-size",
        str(service.get("dp", 1)),
        "--max-model-len",
        str(service.get("max_model_len", 65536)),
        "--gpu-memory-utilization",
        str(service.get("memory_utilization", 0.8)),
        "--generation-config",
        "vllm",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        service.get("tool_parser", "hermes"),
        "--reasoning-parser",
        service.get("reasoning_parser", "qwen3"),
        *extra,
    ]
    return command, {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, service["gpus"])), "VLLM_LOGGING_LEVEL": "WARNING"}


def stop_owned(process, timeout=20):
    """Never enumerate/kill arbitrary vLLM or Ray processes on the machine."""
    descendants = []
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    # Workers that start new sessions or outlive their parent retain this unique
    # launch marker. No process-name or blanket GPU matching is used.
    token = getattr(process, "aopd_owner_token", None)
    if token:
        for candidate in psutil.process_iter():
            try:
                if candidate.environ().get("AOPD_OWNER_TOKEN") == token and candidate.pid != process.pid:
                    descendants.append(candidate)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    descendants = list(set(descendants))
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout)
    except ProcessLookupError:
        process.wait(timeout=timeout)
    finally:
        _, alive = psutil.wait_procs(descendants, timeout=timeout)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=timeout)


def owned_process(command, *, env, **kwargs):
    token = uuid.uuid4().hex
    process = subprocess.Popen(command, env={**env, "AOPD_OWNER_TOKEN": token}, start_new_session=True, **kwargs)
    process.aopd_owner_token = token
    return process


def require_free_gpus(runtime):
    devices = {str(d) for d in runtime["student"]["gpus"]}
    for service in runtime.get("services", {}).values():
        if service["mode"] == "local":
            devices.update(map(str, service["gpus"]))
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    ids = dict(line.strip().split(", ") for line in rows.splitlines())
    missing = devices - ids.keys() - set(ids.values())
    if missing:
        raise RuntimeError(f"unknown GPU IDs: {sorted(missing)}")
    selected = {ids.get(d, d) for d in devices}
    allocations = [runtime["student"]["gpus"]] + [s["gpus"] for s in runtime.get("services", {}).values() if s["mode"] == "local"]
    allocated = [ids.get(str(d), str(d)) for group in allocations for d in group]
    if len(allocated) != len(set(allocated)):
        raise RuntimeError("GPU overlap: physical index and UUID may name the same device")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = [line for line in processes.splitlines() if line.split(",")[0].strip() in selected]
    if busy:
        raise RuntimeError(f"assigned GPUs are occupied; stop their owning jobs explicitly: {busy}")


def require_free_port(port):
    # Match the server's bind semantics: TIME_WAIT from our previous run is
    # reusable, but an active listener must still fail (no SO_REUSEPORT).
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


@contextmanager
def local_services(runtime, run_dir):
    processes, logs = [], []
    try:
        for name, service in runtime.get("services", {}).items():
            if service["mode"] != "local":
                continue
            require_free_port(service["port"])
            path = Path(run_dir) / "services" / name
            path.mkdir(parents=True, exist_ok=True)
            log = (path / "server.log").open("ab")
            logs.append(log)
            command, env = local_command(name, service, run_dir)
            process = owned_process(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            deadline = time.monotonic() + service.get("startup_timeout", 900)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited; see {path / 'server.log'}")
                try:
                    with urlopen(f"http://127.0.0.1:{service['port']}/v1/models", timeout=5) as response:
                        models = json.load(response)["data"]
                    if not any(item["id"] == service["model"] for item in models):
                        raise RuntimeError(f"{name} served the wrong model")
                    break
                except (OSError, ValueError, KeyError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{name} readiness timed out; see {path / 'server.log'}") from None
                    time.sleep(2)
        yield
    finally:
        for process in reversed(processes):
            stop_owned(process)
        for log in logs:
            log.close()


def probe_roles(roles, environment):
    """Bounded real Chat API preflight, no dependency on remote GET /models."""
    from aopd.providers import adapt_chat_payload

    checked = set()
    for role in roles.values():
        identity = (role["base_url"], role["model"], role["provider"])
        if identity in checked:
            continue
        checked.add(identity)
        key = environment.get(role["api_key_env"])
        if not key:
            raise ValueError(f"missing API key environment variable {role['api_key_env']}")
        payload = adapt_chat_payload({"model": role["model"], "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": role["provider"] == "zai"}}, role["provider"])
        request = Request(role["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
        if not result.get("choices"):
            raise RuntimeError(f"preflight returned no choices for {role['model']}")
