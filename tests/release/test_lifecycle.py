"""Resume integrity and owned-process cleanup without models, GPUs or APIs."""

import copy
import os
import select
import signal
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import psutil
import pytest

from aopd import __main__ as launcher
from aopd.launch import build_plan
from aopd.services import stop_owned


def _saved_training(monkeypatch, tmp_path):
    model = tmp_path / "student"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3"}\n')
    (model / "model.safetensors").write_bytes(b"first-model-weights")
    tau = tmp_path / "tau"
    (tau / "src/tau2").mkdir(parents=True)
    (tau / "src/tau2/__init__.py").write_text("VERSION = 1\n")
    (tau / "data").mkdir()
    (tau / "data/tasks.json").write_text('[{"id": "1", "instruction": "original"}]')
    runtime = {
        "model": str(model),
        "sources": {"tau": str(tau)},
        "student": {"gpus": [0], "tp": 1, "sp": 1, "ppo_tokens_per_gpu": 32768, "logprob_tokens_per_gpu": 32768},
        "services": {
            "user": {
                "mode": "api",
                "provider": "openai-compatible",
                "model": "test-user",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEST_ONLY_USER_KEY",
            }
        },
        "roles": {"teacher": {"service": "user"}, "matcher": {"service": "user"}, "user": {"service": "user", "generation": {"temperature": 0.7}}},
    }
    monkeypatch.setenv("TEST_ONLY_USER_KEY", "test-only")
    monkeypatch.setattr(launcher, "require_free_gpus", lambda _: None)
    monkeypatch.setattr(launcher, "local_services", lambda *_: nullcontext())
    monkeypatch.setattr(launcher, "probe_roles", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_, **__: SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr(launcher, "stop_owned", lambda _: None)
    run_dir = tmp_path / "evaluation"
    launcher.run_plan(build_plan("tau-full", runtime, run_dir), runtime, run_dir)
    return runtime, run_dir


def test_training_resume_accepts_unchanged_weights_and_configuration(monkeypatch, tmp_path):
    runtime, run_dir = _saved_training(monkeypatch, tmp_path)
    launcher.run_plan(build_plan("tau-full", runtime, run_dir), runtime, run_dir)
    assert (run_dir / "exit_code").read_text().strip() == "0"


@pytest.mark.parametrize("changed", ["weights", "user_generation", "smoke", "context", "tau_source", "tau_tasks"])
def test_training_resume_rejects_changed_identity_before_overwriting_artifacts(monkeypatch, tmp_path, changed):
    runtime, run_dir = _saved_training(monkeypatch, tmp_path)
    before = {p.relative_to(run_dir): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    resumed = copy.deepcopy(runtime)
    if changed == "weights":
        # Same served name and filesystem path, different evaluated parameters.
        (tmp_path / "student" / "model.safetensors").write_bytes(b"other-model-weights")
    elif changed == "user_generation":
        resumed["roles"]["user"]["generation"]["temperature"] = 0.8
    elif changed == "context":
        resumed["training"] = {"MAX_PROMPT": 8192}
    elif changed == "tau_source":
        (tmp_path / "tau/src/tau2/__init__.py").write_text("VERSION = 2\n")
    elif changed == "tau_tasks":
        (tmp_path / "tau/data/tasks.json").write_text('[{"id": "1", "instruction": "changed"}]')

    def must_not_start(*_, **__):
        raise AssertionError("Changed resume identity reached service startup")

    monkeypatch.setattr(launcher, "local_services", must_not_start)
    with pytest.raises((ValueError, RuntimeError), match="(?i)identity|protocol|resume|changed"):
        launcher.run_plan(build_plan("tau-full", resumed, run_dir, smoke=changed == "smoke"), resumed, run_dir)
    after = {p.relative_to(run_dir): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assert after == before, "Rejected resume changed the existing experiment artifacts"


def _is_running(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_cleanup_stops_owned_detached_descendants_and_preserves_unrelated_jobs():
    # Mirrors workers that start their own session, as inference/Ray workers can.
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(120)"
    owner_code = f"import subprocess,sys,time; child=subprocess.Popen([sys.executable, '-c', {child_code!r}], stdout=subprocess.PIPE, text=True, start_new_session=True); child.stdout.readline(); print(child.pid, flush=True); time.sleep(120)"
    owned = subprocess.Popen([sys.executable, "-c", owner_code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    child = None
    try:
        ready, _, _ = select.select([owned.stdout], [], [], 5)
        assert ready, "Test worker failed to start"
        child = int(owned.stdout.readline())
        assert os.getpgid(child) != owned.pid
        assert _is_running(child)

        stop_owned(owned, timeout=2)

        assert not _is_running(child), "Owned worker escaped cleanup by starting a new session"
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        # Explicitly reap test processes even when the production cleanup fails.
        for pid in (child, owned.pid, unrelated.pid):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        owned.wait(timeout=5)
        unrelated.wait(timeout=5)
        owned.stdout.close()
