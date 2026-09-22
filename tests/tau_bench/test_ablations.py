import asyncio
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_protocol import _tau_scoring_worker

from agent_system.environments.env_package.tau_bench.ablations import validate_ablation
from agent_system.environments.env_package.tau_bench.matcher_profiles import MESSAGE_CONCISE, TOOL_CONCISE, order_evidence
from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient
from agent_system.environments.env_package.tau_bench.paired_training import ROOT, OwnedProcess, build_plan, preflight, start_ray_head, training_command
from examples.tau_bench.train.prepare_data import build_train_rows


@pytest.mark.parametrize("case", ["legacy", "tagged", "active", "other_gpu", "outside_repo", "untagged", "other_protocol"])
def test_startup_cleanup_only_reaps_stale_owned_processes(case, monkeypatch, tmp_path):
    from agent_system.environments.env_package.tau_bench import paired_training

    root = tmp_path / "repo"
    monkeypatch.setattr(paired_training, "ROOT", root)
    dest = (tmp_path / "elsewhere" if case == "outside_repo" else root / "runs") / "old-run"
    dest.mkdir(parents=True)
    plan = {"protocol": "unrelated" if case == "other_protocol" else paired_training.PROTOCOL, "pair": "full-a1", "services": [{"gpus": ["0", "1"]}], "experiments": {"full": {"CUDA_VISIBLE_DEVICES": "2,3", "RUN_DIR": str(dest / "full")}}}
    (dest / "protocol.json").write_text(json.dumps(plan))
    env = dict(os.environ, TAU_PAIR_OWNER_TOKEN=uuid.uuid4().hex, RUN_DIR=str(dest / "full"))
    env.pop("TAU_PAIR_RUN_DIR", None)
    if case == "tagged":
        env["TAU_PAIR_RUN_DIR"] = str(dest)
    if case == "untagged":
        env.pop("TAU_PAIR_OWNER_TOKEN")
    with (dest / ".launcher.lock").open("a") as lock:
        if case == "active":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Detached, TERM-resistant leftover simulates an orphaned Ray/vLLM
        # worker. Readiness ensures this also tests escalation to SIGKILL.
        child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"], env=env, start_new_session=True, stdout=subprocess.PIPE, text=True)
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            assert child.stdout.readline().strip() == "ready"
            paired_training.cleanup_stale_runs({"8", "9"} if case == "other_gpu" else {"0", "1", "2", "3"}, timeout=0.05)
            if case in ("legacy", "tagged"):
                child.wait(timeout=5)
            else:
                assert child.poll() is None
            assert unrelated.poll() is None
            assert (dest / "protocol.json").is_file()
        finally:
            for process in (child, unrelated):
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            child.stdout.close()


def test_shutdown_cleanup_ignores_repeated_interrupts_then_restores_handlers():
    from agent_system.environments.env_package.tau_bench.paired_training import uninterrupted_cleanup

    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with uninterrupted_cleanup():
        for sig in before:
            os.kill(os.getpid(), sig)
    assert {sig: signal.getsignal(sig) for sig in before} == before


@pytest.mark.parametrize("flag", ["--cleanup-stale", "--cleanup-machine"])
def test_cleanup_dry_run_never_signals_processes(flag, monkeypatch, tmp_path, capsys):
    from agent_system.environments.env_package.tau_bench import paired_training

    monkeypatch.setattr(paired_training, "build_plan", lambda *args: {"dry_run": True})
    monkeypatch.setattr(paired_training, "cleanup_stale_runs", lambda *args: pytest.fail("dry-run must not clean processes"))
    monkeypatch.setattr(paired_training, "cleanup_machine_compute", lambda: pytest.fail("dry-run must not clean processes"))
    dest = tmp_path / "dry-run"
    paired_training.main(["--pair", "full-a1", "--run-dir", str(dest), flag, "--dry-run"])
    assert json.loads(capsys.readouterr().out) == {"dry_run": True}
    assert not dest.exists()


@pytest.mark.parametrize(
    "name,command,expected",
    [
        ("VLLM::EngineCore", [], True),
        ("ray::Worker", [], True),
        ("gcs_server", [], True),
        ("python", ["python", "-m", "vllm.entrypoints.openai.api_server"], True),
        ("python", ["python", "/venv/lib/ray/dashboard/agent.py"], True),
        ("python", ["python", "-m", "agent_system.environments.env_package.tau_bench.paired_training"], True),
        ("python", ["python", "-m", "verl.trainer.main_ppo"], True),
        ("bash", ["bash", "/repo/examples/tau_bench/train/run.sh"], True),
        ("sshd", ["sshd"], False),
        ("tmux", ["tmux", "new-session"], False),
        ("python", ["python", "-c", "print('vllm.entrypoints.openai.api_server')"], False),
    ],
)
def test_machine_cleanup_recognizes_compute_services(name, command, expected):
    from agent_system.environments.env_package.tau_bench.paired_training import is_compute_service

    assert is_compute_service(name, command) is expected


def test_machine_cleanup_reaps_untagged_gpu_and_cpu_service_preserving_launcher(monkeypatch):
    import psutil

    from agent_system.environments.env_package.tau_bench import paired_training

    # No real GPU/other job is touched: enumerate only processes created by
    # this test, plus our launcher ancestry to verify the exclusion contract.
    gpu = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"], start_new_session=True, stdout=subprocess.PIPE, text=True)
    service = subprocess.Popen([sys.executable, "-c", "import os; os.execv('/bin/sleep', ['vllm', '60'])"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        assert gpu.stdout.readline().strip() == "ready"
        deadline = time.monotonic() + 5
        while psutil.Process(service.pid).cmdline()[:1] != ["vllm"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        current = psutil.Process()
        parents = current.parents()
        all_processes = [current, *parents, *(psutil.Process(p.pid) for p in (gpu, service, unrelated))]
        monkeypatch.setattr(paired_training.psutil, "process_iter", lambda: iter(all_processes))

        def nvidia_smi(command, **kwargs):
            assert command[0] == "nvidia-smi"
            if "--query-compute-apps=pid" in command:
                return "\n".join(str(pid) for pid in (gpu.pid, current.pid, *(p.pid for p in parents)))
            assert "--query-gpu=memory.used" in command
            return "0\n0\n"

        monkeypatch.setattr(paired_training.subprocess, "check_output", nvidia_smi)
        paired_training.cleanup_machine_compute(timeout=0.05)
        gpu.wait(timeout=5)
        service.wait(timeout=5)
        assert unrelated.poll() is None
    finally:
        for process in (gpu, service, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        gpu.stdout.close()


def test_three_domain_schedule_and_cycle_are_fixed():
    pools = {d: [{"id": str(i)} for i in range(n)] for d, n in zip(("airline", "retail", "telecom"), (30, 74, 74), strict=True)}
    rows = build_train_rows(pools, counts=dict.fromkeys(pools, 8), num_batches=10)
    assert len(rows) == 240
    for offset in range(0, 240, 24):
        assert [r["env_kwargs"]["domain"] for r in rows[offset : offset + 24]] == list(pools) * 8
    for domain, size in (("airline", 30), ("retail", 74), ("telecom", 74)):
        ids = [r["env_kwargs"]["task_id"] for r in rows if r["env_kwargs"]["domain"] == domain]
        assert ids == [str(i % size) for i in range(80)]


def test_three_domain_metrics_include_telecom_and_keep_separate_outcomes():
    from agent_system.environments.env_package.tau_bench.manager import TauBenchEnvironmentManager

    manager = object.__new__(TauBenchEnvironmentManager)
    manager.oracle_actor = None
    episodes = [
        [dict(tau_domain=domain, terminal_success=success, protocol_reward=float(success), action_kind="tool", is_action_valid=valid, move_optimal=hit, oracle_set_size=2, matcher_required_group=False)]
        for domain, success, valid, hit in [("airline", True, True, True), ("retail", False, True, False), ("telecom", False, False, False), ("telecom", True, True, True)]
    ]
    metrics = manager.success_evaluator(total_infos=episodes)
    for domain, count in (("airline", 1), ("retail", 1), ("telecom", 2)):
        assert metrics[f"env/{domain}/trajectory_count"].tolist() == [count]
        assert metrics[f"env/{domain}/trajectory_share"].tolist() == [count / 4]
    for name in ("success_rate", "valid_action_rate", "oracle_hit_rate", "protocol_reward"):
        assert metrics[f"env/telecom/{name}"].tolist() == [0, 1]
    assert metrics["env/airline/success_rate"].tolist() == [1]
    assert metrics["env/retail/success_rate"].tolist() == [0]


def test_ray_connection_overrides_compose_without_resource_counts():
    from hydra import compose, initialize_config_dir

    command = training_command("127.0.0.1:34567")
    # run.sh defines +ray_init.num_cpus before forwarding these overrides.
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="tau_agentic_opd", overrides=["+ray_init.num_cpus=48", *command[2:]])
    assert config.ray_init.address == "127.0.0.1:34567"
    assert config.ray_init.num_cpus is None
    assert "num_gpus" not in config.ray_init
    assert config.actor_rollout_ref.actor.loss_normalization == "global-minibatch-v1"
    assert list(config.trainer.logger) == ["console", "tensorboard"]


@pytest.mark.parametrize(
    "thinking,message,accepted",
    [
        (True, {"content": None, "reasoning": "Working through the reply."}, True),
        (True, {"content": "", "reasoning_content": "Working through the reply."}, True),
        (True, {"content": "OK", "reasoning": None}, True),
        (False, {"content": "OK"}, True),
        (False, {"content": None, "reasoning": "Unexpected reasoning only."}, False),
        (True, {"content": None, "reasoning": None, "reasoning_content": ""}, False),
        (True, {"content": " ", "reasoning": "\n"}, False),
    ],
)
def test_preflight_accepts_both_reasoning_fields_but_rejects_empty_output(monkeypatch, thinking, message, accepted):
    from agent_system.environments.env_package.tau_bench import paired_training

    def request(base, route, key, payload=None):
        if route == "/models":
            return {"data": [{"id": "teacher"}]}
        assert route == "/chat/completions"
        assert payload["chat_template_kwargs"]["enable_thinking"] is thinking
        return {"choices": [{"message": message, "finish_reason": "length"}]}

    monkeypatch.setattr(paired_training, "api_request", request)
    if accepted:
        preflight("http://teacher/v1", "teacher", "test", thinking=thinking)
    else:
        with pytest.raises(RuntimeError, match="empty preflight response.*finish_reason=length"):
            preflight("http://teacher/v1", "teacher", "test", thinking=thinking)


def test_ray_head_requests_os_assigned_worker_ports(monkeypatch, tmp_path):
    from agent_system.environments.env_package.tau_bench import paired_training

    commands = []

    def process(command, env, log):
        commands.append(command)
        metric_port = int(env["DASHBOARD_METRIC_PORT"])
        assert metric_port > 0
        assert f"--dashboard-agent-listen-port={metric_port}" not in command
        (tmp_path / "ray_current_cluster").write_text("127.0.0.1:45678")
        return SimpleNamespace(track=lambda: None, proc=SimpleNamespace(poll=lambda: None))

    monkeypatch.setattr(paired_training, "OwnedProcess", process)
    _, address = start_ray_head(sys.executable, {}, str(tmp_path), tmp_path / "head.log", num_cpus=1, num_gpus=0)
    assert address == "127.0.0.1:45678"
    assert "--min-worker-port=0" in commands[0]
    assert "--max-worker-port=0" in commands[0]
    assert "--no-monitor" in commands[0]


@pytest.mark.skipif(os.environ.get("TAU_RUN_RAY_PAIR_TEST") != "1", reason="opt-in real dual-Ray-head integration test")
def test_real_ray_heads_have_distinct_live_agent_and_worker_ports(tmp_path):
    import socket

    import psutil

    heads, drivers, temporary = [], [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RAY_USAGE_STATS_ENABLED="0")
    env.pop("RAY_ADDRESS", None)
    code = """
import json, os, psutil, ray, sys, time
from pathlib import Path
ray.init(address=sys.argv[1], num_cpus=None)
@ray.remote(num_cpus=0.25)
class Probe:
    def inspect(self):
        ports = sorted({c.laddr.port for c in psutil.Process().net_connections(kind="tcp") if c.status == psutil.CONN_LISTEN})
        return {"answer": 42, "pid": os.getpid(), "ports": ports}
    def log_metric(self):
        from verl.utils.tracking import _TensorboardAdapter
        logger = _TensorboardAdapter()
        logger.log({"probe/value": 42}, step=1)
        logger.finish()
        return os.environ["TENSORBOARD_DIR"]
actors = [Probe.remote() for _ in range(4)]
result = {"workers": ray.get([a.inspect.remote() for a in actors]), "resources": ray.cluster_resources()}
result["tensorboard_dir"] = ray.get(actors[0].log_metric.remote())
Path(sys.argv[2]).write_text(json.dumps(result))
if len(sys.argv) > 3:
    deadline = time.monotonic() + 180
    while not Path(sys.argv[3]).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("parent did not release port probes")
        time.sleep(0.1)
ray.shutdown()
"""
    try:
        addresses = []
        for name in ("one", "two"):
            dest = tmp_path / name
            dest.mkdir()
            # Ray's Unix socket path limit is shorter than some pytest tmp paths.
            temp = tempfile.TemporaryDirectory(prefix="tau-ray-test-")
            temporary.append(temp)
            head_env = dict(env, TENSORBOARD_DIR=str(dest / "tensorboard"))
            head, address = start_ray_head(sys.executable, head_env, temp.name, dest / "head.log", num_cpus=1, num_gpus=0, object_store_memory=100 * 1024 * 1024)
            heads.append(head)
            addresses.append(address)
            # A later shell export must not redirect the already-running Ray
            # head's workers, or mix the two experiments' event files.
            driver_env = dict(env, TENSORBOARD_DIR=str(tmp_path / "wrong-driver-dir"))
            driver = OwnedProcess([sys.executable, "-c", code, address, str(dest / "result.json"), str(dest / "release")], driver_env, dest / "driver.log")
            drivers.append(driver)
        deadline = time.monotonic() + 90
        while not all((tmp_path / name / "result.json").exists() for name in ("one", "two")):
            assert all(driver.proc.poll() is None for driver in drivers)
            if time.monotonic() > deadline:
                pytest.fail("Ray worker probes did not finish")
            time.sleep(0.1)
        assert len(set(addresses)) == 2
        agent_ports, metric_ports, worker_ports, worker_pids = [], [], [], []
        for name, head, temp, address in zip(("one", "two"), heads, temporary, addresses, strict=True):
            result = json.loads((tmp_path / name / "result.json").read_text())
            assert result["resources"]["CPU"] == 1
            expected_tb = tmp_path / name / "tensorboard"
            assert result["tensorboard_dir"] == str(expected_tb)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            events = EventAccumulator(str(expected_tb)).Reload().Scalars("probe/value")
            assert [(event.step, event.value) for event in events] == [(1, 42)]
            assert not (tmp_path / "wrong-driver-dir").exists()
            for worker in result["workers"]:
                assert worker["answer"] == 42 and worker["ports"]
                worker_pids.append(worker["pid"])
                worker_ports.extend(worker["ports"])
                for port in worker["ports"]:
                    with socket.create_connection((address.split(":")[0], port), timeout=1):
                        pass
            assert head.proc.poll() is None
            session = Path(temp.name) / "session_latest"
            allocation = next(iter(json.loads((session / "ports_by_node.json").read_text()).values()))
            agent_port = allocation["dashboard_agent_listen_port"]
            agent_ports.append(agent_port)
            metric_port = int(psutil.Process(head.proc.pid).environ()["DASHBOARD_METRIC_PORT"])
            metric_ports.append(metric_port)
            node_ip = json.loads((session / "node_ip_address.json").read_text())["node_ip_address"]
            deadline = time.monotonic() + 30
            while True:
                try:
                    with socket.create_connection((node_ip, agent_port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            with socket.create_connection((node_ip, metric_port), timeout=1):
                pass
            assert "address already in use" not in (session / "logs/dashboard_agent.log").read_text().lower()
            for log in (session / "logs").iterdir():
                if log.is_file() and log.suffix in {".log", ".out", ".err"}:
                    assert "address already in use" not in log.read_text().lower(), log
        assert len(set(agent_ports)) == 2
        assert len(set(agent_ports + metric_ports)) == 4
        assert len(set(worker_pids)) == 8
        assert len(set(worker_ports)) == len(worker_ports)
        for name in ("one", "two"):
            (tmp_path / name / "release").touch()
        for driver in drivers:
            assert driver.proc.wait(timeout=30) == 0
        # Cleaning one experiment must not disconnect or stop its sibling.
        heads[0].stop()
        probe = OwnedProcess([sys.executable, "-c", code, addresses[1], str(tmp_path / "after_cleanup.json")], env, tmp_path / "after_cleanup.log")
        drivers.append(probe)
        assert probe.proc.wait(timeout=90) == 0
        assert all(w["answer"] == 42 for w in json.loads((tmp_path / "after_cleanup.json").read_text())["workers"])
        assert heads[1].proc.poll() is None
    finally:
        for process in reversed([*heads, *drivers]):
            process.stop()
        for temp in temporary:
            temp.cleanup()


def test_a1_learning_reward_does_not_change_commit_or_oracle_hits():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.ablation = "a1"
    rows, selected, _, reward, _, _ = asyncio.run(worker.step_candidate_group(["A", "B", "C", ""]))
    assert [r[1] for r in rows] == [1, 1, 1, -1]
    assert selected == 0 and reward == 1
    assert [r[3]["move_optimal"] for r in rows] == [True, True, False, False]
    assert [r[3]["selection_score"] for r in rows] == [1.25, 1, 0, -1]
    assert sum(r[3]["state_group_advanced"] for r in rows) == 1


def test_a5_can_commit_invalid_without_resampling():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.ablation = "a5"
    worker._rng = SimpleNamespace(randrange=lambda size: size - 1)
    rows, selected, _, reward, _, info = asyncio.run(worker.step_candidate_group(["A", "B", "C", ""]))
    assert selected == 3 and reward == -1
    assert [r[1] for r in rows] == [1.25, 1, 0, -1]
    assert info["terminal_reason"] == "invalid_noop"
    assert info["state_group_selection_type"] == "random"
    assert info["state_group_random_select_prob"] == 1
    assert worker._step == 1
    assert not any(r[3]["appearance_counterfactual_selected"] for r in rows)


@pytest.mark.parametrize("variant", ["full", "a1", "a4", "a5"])
def test_every_ablation_keeps_minus_one_transfer_guard(variant):
    from agent_system.environments.env_package.tau_bench.actions import TRANSFER_HANDOFF_MESSAGE

    worker = _tau_scoring_worker("appearance" if variant == "a4" else "frequency_weighted")
    worker.ablation = variant
    worker.transfer_reward_guard_enabled = True
    worker._transfer_succeeded = False
    rows = asyncio.run(worker.step_candidate_group([TRANSFER_HANDOFF_MESSAGE, "B", "C", ""]))[0]
    assert rows[0][1] == -1 and rows[0][3]["transfer_without_tool"]
    assert rows[0][3]["move_optimal"] is False


def test_ablation_rejects_conflicting_switches():
    with pytest.raises(ValueError):
        validate_ablation("a1", reward_mode="frequency_weighted", programmatic_only=True)
    with pytest.raises(ValueError):
        validate_ablation("a4", reward_mode="frequency_weighted")
    with pytest.raises(ValueError):
        validate_ablation("a5", reward_mode="appearance")


def test_internal_budget_is_distinct_from_agent_decision_limit():
    worker = _tau_scoring_worker("frequency_weighted")
    worker._last_step_hit_decision_limit = False
    worker._env = SimpleNamespace(_simulation_run=SimpleNamespace(termination_reason="max_steps"))
    assert worker._terminal_reason(True) == "internal_step_limit"
    assert worker._terminal_reason(False) is None
    worker._last_step_hit_decision_limit = True
    assert worker._terminal_reason(True) == "decision_limit"


def test_disabled_matcher_ignores_stale_profile(monkeypatch):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    client = TauTeacherClient(matcher_enabled=False, matcher_profile="qwen38_concise", matcher_enable_thinking=True, matcher_max_tokens=128)
    assert client.matcher_profile == "default"
    assert client.matcher_enabled is False


def test_qwen38_profile_payload_and_cache_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    path = str(tmp_path / "matcher.jsonl")
    captured = []
    client = TauTeacherClient(matcher_profile="qwen38_concise", matcher_cache_path=path)

    def post(payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": '{"equivalent":true}'}}]}

    monkeypatch.setattr(client, "_post_matcher", post)
    result = client.match_message_pairs(["Teacher text"], ["Candidate text"], [{"role": "user", "content": "context"}], [])
    assert result["counts"] == [1]
    payload = captured[0]
    assert payload["temperature"] == 0.7 and payload["top_p"] == 0.8
    assert payload["max_tokens"] == 32768 and payload["top_k"] == 20
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["messages"][0]["content"] == MESSAGE_CONCISE
    assert list(json.loads(payload["messages"][1]["content"]))[:2] == ["public_context", "tools"]
    assert client.tool_instruction == TOOL_CONCISE
    assert client.tool_matcher_decoding == client.matcher_decoding
    reloaded = TauTeacherClient(matcher_profile="qwen38_concise", matcher_cache_path=path)
    assert reloaded.stats()["matcher_cache_records_loaded"] == 1
    other = TauTeacherClient(matcher_cache_path=path)
    assert other.stats()["matcher_cache_records_loaded"] == 0
    assert list(order_evidence({"candidate_arguments": {}, "public_context": [], "tool": {}, "teacher_arguments": {}}, tool=True)) == ["public_context", "tool", "teacher_arguments", "candidate_arguments"]


def test_teacher_import_across_endpoint_preserves_model_and_decoding_checks(monkeypatch, tmp_path):
    from agent_system.environments.env_package.tau_bench.actions import ParsedAction

    monkeypatch.setenv("TAU_TEACHER_API_KEY", "test")
    source = tmp_path / "teacher.jsonl"
    original = TauTeacherClient(api_base="http://remote/v1", matcher_enabled=False, cache_path=str(source))
    monkeypatch.setattr(original, "_sample_once", lambda **kw: ParsedAction(kind="message", content="Hello"))
    kwargs = dict(state_fingerprint="s", messages=[{"role": "user", "content": "hi"}], tools=[])
    original.sample_multiset(**kwargs)
    content = source.read_bytes()
    local = TauTeacherClient(api_base="http://localhost/v1", matcher_enabled=False, cache_path=str(tmp_path / "new.jsonl"), teacher_cache_import_paths=[str(source)])
    monkeypatch.setattr(local, "_sample_once", lambda **kw: pytest.fail("must reuse"))
    assert len(local.sample_multiset(**kwargs)) == 3
    assert source.read_bytes() == content
    for change in ({"model": "different-model"}, {"temperature": 0.7}):
        other = TauTeacherClient(api_base="http://localhost/v1", matcher_enabled=False, teacher_cache_import_paths=[str(source)], **change)
        assert other._import_samples("s", kwargs["messages"], [], "student_visible") == []


@pytest.mark.parametrize("pair, sizes, local_teacher", [("full-a1", [4, 2], False), ("full-a1", [2, 2], True), ("a4-a5", [2, 2], False)])
@pytest.mark.parametrize("smoke", [False, True])
def test_pair_plan_has_disjoint_gpus_and_matched_training_protocol(pair, sizes, local_teacher, smoke, tmp_path):
    env = {"MODEL_PATH": "/models/student", "TAU_SHARED_MODEL_PATH": "/models/q38", "TAU_LOCAL_TEACHER_PATH": "/models/q32", "TAU2_ROOT": "/repos/tau", "TAU_TEACHER_API_BASE": "http://teacher/v1"}
    if local_teacher or pair == "a4-a5":
        # Local modes must not require, or accidentally call, a remote API.
        del env["TAU_TEACHER_API_BASE"]
    args = SimpleNamespace(pair=pair, run_dir=str(tmp_path), steps=100, shared_port=8180, teacher_port=8181, cpus_per_run=48, smoke=smoke, local_teacher=local_teacher)
    plan = build_plan(args, env)
    if local_teacher or pair == "a4-a5":
        teacher = plan["services"][1]
        assert teacher["name"] == "teacher" and teacher["gpus"] == ["2", "3"]
        assert "/models/q32" in teacher["command"]
        assert teacher["command"][teacher["command"].index("--tensor-parallel-size") + 1] == "2"
        capacity = int(teacher["command"][teacher["command"].index("--max-num-seqs") + 1])
        assert sum(int(s["TAU_TEACHER_MAX_CONCURRENT_REQUESTS"]) for s in plan["experiments"].values()) == capacity == 64
        for settings in plan["experiments"].values():
            assert settings["TAU_TEACHER_API_BASE"] == teacher["api_base"] == "http://127.0.0.1:8181/v1"
            assert settings["TAU_TEACHER_API_KEY_ENV"] == "TAU_PAIR_LOCAL_API_KEY"
    else:
        assert len(plan["services"]) == 1
    launcher = (ROOT / "examples/tau_bench/train/run.sh").read_text()
    # Exercise the real shell smoke overrides before checking trainer config;
    # checking only AIRLINE_TRAJ etc. would miss the smoke-only batch of 3 bug.
    sizing = launcher[launcher.index('if [[ "$SMOKE" == "1" ]]; then') : launcher.index('mkdir -p "$RUN_DIR/ckpt"')]
    sizing += '\nprintf "%s %s %s %s\\n" "$TRAIN_BATCH" "$PPO_MINI_BATCH" "$MAX_PROMPT" "$MAX_RESPONSE"\n'
    gpu_sets = [set(s["gpus"]) for s in plan["services"]]
    for settings, size in zip(plan["experiments"].values(), sizes, strict=True):
        assert int(settings["N_GPUS"]) == size
        assert settings["PPO_MINI_BATCH"] == "32"
        assert settings["GPU_MEM_UTIL"] == "0.65"
        assert settings["MAX_NUM_BATCHED_TOKENS"] == "49152"
        assert int(settings["MAX_NUM_BATCHED_TOKENS"]) >= int(settings["MAX_MODEL_LEN"])
        assert settings["TENSORBOARD_DIR"] == str(Path(settings["RUN_DIR"]) / "tensorboard")
        assert settings["TAU_USER_REASONING_ENABLED"] == "false"
        assert settings["TAU_MATCHER_PROFILE"] == "qwen38_concise"
        assert settings["TAU_TEACHER_MAX_CONCURRENT_REQUESTS"] == "32"
        assert [settings[f"{d}_TRAJ"] for d in ("AIRLINE", "RETAIL", "TELECOM")] == ["8"] * 3
        assert settings["TEST_FREQ"] == "-1" and settings["VAL_BEFORE_TRAIN"] == "false"
        gpu_sets.append(set(settings["CUDA_VISIBLE_DEVICES"].split(",")))
        result = subprocess.run(["bash", "-eu", "-c", sizing], env=dict(os.environ, **settings), capture_output=True, text=True, check=True)
        batch, mini_batch, prompt, response = map(int, result.stdout.split())
        assert batch == (12 if smoke else 24)
        assert batch % size == 0
        assert mini_batch == 32
        assert (prompt, response) == ((8192, 1024) if smoke else (24576, 4096))

        from hydra import compose, initialize_config_dir

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
            config = compose(
                config_name="tau_agentic_opd",
                overrides=[
                    f"data.train_batch_size={batch}",
                    f"trainer.n_gpus_per_node={size}",
                    "trainer.nnodes=1",
                    "actor_rollout_ref.rollout.n=1",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch}",
                    "actor_rollout_ref.actor.use_dynamic_bsz=true",
                    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=2",
                    "actor_rollout_ref.model.use_remove_padding=true",
                ],
            )
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = config
        trainer.use_critic = trainer.use_reference_policy = False
        trainer._validate_config()

        if smoke:
            invalid = dict(os.environ, **settings)
            invalid.update({f"SMOKE_{domain}_TRAJ": "1" for domain in ("AIRLINE", "RETAIL", "TELECOM")})
            rejected = subprocess.run(["bash", "-eu", "-c", sizing], env=invalid, capture_output=True, text=True)
            assert rejected.returncode != 0
            assert f"TRAIN_BATCH=3 must be positive and divisible by N_GPUS={size}" in rejected.stderr
    assert sum(map(len, gpu_sets)) == len(set.union(*gpu_sets)) == 8


def test_pair_memory_settings_are_overridable_and_tensorboard_is_private(tmp_path):
    env = {"MODEL_PATH": "/models/student", "TAU_SHARED_MODEL_PATH": "/models/q38", "TAU_LOCAL_TEACHER_PATH": "/models/q32", "TAU2_ROOT": "/repos/tau", "GPU_MEM_UTIL": "0.60", "MAX_NUM_BATCHED_TOKENS": "32768", "TENSORBOARD_DIR": "/wrong/shared/tensorboard"}
    args = SimpleNamespace(pair="a4-a5", run_dir=str(tmp_path), steps=100, shared_port=8180, teacher_port=8181, cpus_per_run=48, smoke=False)
    plan = build_plan(args, env)
    paths = set()
    for settings in plan["experiments"].values():
        assert settings["GPU_MEM_UTIL"] == "0.60"
        assert settings["MAX_NUM_BATCHED_TOKENS"] == "32768"
        assert settings["TENSORBOARD_DIR"] == str(Path(settings["RUN_DIR"]) / "tensorboard")
        paths.add(settings["TENSORBOARD_DIR"])
    assert len(paths) == 2


@pytest.mark.parametrize("limit", ["16", "24", "32", "0", "-1"])
def test_pair_teacher_concurrency_override(limit, tmp_path):
    env = {"MODEL_PATH": "/models/student", "TAU_SHARED_MODEL_PATH": "/models/q38", "TAU_LOCAL_TEACHER_PATH": "/models/q32", "TAU2_ROOT": "/repos/tau", "TAU_TEACHER_MAX_CONCURRENT_REQUESTS": limit}
    args = SimpleNamespace(pair="a4-a5", run_dir=str(tmp_path), steps=100, shared_port=8180, teacher_port=8181, cpus_per_run=48, smoke=False)
    if int(limit) < 1:
        with pytest.raises(ValueError, match="TAU_TEACHER_MAX_CONCURRENT_REQUESTS must be positive"):
            build_plan(args, env)
    else:
        plan = build_plan(args, env)
        assert all(s["TAU_TEACHER_MAX_CONCURRENT_REQUESTS"] == limit for s in plan["experiments"].values())


def test_owned_process_cleanup_preserves_unrelated_process(tmp_path):
    import psutil

    pid_file = tmp_path / "child.pid"
    code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    owned = OwnedProcess([sys.executable, "-c", code, str(pid_file)], dict(os.environ), tmp_path / "owner.log")
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        child = psutil.Process(int(pid_file.read_text()))
        # Do not call track: exercise discovery after a child detached.
        owned.stop()
        assert unrelated.poll() is None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        owned.stop()  # Idempotent finally cleanup.
    finally:
        owned.stop()
        unrelated.terminate()
        unrelated.wait()
