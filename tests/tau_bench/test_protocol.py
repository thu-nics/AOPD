import json
import random
from types import SimpleNamespace

import pytest

import agent_system.environments.env_package.tau_bench.envs as tau_envs
from agent_system.environments.env_package.tau_bench.envs import (
    OFFICIAL_TASK_COUNTS,
    TASK_MANIFEST_PROTOCOL_VERSION,
    TAU2_COMMIT,
    TERMINAL_REWARD_PROTOCOL,
    TauBenchVectorEnv,
    TauBenchWorker,
    compatibility_patch_sha256,
    interleave_grouped_domains,
    select_uniform_argmax,
    validate_tau_runtime_config,
    validate_tau_source,
)
from examples.tau_bench.prepare_tau_training import (
    allocate_validation_counts,
    build_validation_rows,
)


def test_grouped_domain_schedule_keeps_outcome_replicas_contiguous():
    labels = interleave_grouped_domains({"airline": 4, "retail": 4}, group_n=4)
    assert len(labels) == 32
    assert all(len(set(labels[index : index + 4])) == 1 for index in range(0, 32, 4))
    assert labels.count("airline") == labels.count("retail") == 16


def test_official_tau_task_counts_are_explicit():
    assert TASK_MANIFEST_PROTOCOL_VERSION == 2
    assert OFFICIAL_TASK_COUNTS == {
        "train": {"airline": 30, "retail": 74},
        "base": {"airline": 50, "retail": 114},
    }


def test_validation_plan_uses_fixed_proportional_complete_batches():
    counts = allocate_validation_counts(
        {"airline": 50, "retail": 114},
        domains=["airline", "retail"],
        batch_size=16,
    )
    assert counts == {"airline": 5, "retail": 11}
    pools = {
        "airline": [{"id": f"a{index}"} for index in range(50)],
        "retail": [{"id": f"r{index}"} for index in range(114)],
    }
    rows, plan = build_validation_rows(
        pools,
        domains=["airline", "retail"],
        trials=1,
        base_seed=300,
        num_tasks=None,
        batch_size=16,
    )
    assert len(rows) == 160
    assert plan["evaluated_rows"] == {"airline": 50, "retail": 110}
    assert plan["dropped_rows"] == {"airline": 0, "retail": 4}
    template = [row["env_kwargs"]["domain"] for row in rows[:16]]
    assert all([row["env_kwargs"]["domain"] for row in rows[offset : offset + 16]] == template for offset in range(0, len(rows), 16))


def test_uniform_argmax_never_selects_lower_reward():
    rng = random.Random(3)
    selected = {select_uniform_argmax([-1.0, 0.0, 1.0, 1.0], rng) for _ in range(100)}
    assert selected == {2, 3}


def test_all_invalid_candidates_still_use_uniform_argmax():
    rng = random.Random(11)
    selected = {select_uniform_argmax([-1.0] * 4, rng) for _ in range(100)}
    assert selected == {0, 1, 2, 3}


def _write_required_tau_user_data(root):
    data_dir = root / "data" / "tau2" / "user_simulator"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "simulation_guidelines.md").write_text("guidelines")
    (data_dir / "simulation_guidelines_tools.md").write_text("tool guidelines")


def test_tau_source_validation_binds_root_and_commit(tmp_path, monkeypatch):
    root = tmp_path / "tau2-bench"
    root.mkdir()
    _write_required_tau_user_data(root)
    monkeypatch.setattr(tau_envs, "tau_source_root", lambda: root)
    monkeypatch.setattr(
        tau_envs.subprocess,
        "check_output",
        lambda *args, **kwargs: TAU2_COMMIT + "\n",
    )

    identity = validate_tau_source(root)

    assert identity == {
        "source_root": str(root),
        "tau2_commit": TAU2_COMMIT,
        "compatibility_patch_sha256": compatibility_patch_sha256(),
    }
    with pytest.raises(RuntimeError, match="source root mismatch"):
        validate_tau_source(tmp_path / "other")


def test_tau_source_validation_rejects_missing_user_simulator_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_envs, "tau_source_root", lambda: tmp_path)
    monkeypatch.setattr(
        tau_envs.subprocess,
        "check_output",
        lambda *args, **kwargs: TAU2_COMMIT + "\n",
    )
    with pytest.raises(RuntimeError, match="user-simulator data"):
        validate_tau_source(tmp_path)


def test_tau_source_validation_rejects_commit_drift(tmp_path, monkeypatch):
    _write_required_tau_user_data(tmp_path)
    monkeypatch.setattr(tau_envs, "tau_source_root", lambda: tmp_path)
    monkeypatch.setattr(
        tau_envs.subprocess,
        "check_output",
        lambda *args, **kwargs: "wrong\n",
    )
    with pytest.raises(RuntimeError, match="must be pinned"):
        validate_tau_source(tmp_path)


def test_tau_source_validation_rejects_missing_compatibility_patch(tmp_path, monkeypatch):
    _write_required_tau_user_data(tmp_path)
    monkeypatch.setattr(tau_envs, "tau_source_root", lambda: tmp_path)

    def fake_check_output(command, **kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return TAU2_COMMIT + "\n"
        raise tau_envs.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tau_envs.subprocess, "check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="compatibility patch"):
        validate_tau_source(tmp_path)


def _runtime_config(**updates):
    values = {
        "user_temperature": 0.0,
        "user_reasoning_enabled": False,
        "oracle": SimpleNamespace(
            model="deepseek/deepseek-v4-flash",
            samples=3,
        ),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_runtime_config_requires_fixed_user_and_k3_oracle():
    validate_tau_runtime_config(_runtime_config(), require_oracle=True)

    with pytest.raises(RuntimeError, match="reasoning"):
        validate_tau_runtime_config(
            _runtime_config(user_reasoning_enabled=True),
            require_oracle=False,
        )
    with pytest.raises(RuntimeError, match="user_temperature"):
        validate_tau_runtime_config(
            _runtime_config(user_temperature=0.1),
            require_oracle=False,
        )
    config = _runtime_config()
    config.oracle.samples = 2
    with pytest.raises(RuntimeError, match="three oracle samples"):
        validate_tau_runtime_config(config, require_oracle=True)


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _FakeWorker:
    def __init__(self, index, domain):
        self.index = index
        self.domain = domain
        self.resets = []
        self.actions = []
        self.reset = _RemoteMethod(self._reset)
        self.step = _RemoteMethod(self._step)

    def _reset(self, **kwargs):
        self.resets.append(kwargs)
        return f"obs-{self.index}", {
            "domain": self.domain,
            "task_id": kwargs["task_id"],
        }

    def _step(self, action):
        self.actions.append(action)
        return f"next-{self.index}", 0.0, False, {"worker": self.index}


def test_builder_owns_vanilla_group_expansion(monkeypatch):
    created = []
    options = []

    def fake_remote(**kwargs):
        created.append(kwargs)
        return object()

    def fake_options(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(remote=fake_remote)

    monkeypatch.setattr(
        tau_envs,
        "TauBenchWorker",
        SimpleNamespace(options=fake_options),
    )
    env_config = SimpleNamespace(
        resources_per_worker={"num_cpus": 0.5, "num_gpus": 0},
        tau=SimpleNamespace(
            train_max_steps=20,
            eval_max_steps=30,
            user_llm="test-user",
            user_temperature=0.0,
            user_reasoning_enabled=False,
        ),
    )

    env = tau_envs.build_tau_bench_envs(
        seed=7,
        counts={"airline": 1, "retail": 1},
        group_n=4,
        env_config=env_config,
        is_train=True,
    )

    assert len(env.workers) == 8
    assert env.seeds == list(range(7, 15))
    assert len(created) == 8
    assert [worker["domain"] for worker in created] == ["airline"] * 4 + ["retail"] * 4
    assert options == [{"num_cpus": 0.5, "num_gpus": 0}]


def test_vector_env_requires_complete_fixed_domain_batches(monkeypatch):
    monkeypatch.setattr(tau_envs.ray, "get", lambda values: values)
    domains = ["airline", "retail"]
    workers = [_FakeWorker(index, domain) for index, domain in enumerate(domains)]
    env = TauBenchVectorEnv(workers, domains, [10, 11])

    observations, infos = env.reset(
        kwargs=[
            {"domain": "airline", "task_id": "a0", "seed": 100},
            {"domain": "retail", "task_id": "r0", "seed": 101},
        ]
    )
    assert observations == ["obs-0", "obs-1"]
    assert [info["domain"] for info in infos] == domains
    assert workers[0].resets[-1]["seed"] == 100
    assert workers[1].resets[-1]["seed"] == 101

    _, rewards, dones, _ = env.step(["message one", "message two"])
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    with pytest.raises(ValueError, match="expected 2 Tau env kwargs"):
        env.reset(kwargs=[{"domain": "airline", "task_id": "a1"}])
    with pytest.raises(ValueError, match="does not match slot"):
        env.reset(
            kwargs=[
                {"domain": "retail", "task_id": "r1"},
                {"domain": "airline", "task_id": "a1"},
            ]
        )


def test_finished_tau_worker_step_is_an_idempotent_zero_reward_noop():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=0.0,
        user_reasoning_enabled=False,
    )
    worker._done = True
    worker._last_observation = "terminal observation"
    worker._last_info = {"protocol_reward": 1.0}

    observation, reward, done, info = worker.step("ignored action")

    assert observation == "terminal observation"
    assert reward == 0.0
    assert done is True
    assert info["terminal_reason"] == "already_done"
    assert info["protocol_reward"] == 0.0
    assert info["tool_calling"] == 0


def test_tau_worker_marks_executed_native_tool_action():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=0.0,
        user_reasoning_enabled=False,
    )
    worker._validate = lambda action: tau_envs.ParsedAction(kind="tool", name="get_user_details", arguments={"user_id": "u1"})
    worker._execute = lambda action: ("tool observation", 0.0, False, {})

    _, _, done, info = worker.step("ignored raw action")

    assert done is False
    assert info["tool_calling"] == 1
    assert info["parsed_action"] == ('{"arguments":{"user_id":"u1"},"kind":"tool","name":"get_user_details"}')


def test_terminal_reward_uses_db_and_communicate_but_not_nl(monkeypatch):
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator import evaluator

    calls = []

    class Result:
        def __init__(self, reward):
            self.reward = reward

        def model_dump(self, mode):
            assert mode == "json"
            return {"reward": self.reward}

    def fake_evaluate_simulation(*, evaluation_type, **kwargs):
        calls.append(evaluation_type.value)
        return Result({"env": 0.5, "communicate": 0.25}[evaluation_type.value])

    monkeypatch.setattr(evaluator, "evaluate_simulation", fake_evaluate_simulation)
    task = SimpleNamespace(
        evaluation_criteria=SimpleNamespace(
            reward_basis=[
                RewardType.DB,
                RewardType.COMMUNICATE,
                RewardType.NL_ASSERTION,
            ]
        )
    )
    fake_env = SimpleNamespace(
        _simulation_run=object(),
        _get_task=lambda: task,
        solo_mode=False,
        domain="airline",
    )

    reward, info = tau_envs._db_communicate_reward(fake_env)

    assert reward == 0.125
    assert calls == ["env", "communicate"]
    assert json.loads(info)["protocol"] == TERMINAL_REWARD_PROTOCOL
