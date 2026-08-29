import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.environment import Environment
from tau2.user.user_simulator import UserSimulator as TauUserSimulator

import agent_system.environments.env_package.tau_bench.envs as tau_envs
from agent_system.environments.env_package.tau_bench.actions import (
    ParsedAction,
    state_fingerprint,
)
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
    tau_user_simulator_llm_args,
    validate_tau_runtime_config,
    validate_tau_source,
)
from agent_system.environments.env_package.tau_bench.manager import (
    TRANSFER_HANDOFF_MESSAGE,
    TauBenchEnvironmentManager,
)
from examples.tau_bench.native_tau_eval import (
    LEGACY_EVALUATION_PROTOCOL,
    _write_or_validate_domain_manifest,
)
from examples.tau_bench.prepare_tau_training import (
    allocate_validation_counts,
    build_validation_rows,
)
from examples.tau_bench.validated_user_simulator import (
    ValidatedUserSimulator,
    validate_user_generation,
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
        "user_temperature": 1.0,
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
            _runtime_config(user_temperature=0.0),
            require_oracle=False,
        )
    config = _runtime_config()
    config.oracle.samples = 2
    with pytest.raises(RuntimeError, match="three oracle samples"):
        validate_tau_runtime_config(config, require_oracle=True)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "deepseek/deepseek-v4-flash",
            {"temperature": 1.0, "thinking": {"type": "disabled"}},
        ),
        (
            "openrouter/qwen/qwen3.6-27b",
            {"temperature": 1.0, "reasoning": {"enabled": False}},
        ),
    ],
)
def test_tau_user_simulator_uses_provider_native_reasoning_switch(model, expected):
    assert (
        tau_user_simulator_llm_args(
            model,
            temperature=1.0,
            reasoning_enabled=False,
        )
        == expected
    )


def test_native_tau_eval_supports_local_user_and_remote_fallback():
    root = Path(__file__).parents[2]
    driver = (root / "examples/tau_bench/native_tau_eval.py").read_text(encoding="utf-8")
    launcher = (root / "examples/tau_bench/run_tau_native_eval.sh").read_text(encoding="utf-8")

    assert '"DEEPSEEK_API_KEY"' in driver
    assert '"telecom-workflow"' in driver
    assert 'default="local"' in driver
    assert '"presence_penalty": args.user_presence_penalty' in driver
    assert '"repetition_penalty": args.user_repetition_penalty' in driver
    assert "deepseek | deepseek/*" in launcher
    assert "airline | retail | telecom | telecom-workflow" in launcher
    assert "USER_GPU_ID=0" in launcher
    assert "--tool-call-parser qwen3_xml" in launcher
    assert "--reasoning-parser qwen3" in launcher
    assert "--tool-call-parser hermes" in launcher
    assert '--user-generation-retries "$USER_GENERATION_RETRIES"' in launcher
    assert 'USER_MAX_TOKENS="${USER_MAX_TOKENS:-8192}"' in launcher
    assert 'AGENT_TOP_P="${AGENT_TOP_P:-1.0}"' in launcher
    assert 'AGENT_TOP_K="${AGENT_TOP_K:--1}"' in launcher
    assert 'parser.add_argument("--agent-temperature", type=float, default=0.0)' in driver
    assert 'parser.add_argument("--agent-top-p", type=float, default=1.0)' in driver
    assert 'parser.add_argument("--agent-top-k", type=int, default=-1)' in driver
    assert 'USER_MAX_MODEL_LEN="${USER_MAX_MODEL_LEN:-65536}"' in launcher
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"' in launcher
    assert 'AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.0}"' in launcher
    assert "TAU_COMPATIBILITY_PATCH_SHA256" in launcher
    assert 'agent="llm_agent"' in driver
    assert '"agent_protocol": "strict_native"' in driver
    assert 'echo "PROTOCOL_VERSION=6"' in launcher


def test_validated_local_user_rejects_truncated_and_empty_generations():
    valid = UserMessage(
        role="user",
        content="Yes, please continue.",
        raw_data={"choices": [{"finish_reason": "stop"}]},
    )
    validate_user_generation(valid)

    truncated = UserMessage(
        role="user",
        content="partial",
        raw_data={"choices": [{"finish_reason": "length"}]},
    )
    with pytest.raises(RuntimeError, match="truncated"):
        validate_user_generation(truncated)

    empty = UserMessage(
        role="user",
        content=None,
        raw_data={"choices": [{"finish_reason": "stop"}]},
    )
    with pytest.raises(RuntimeError, match="no final content"):
        validate_user_generation(empty)


def test_validated_local_user_retries_without_polluting_state(monkeypatch):
    simulator = ValidatedUserSimulator.__new__(ValidatedUserSimulator)
    simulator.validation_retries = 1
    simulator.llm_args = {"seed": 17}
    original_state = SimpleNamespace(messages=[])
    incoming = object()
    responses = [
        UserMessage(
            role="user",
            content="partial",
            raw_data={"choices": [{"finish_reason": "length"}]},
        ),
        UserMessage(
            role="user",
            content="Please continue.",
            raw_data={"choices": [{"finish_reason": "stop"}]},
        ),
    ]
    seen_seeds = []

    def fake_generate(self, message, state):
        seen_seeds.append(self.llm_args["seed"])
        response = responses.pop(0)
        state.messages.extend([message, response])
        return response, state

    monkeypatch.setattr(TauUserSimulator, "generate_next_message", fake_generate)
    response, updated_state = simulator.generate_next_message(incoming, original_state)

    assert response.content == "Please continue."
    assert original_state.messages == []
    assert updated_state.messages == [incoming, response]
    assert seen_seeds == [17, 17 + 104_729]
    assert simulator.llm_args["seed"] == 17


def test_native_eval_legacy_manifest_requires_explicit_protocol_upgrade(tmp_path, monkeypatch):
    manifest_path = tmp_path / "domain_manifest.json"
    current = {
        "evaluation_protocol": "tau_all_without_nl_assertions_replay_repair_v2",
        "evaluation_type": "all",
        "user_simulator_mode": "local",
        "user_sampling": {
            "max_tokens": 8192,
            "generation_retries": 2,
        },
    }
    legacy_without_protocol = json.loads(json.dumps(current))
    legacy_without_protocol.pop("evaluation_protocol")
    legacy_without_protocol["user_sampling"]["max_tokens"] = 4096
    legacy_without_protocol["user_sampling"].pop("generation_retries")
    manifest_path.write_text(json.dumps(legacy_without_protocol))

    monkeypatch.delenv("ALLOW_INFRASTRUCTURE_PROTOCOL_UPGRADE", raising=False)
    with pytest.raises(RuntimeError, match="Domain protocol changed"):
        _write_or_validate_domain_manifest(manifest_path, current)
    recorded = json.loads(manifest_path.read_text())
    assert recorded["evaluation_protocol"] == LEGACY_EVALUATION_PROTOCOL

    monkeypatch.setenv("ALLOW_INFRASTRUCTURE_PROTOCOL_UPGRADE", "1")
    _write_or_validate_domain_manifest(manifest_path, current)
    assert json.loads(manifest_path.read_text()) == current
    assert json.loads((tmp_path / "domain_manifest.pre_infrastructure_repair_v1.json").read_text()) == recorded


def test_tau_replay_skips_only_recorded_failed_unknown_tools():
    environment = Environment.__new__(Environment)
    environment.solo_mode = False
    environment.tools = SimpleNamespace(has_tool=lambda _name: False)
    environment.user_tools = None
    tool_call = ToolCall(
        id="call-1",
        name="invented_user_tool",
        arguments={},
        requestor="user",
    )
    tool_request = AssistantMessage(role="assistant", tool_calls=[tool_call])
    failed_result = ToolMessage(
        role="tool",
        id="call-1",
        content="Error: unknown tool",
        requestor="user",
        error=True,
    )
    environment.set_state(None, None, [tool_request, failed_result])

    non_error_result = failed_result.model_copy(update={"error": False})
    with pytest.raises(ValueError, match="Unknown tool"):
        environment.set_state(None, None, [tool_request, non_error_result])


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


def _tau_scoring_worker(mode):
    class Oracle:
        match_message_pairs = _AsyncRemoteMethod(
            lambda teacher_messages, candidate_messages: {
                "counts": [2, 1, 0],
                "matrix": [
                    [True, True, False],
                    [False, False, True],
                    [False, False, False],
                ],
            }
        )

    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=20,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=False,
        oracle_actor=Oracle(),
        teacher_reward_mode=mode,
        frequency_bonus_scale=0.5,
        seed=0,
    )
    worker._task_id = "task-1"
    worker._student_chat = lambda: [{"role": "user", "content": "task"}]
    worker._tools = lambda: []
    worker._validate = lambda action: action
    worker._last_observation = "current"
    worker._last_info = {"protocol_reward": 0.0}
    worker._execute = lambda action: (
        "next",
        0.0,
        False,
        {"protocol_reward": 0.0},
    )
    teacher_actions = [
        ParsedAction(kind="message", content="A"),
        ParsedAction(kind="message", content="A"),
        ParsedAction(kind="message", content="B"),
    ]
    fingerprint = state_fingerprint(
        "airline",
        "task-1",
        worker._student_chat(),
        [],
    )
    worker._prepared_teacher_supervision = {
        "state_fingerprint": fingerprint,
        "teacher_actions": teacher_actions,
        "teacher_sample_count": 3,
        "teacher_invalid_sample_count": 0,
        "teacher_unique_action_count": 2,
        "teacher_context_mode": "student_visible",
    }
    return worker


def test_tau_reward_mode_switches_multiset_scoring_and_advancement():
    raw_actions = ["A", "B", "C", ""]
    appearance = asyncio.run(_tau_scoring_worker("appearance").step_candidate_group(raw_actions))
    weighted = asyncio.run(_tau_scoring_worker("frequency_weighted").step_candidate_group(raw_actions))

    assert [row[1] for row in appearance[0]] == [1.0, 1.0, 0.0, -1.0]
    assert appearance[1] == 1
    assert [row[1] for row in weighted[0]] == [1.25, 1.0, 0.0, -1.0]
    assert weighted[1] == 0
    assert weighted[0][0][3]["teacher_frequency"] == 2
    assert weighted[0][1][3]["teacher_frequency"] == 1
    assert weighted[0][0][3]["teacher_reward_mode"] == "frequency_weighted"


def test_tau_message_candidates_are_unmatched_when_teacher_has_only_tools():
    worker = _tau_scoring_worker("appearance")
    worker._prepared_teacher_supervision.update(
        teacher_actions=[
            ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}),
            ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}),
            ParsedAction(kind="tool", name="lookup", arguments={"id": "2"}),
        ],
        teacher_unique_action_count=2,
    )

    result = asyncio.run(worker.step_candidate_group(["first", "second", "third", "fourth"]))

    assert [row[1] for row in result[0]] == [0.0, 0.0, 0.0, 0.0]
    assert all(row[3]["teacher_frequency"] == 0 for row in result[0])
    assert all(row[3]["matcher_matrix"] == [] for row in result[0])


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
        teacher_reward=SimpleNamespace(
            mode="appearance",
            frequency_bonus_scale=0.5,
        ),
        resources_per_worker={"num_cpus": 0.5, "num_gpus": 0},
        tau=SimpleNamespace(
            train_max_steps=20,
            eval_max_steps=30,
            user_llm="test-user",
            user_temperature=1.0,
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
        user_temperature=1.0,
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


def test_tau_worker_passes_deepseek_native_thinking_switch(monkeypatch):
    captured = {}

    def fake_make_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(tau_envs, "make_tau_agent_gym_env", fake_make_env)
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="deepseek/deepseek-v4-flash",
        user_temperature=1.0,
        user_reasoning_enabled=False,
    )

    worker._make_env("0")

    assert captured["user_llm_args"] == {
        "temperature": 1.0,
        "thinking": {"type": "disabled"},
    }


def test_tau_worker_marks_executed_native_tool_action():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=1.0,
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


def test_tau_manager_reports_transfer_ack_and_decision_limit_metrics():
    manager = TauBenchEnvironmentManager(None, None, None)
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "tau_domain": "airline",
                    "parsed_action": ('{"arguments":{"summary":"help"},"kind":"tool","name":"transfer_to_human_agents"}'),
                    "raw_action": "tool call",
                    "observation": "Transfer successful",
                },
                {
                    "tau_domain": "airline",
                    "parsed_action": json.dumps({"content": TRANSFER_HANDOFF_MESSAGE, "kind": "message"}),
                    "raw_action": f"<think>reason</think>{TRANSFER_HANDOFF_MESSAGE}",
                    "observation": "###TRANSFER###",
                    "terminal_success": False,
                    "terminal_reason": "environment_done",
                },
            ],
            [
                {
                    "tau_domain": "airline",
                    "parsed_action": json.dumps({"content": TRANSFER_HANDOFF_MESSAGE, "kind": "message"}),
                    "raw_action": TRANSFER_HANDOFF_MESSAGE,
                    "observation": "Please do not transfer me",
                    "terminal_success": False,
                    "terminal_reason": "decision_limit",
                    "decision_limit_reached": True,
                }
            ],
        ]
    )

    assert metrics["env/trajectory_count"].tolist() == [2.0]
    assert metrics["env/airline/trajectory_count"].tolist() == [2.0]
    assert metrics["env/transfer_tool_call_rate"].tolist() == [1.0, 0.0]
    assert metrics["env/transfer_handoff_rate"].tolist() == [1.0, 1.0]
    assert metrics["env/transfer_acknowledged_rate"].tolist() == [1.0, 0.0]
    assert metrics["env/transfer_ack_failure_rate"].tolist() == [0.0, 1.0]
    assert metrics["env/transfer_handoff_count"].tolist() == [2.0]
    assert metrics["env/transfer_ack_success_rate_given_handoff"].tolist() == [
        1.0,
        0.0,
    ]
    assert metrics["env/decision_limit_rate"].tolist() == [0.0, 1.0]


def test_tau_worker_records_forced_decision_limit():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=False,
    )
    worker._step = 2
    worker._env = SimpleNamespace(step=lambda action: ("final", 0.0, True, False, {}))

    _, _, done, _ = worker._finalize_at_decision_limit("before", 0.0, False, {})

    assert done is True
    assert worker._last_step_hit_decision_limit is True


class _AsyncRemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_tau_teacher_preflight_defaults_to_exact_student_visible_context():
    calls = []

    class Oracle:
        sample_multiset = _AsyncRemoteMethod(
            lambda **kwargs: (
                calls.append(kwargs)
                or [
                    {"kind": "message", "content": "I can help."},
                    {"kind": "message", "content": "I can help."},
                    {"kind": "message", "content": "Another answer."},
                ]
            )
        )

    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=20,
        user_llm="deepseek/deepseek-v4-flash",
        user_temperature=1.0,
        user_reasoning_enabled=False,
        oracle_actor=Oracle(),
        use_privileged_teacher_context=False,
    )
    logical_chat = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "greeting"},
        {"role": "user", "content": "task"},
    ]
    visible_chat = [logical_chat[0], logical_chat[2]]
    worker._task_id = "task-1"
    worker._student_chat = lambda: logical_chat
    worker._tools = lambda: []
    worker._validate = lambda action: action
    worker._teacher_privileged_context = lambda: (_ for _ in ()).throw(AssertionError("default mode must not read hidden task metadata"))
    worker._validate_teacher_visible_chat(logical_chat)

    ready, info = asyncio.run(worker.prepare_teacher_supervision(visible_chat))

    assert ready is True
    assert info["teacher_context_mode"] == "student_visible"
    assert calls[0]["messages"] == visible_chat
    assert calls[0]["teacher_context_mode"] == "student_visible"
    assert isinstance(
        worker._prepared_teacher_supervision["teacher_actions"][0],
        ParsedAction,
    )
    assert worker._prepared_teacher_supervision["teacher_sample_count"] == 3
    assert worker._prepared_teacher_supervision["teacher_unique_action_count"] == 2
