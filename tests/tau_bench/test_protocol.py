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
    canonical_action,
    state_fingerprint,
    validate_tau_action,
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
from agent_system.environments.env_package.tau_bench.user_simulator import (
    ValidatedUserSimulator,
    validate_user_generation,
)
from examples.tau_bench.eval.native_eval import (
    LEGACY_EVALUATION_PROTOCOL,
    _write_or_validate_domain_manifest,
)
from examples.tau_bench.train.prepare_data import (
    allocate_validation_counts,
    build_validation_rows,
)


def test_grouped_domain_schedule_keeps_outcome_replicas_contiguous():
    labels = interleave_grouped_domains({"airline": 4, "retail": 4}, group_n=4)
    assert len(labels) == 32
    assert all(len(set(labels[index : index + 4])) == 1 for index in range(0, 32, 4))
    assert labels.count("airline") == labels.count("retail") == 16


def test_official_tau_task_counts_are_explicit():
    assert TASK_MANIFEST_PROTOCOL_VERSION == 3
    assert OFFICIAL_TASK_COUNTS == {
        "train": {"airline": 30, "retail": 74},
        "test": {"airline": 20, "retail": 40},
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
        "user_reasoning_enabled": True,
        "oracle": SimpleNamespace(
            model="qwen3-32b",
            samples=3,
        ),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_runtime_config_requires_fixed_user_and_k3_oracle():
    validate_tau_runtime_config(_runtime_config(), require_oracle=True)

    with pytest.raises(RuntimeError, match="thinking"):
        validate_tau_runtime_config(
            _runtime_config(user_reasoning_enabled=False),
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


def test_tau_user_simulator_uses_qwen_recommended_sampling(monkeypatch):
    monkeypatch.setenv("TAU_USER_API_KEY", "test-only")
    values = tau_user_simulator_llm_args(
        "openai/qwen3.5-9b",
        api_base="http://user.example/v1",
        api_key_env="TAU_USER_API_KEY",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=8192,
        reasoning_enabled=True,
        generation_retries=2,
    )
    assert values["api_base"] == "http://user.example/v1"
    assert values["api_key"] == "test-only"
    assert values["temperature"] == 1.0
    assert values["top_p"] == 0.95
    assert values["presence_penalty"] == 1.5
    assert values["max_tokens"] == 8192
    assert values["_validation_retries"] == 2
    assert values["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_tau_native_logging_replaces_default_sink(monkeypatch):
    calls = []

    class FakeLogger:
        def remove(self):
            calls.append(("remove",))

        def add(self, sink, **kwargs):
            calls.append(("add", sink, kwargs))

    monkeypatch.setattr("loguru.logger", FakeLogger())
    assert tau_envs.configure_tau_native_logging("warning") == "WARNING"
    assert calls[0] == ("remove",)
    assert calls[1][0] == "add"
    assert calls[1][2]["level"] == "WARNING"
    assert calls[1][2]["backtrace"] is False
    assert calls[1][2]["diagnose"] is False

    with pytest.raises(ValueError, match="unsupported Tau native log level"):
        tau_envs.configure_tau_native_logging("TRACE")


def test_native_tau_eval_supports_remote_first_and_local_fallback():
    root = Path(__file__).parents[2]
    driver = (root / "examples/tau_bench/eval/native_eval.py").read_text(encoding="utf-8")
    launcher = (root / "examples/tau_bench/eval/run.sh").read_text(encoding="utf-8")

    assert '"telecom-workflow"' in driver
    assert 'default="remote"' in driver
    assert 'parser.add_argument("--task-split", default="test")' in driver
    assert '"presence_penalty": args.user_presence_penalty' in driver
    assert '"repetition_penalty": args.user_repetition_penalty' in driver
    assert 'USER_SIMULATOR_MODE="${USER_SIMULATOR_MODE:-auto}"' in launcher
    assert "TAU_USER_API_BASE_EXPLICIT=0" in launcher
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


def _tau_scoring_worker(mode, domain="airline"):
    class Oracle:
        match_message_pairs = _AsyncRemoteMethod(
            lambda teacher_messages, candidate_messages, *context: {
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
        domain=domain,
        max_steps=20,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=True,
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
        domain,
        "task-1",
        worker._student_chat(),
        [],
    )
    worker._prepared_teacher_supervision = {
        "state_fingerprint": fingerprint,
        "teacher_actions": teacher_actions,
        "teacher_multiset": [action.to_dict() for action in teacher_actions],
        "teacher_sample_count": 3,
        "teacher_invalid_sample_count": 0,
        "teacher_unique_action_count": 2,
        "teacher_action_kind_disagreement": False,
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
    assert weighted[0][0][3]["agentic_env_family"] == "tau"
    assert weighted[0][0][3]["action_kind"] == "message"
    assert [action["content"] for action in weighted[0][0][3]["teacher_multiset"]] == ["A", "A", "B"]
    assert weighted[0][0][3]["teacher_multiset_size"] == 3
    assert weighted[0][0][3]["teacher_action_kind_disagreement"] is False
    assert weighted[0][0][3]["frequency_sensitive_group"] is True
    assert weighted[0][0][3]["semantic_train_mask"] is True
    assert weighted[0][0][3]["matcher_failure"] is False


def test_tau_matcher_ablation_masks_whole_group_but_executes_one_valid_action():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.mask_matcher_required_groups = True
    worker.oracle_actor = object()  # Any matcher access would fail.
    prepared = dict(worker._prepared_teacher_supervision)
    executed = []
    worker._execute = lambda action: (executed.append(action) or "next", 0.0, False, {})
    rows, selected, _, _, done, info = asyncio.run(worker.step_candidate_group(["A", "B", "C", ""]))
    assert not done and not info["matcher_failure"] and not info["teacher_failure"]
    assert selected in {0, 1, 2} and len(executed) == 1
    assert all(not row[3]["semantic_train_mask"] for row in rows)
    assert all(row[3]["runtime_train_mask"] and row[3]["matcher_required_group"] for row in rows)
    assert [row[1] for row in rows] == [0.0] * 4
    assert sum(row[3]["state_group_advanced"] for row in rows) == 1
    assert info["state_group_selection_type"] == "random"
    assert info["matcher_matrix"] == []
    # Only this state was masked; the following state can train, including a
    # partial teacher multiset whose reward denominator must still be K=3.
    prepared.update(teacher_actions=prepared["teacher_actions"][:2], teacher_multiset=prepared["teacher_multiset"][:2], teacher_invalid_sample_count=1, teacher_unique_action_count=1)
    worker._prepared_teacher_supervision = prepared
    following = asyncio.run(worker.step_candidate_group(["A", " A ", '<tool_call>{"name":"lookup","arguments":{}}</tool_call>', ""]))
    assert [row[1] for row in following[0]] == [1.25, 1.25, 0.0, -1.0]
    assert all(row[3]["semantic_train_mask"] and not row[3]["matcher_required_group"] for row in following[0])
    assert len(executed) == 2 and not following[4]


def test_tau_matcher_ablation_unknown_tool_arguments_mask_without_source_or_api():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.mask_matcher_required_groups = True
    worker.oracle_actor = object()
    teacher = ParsedAction(kind="tool", name="lookup", arguments={"id": "1"})
    worker._prepared_teacher_supervision.update(teacher_actions=[teacher] * 3, teacher_multiset=[teacher.to_dict()] * 3, teacher_unique_action_count=1)
    raw = ['<tool_call>{"name":"lookup","arguments":{"id":"1"}}</tool_call>', '<tool_call>{"name":"lookup","arguments":{"id":"2"}}</tool_call>', "message", ""]
    result = asyncio.run(worker.step_candidate_group(raw))
    assert not result[4]
    assert all(row[3]["matcher_required_group"] and not row[3]["semantic_train_mask"] for row in result[0])


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_tau_matcher_ablation_preserves_transfer_summary_programmatic_match(domain):
    from tau2.domains.airline.tools import AirlineTools
    from tau2.domains.retail.tools import RetailTools

    from agent_system.environments.tool_matching_metadata import callable_tool_matching_metadata

    worker = _tau_scoring_worker("frequency_weighted", domain=domain)
    worker.mask_matcher_required_groups = True
    worker.oracle_actor = object()
    schema = {"type": "function", "function": {"name": "transfer_to_human_agents", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}}
    native = AirlineTools if domain == "airline" else RetailTools
    tool = SimpleNamespace(name="transfer_to_human_agents", _func=native.transfer_to_human_agents, openai_schema=schema)
    worker._tool_matching_metadata = callable_tool_matching_metadata([tool], family="tau", environment=domain)
    teacher = ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": "teacher"})
    worker._prepared_teacher_supervision.update(teacher_actions=[teacher] * 3, teacher_multiset=[teacher.to_dict()] * 3, teacher_unique_action_count=1)
    result = asyncio.run(worker.step_candidate_group(['<tool_call>{"name":"transfer_to_human_agents","arguments":{"summary":"student"}}</tool_call>', "message", "", ""]))
    assert [row[1] for row in result[0]] == [1.5, 0.0, -1.0, -1.0]
    assert all(row[3]["semantic_train_mask"] for row in result[0])
    assert result[1] == 0


def test_tau_matcher_ablation_metrics_do_not_count_unscored_as_misses():
    manager = object.__new__(TauBenchEnvironmentManager)
    manager.oracle_actor = None
    masked = dict(tau_domain="airline", action_kind="message", matcher_required_group=True, semantic_train_mask=False, move_optimal=False, teacher_frequency=0)
    scored = dict(tau_domain="airline", action_kind="tool", matcher_required_group=False, semantic_train_mask=True, move_optimal=True, teacher_frequency=3)
    metrics = manager.success_evaluator(total_infos=[[masked, scored]], total_batch_list=[[masked, scored]])
    assert metrics["env/matcher_required_group_rate"].item() == 0.5
    assert metrics["env/airline/matcher_required_group_rate"].item() == 0.5
    assert metrics["env/oracle_hit_rate"].item() == 1.0
    assert metrics["env/selected_message_action_rate"].item() == 0.5
    assert metrics["env/tool_candidate_teacher_match_count_mean"].item() == 3.0


def test_tau_matcher_ablation_loss_mask_does_not_drop_other_states_in_trajectory():
    import numpy as np
    import torch

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage

    rewards = np.asarray([0.0] * 4 + [1.5, 0.0, -1.0, 1.5], dtype=np.float32)
    token_rewards = torch.zeros((8, 2))
    token_rewards[:, -1] = torch.from_numpy(rewards)
    batch = DataProto.from_dict(
        tensors={"response_mask": torch.ones((8, 2)), "token_level_rewards": token_rewards},
        non_tensors={
            "uid": np.asarray(["one-trajectory"] * 8, dtype=object),
            "state_group_uid": np.asarray(["unscored"] * 4 + ["scored"] * 4, dtype=object),
            "rewards": rewards,
            "semantic_train_mask": np.asarray([False] * 4 + [True] * 4),
            "runtime_train_mask": np.ones(8, dtype=bool),
            "is_padding": np.zeros(8, dtype=bool),
            "vpr_game": np.asarray(["tau_airline"] * 8, dtype=object),
        },
    )
    result = compute_advantage(batch, AdvantageEstimator.DAPO)
    assert result.non_tensor_batch["dapo_skip_loss"].tolist() == [True] * 4 + [False] * 4
    assert result.batch["response_mask"][:4].sum() == 0
    assert result.batch["advantages"][:4].abs().sum() == 0
    assert result.batch["advantages"][4:].abs().sum() > 0
    logprobs = torch.zeros((8, 2), requires_grad=True)
    loss = -(logprobs * result.batch["advantages"] * result.batch["response_mask"]).sum()
    loss.backward()
    assert logprobs.grad[:4].abs().sum() == 0
    assert logprobs.grad[4:].abs().sum() > 0


@pytest.mark.parametrize("domain", ["airline", "retail"])
@pytest.mark.parametrize("mode", ["appearance", "frequency_weighted"])
@pytest.mark.parametrize("transfer_votes,lookup_votes", [(0, 3), (1, 2), (2, 1), (3, 0), (1, 0), (2, 0)])
def test_tau_transfer_rewards_count_all_votes_without_matching_summary(domain, mode, transfer_votes, lookup_votes):
    from pydantic import BaseModel

    class TransferParams(BaseModel):
        summary: str

    class LookupParams(BaseModel):
        id: str

    tools = [
        SimpleNamespace(name="transfer_to_human_agents", params=TransferParams),
        SimpleNamespace(name="lookup", params=LookupParams),
    ]
    worker = _tau_scoring_worker(mode, domain=domain)
    worker._validate = lambda action: validate_tau_action(action, tools)
    from tau2.domains.airline.tools import AirlineTools
    from tau2.domains.retail.tools import RetailTools

    from agent_system.environments.tool_matching_metadata import callable_tool_matching_metadata, source_tool_matching_metadata

    schemas = [{"type": "function", "function": {"name": tool.name, "parameters": tool.params.model_json_schema()}} for tool in tools]
    native_class = AirlineTools if domain == "airline" else RetailTools
    transfer = SimpleNamespace(name="transfer_to_human_agents", _func=native_class.transfer_to_human_agents, openai_schema=schemas[0])
    worker._tool_matching_metadata = callable_tool_matching_metadata([transfer], family="tau", environment=domain)
    worker._tool_matching_metadata.update(source_tool_matching_metadata("def lookup(id):\n    return db[id]\n", [schemas[1]], family="test", environment=domain))
    worker._tools = lambda: schemas
    worker._prepared_teacher_supervision["state_fingerprint"] = state_fingerprint(domain, "task-1", worker._student_chat(), schemas)
    worker.oracle_actor.match_tool_argument_pairs = _AsyncRemoteMethod(lambda pairs: [False] * len(pairs))
    executed = []

    def execute(action):
        executed.append(action)
        return "next", 0.0, False, {"protocol_reward": 0.0}

    worker._execute = execute
    teacher_actions = [ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": f"Teacher explanation {index}"}) for index in range(transfer_votes)] + [ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}) for _ in range(lookup_votes)]
    teacher_multiset = [action.to_dict() for action in teacher_actions]
    worker._prepared_teacher_supervision.update(
        teacher_actions=teacher_actions,
        teacher_multiset=teacher_multiset,
        teacher_invalid_sample_count=3 - len(teacher_actions),
        teacher_unique_action_count=len({canonical_action(action) for action in teacher_actions}),
    )
    candidates = [
        ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": "Student's own explanation"}),
        ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}),
        ParsedAction(kind="tool", name="lookup", arguments={"id": "2"}),
        ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": 123}),
    ]
    raw_actions = [f"<tool_call>{json.dumps({'name': action.name, 'arguments': action.arguments})}</tool_call>" for action in candidates]
    result = asyncio.run(worker.step_candidate_group(raw_actions))
    rows, selected_index = result[:2]

    reward_by_count = {0: 0.0, 1: 1.0, 2: 1.25, 3: 1.5} if mode == "frequency_weighted" else {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
    expected_rewards = [reward_by_count[transfer_votes], reward_by_count[lookup_votes], 0.0, -1.0]
    assert [row[1] for row in rows] == expected_rewards
    assert [row[3]["teacher_frequency"] for row in rows] == [transfer_votes, lookup_votes, 0, 0]
    assert rows[3][3]["action_kind"] == "invalid"
    assert rows[0][3]["parsed_action"] == canonical_action(candidates[0])
    assert rows[0][3]["teacher_multiset"] == teacher_multiset
    assert rows[0][3]["teacher_multiset_size"] == len(teacher_actions)
    assert rows[0][3]["teacher_sample_count"] == 3
    assert rows[0][3]["teacher_valid_sample_count"] == len(teacher_actions)
    assert all(row[3]["matcher_matrix"] == [] for row in rows)
    assert all(row[3]["semantic_train_mask"] for row in rows)
    assert expected_rewards[selected_index] == max(expected_rewards)
    assert executed == [candidates[selected_index]]
    assert sum(row[3]["state_group_advanced"] for row in rows) == 1


def test_tau_partial_valid_teacher_set_keeps_k3_reward_denominator():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.oracle_actor.match_message_pairs = _AsyncRemoteMethod(
        lambda teacher_messages, candidate_messages, *context: {
            "counts": [2, 0, 0],
            "matrix": [
                [True, True],
                [False, False],
                [False, False],
            ],
        }
    )
    teacher_actions = [
        ParsedAction(kind="message", content="A"),
        ParsedAction(kind="message", content="A"),
    ]
    worker._prepared_teacher_supervision.update(
        teacher_actions=teacher_actions,
        teacher_multiset=[action.to_dict() for action in teacher_actions],
        teacher_sample_count=3,
        teacher_invalid_sample_count=1,
        teacher_unique_action_count=1,
        teacher_action_kind_disagreement=False,
    )

    result = asyncio.run(worker.step_candidate_group(["A", "B", "C", ""]))

    assert result[0][0][1] == 1.25
    assert result[0][0][3]["teacher_sample_count"] == 3
    assert result[0][0][3]["teacher_valid_sample_count"] == 2
    assert result[0][0][3]["teacher_invalid_sample_count"] == 1


def test_tau_message_candidates_are_unmatched_when_teacher_has_only_tools():
    worker = _tau_scoring_worker("appearance")
    teacher_actions = [
        ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}),
        ParsedAction(kind="tool", name="lookup", arguments={"id": "1"}),
        ParsedAction(kind="tool", name="lookup", arguments={"id": "2"}),
    ]
    worker._prepared_teacher_supervision.update(
        teacher_actions=teacher_actions,
        teacher_multiset=[action.to_dict() for action in teacher_actions],
        teacher_unique_action_count=2,
        teacher_action_kind_disagreement=False,
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
            user_api_base="http://user.example/v1",
            user_api_key_env="TAU_USER_API_KEY",
            user_top_p=0.95,
            user_top_k=20,
            user_min_p=0.0,
            user_presence_penalty=1.5,
            user_repetition_penalty=1.0,
            user_max_tokens=8192,
            user_generation_retries=2,
            native_log_level="WARNING",
            user_temperature=1.0,
            user_reasoning_enabled=True,
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
        user_reasoning_enabled=True,
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


def test_tau_worker_passes_qwen_recommended_user_sampling(monkeypatch):
    captured = {}
    monkeypatch.setenv("TAU_USER_API_KEY", "test-only")

    def fake_make_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(tau_envs, "make_tau_agent_gym_env", fake_make_env)
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="openai/qwen3.5-9b",
        user_temperature=1.0,
        user_reasoning_enabled=True,
    )

    worker._make_env("0")

    user_args = captured["user_llm_args"]
    assert user_args["temperature"] == 1.0
    assert user_args["top_p"] == 0.95
    assert user_args["presence_penalty"] == 1.5
    assert user_args["max_tokens"] == 8192
    assert user_args["_validation_retries"] == 2
    assert user_args["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_tau_worker_marks_executed_native_tool_action():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=True,
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


def test_tau_manager_reports_generic_teacher_and_context_health_metrics():
    manager = TauBenchEnvironmentManager(None, None, None)
    metrics = manager.success_evaluator(
        total_infos=[
            [
                {
                    "tau_domain": "airline",
                    "action_kind": "tool",
                    "is_action_valid": 1,
                    "teacher_sample_count": 3,
                    "teacher_failure": False,
                    "matcher_failure": False,
                    "frequency_sensitive_group": True,
                    "teacher_action_kind_disagreement": True,
                }
            ],
            [
                {
                    "tau_domain": "retail",
                    "action_kind": "matcher_failure",
                    "teacher_sample_count": 3,
                    "teacher_failure": False,
                    "matcher_failure": True,
                }
            ],
            [
                {
                    "tau_domain": "retail",
                    "action_kind": "teacher_failure",
                    "teacher_sample_count": 0,
                    "teacher_failure": True,
                    "matcher_failure": False,
                }
            ],
            [
                {
                    "tau_domain": "retail",
                    "action_kind": "context_overflow",
                    "teacher_failure": False,
                    "matcher_failure": False,
                    "context_overflow": True,
                    "context_prompt_tokens": 30000,
                    "context_excess_tokens": 2096,
                }
            ],
        ],
        total_batch_list=[
            [{"semantic_train_mask": True} for _ in range(4)],
            [{"semantic_train_mask": False} for _ in range(4)],
            [],
            [],
        ],
    )

    assert metrics["env/teacher_failure_state_rate"].tolist() == pytest.approx([1 / 3])
    assert metrics["env/matcher_failure_rate"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert metrics["env/semantic_masked_rate"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert metrics["env/frequency_sensitive_group_rate"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics["env/teacher_action_kind_disagreement_rate"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics["env/context_overflow_rate"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert metrics["env/context_overflow_prompt_tokens_mean"].tolist() == [
        30000.0,
        30000.0,
        30000.0,
        30000.0,
    ]
    assert metrics["env/context_overflow_excess_tokens_mean"].tolist() == [
        2096.0,
        2096.0,
        2096.0,
        2096.0,
    ]


def test_tau_worker_records_forced_decision_limit():
    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=2,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=True,
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


class _FinalizingTauEnv:
    def __init__(self, reward=0.0):
        self.reward = float(reward)
        self.actions = []

    def step(self, action):
        self.actions.append(json.loads(action))
        return "final", self.reward, True, False, {"protocol_reward": self.reward}


def _tau_preflight_worker(sample_fn):
    class Oracle:
        sample_multiset = _AsyncRemoteMethod(sample_fn)

    worker_class = TauBenchWorker.__ray_metadata__.modified_class
    worker = worker_class(
        domain="airline",
        max_steps=20,
        user_llm="test-user",
        user_temperature=1.0,
        user_reasoning_enabled=True,
        oracle_actor=Oracle(),
        teacher_reward_mode="frequency_weighted",
        frequency_bonus_scale=0.5,
        seed=0,
    )
    worker._task_id = "task-failure"
    worker._student_chat = lambda: [{"role": "user", "content": "task"}]
    worker._tools = lambda: []
    worker._validate = lambda action: action
    worker._last_observation = "current"
    worker._last_info = {"protocol_reward": 0.0}
    worker._env = _FinalizingTauEnv()
    return worker


def test_tau_teacher_api_failure_terminates_only_current_trajectory():
    def fail(**kwargs):
        raise TimeoutError("teacher unavailable")

    worker = _tau_preflight_worker(fail)
    ready, info = asyncio.run(worker.prepare_teacher_supervision())

    assert ready is False
    assert worker._done is True
    assert info["teacher_failure"] is True
    assert info["semantic_train_mask"] is False
    assert info["terminal_reason"] == "teacher_failure"
    assert info["terminal_outcome_valid"] is True
    assert worker._env.actions == [{"name": "done", "arguments": {}}]


def test_tau_all_invalid_teacher_samples_are_a_masked_trajectory_failure():
    worker = _tau_preflight_worker(lambda **kwargs: [{"kind": "invalid", "error": f"bad-{index}"} for index in range(3)])

    ready, info = asyncio.run(worker.prepare_teacher_supervision())

    assert ready is False
    assert info["teacher_sample_count"] == 3
    assert info["teacher_valid_sample_count"] == 0
    assert info["teacher_invalid_sample_count"] == 3
    assert info["teacher_failure"] is True
    assert info["semantic_train_mask"] is False


def test_tau_malformed_teacher_payload_is_a_masked_trajectory_failure():
    worker = _tau_preflight_worker(lambda **kwargs: [{"kind": "message", "content": "valid", "unexpected": True} for _ in range(3)])

    ready, info = asyncio.run(worker.prepare_teacher_supervision())

    assert ready is False
    assert info["teacher_sample_count"] == 3
    assert info["teacher_invalid_sample_count"] == 3
    assert info["teacher_failure"] is True
    assert "unexpected" in info["teacher_error"]


def test_tau_partial_teacher_samples_are_ready_with_fixed_k3_denominator():
    worker = _tau_preflight_worker(
        lambda **kwargs: [
            {"kind": "message", "content": "first"},
            {"kind": "message", "content": "second"},
        ]
    )

    ready, info = asyncio.run(worker.prepare_teacher_supervision())

    assert ready is True
    assert worker._done is False
    assert worker._env.actions == []
    assert info["teacher_sample_count"] == 3
    assert info["teacher_valid_sample_count"] == 2
    assert info["teacher_invalid_sample_count"] == 1
    assert len(info["teacher_multiset"]) == 2


def test_tau_oversized_teacher_sample_set_still_fails_loudly():
    worker = _tau_preflight_worker(lambda **kwargs: [{"kind": "message", "content": f"vote-{index}"} for index in range(4)])

    with pytest.raises(RuntimeError, match="4 samples; expected at most 3"):
        asyncio.run(worker.prepare_teacher_supervision())
    assert worker._done is False
    assert worker._env.actions == []


def test_tau_matcher_failure_masks_group_without_executing_candidate():
    worker = _tau_scoring_worker("frequency_weighted")
    worker._env = _FinalizingTauEnv()

    class Oracle:
        match_message_pairs = _AsyncRemoteMethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("matcher unavailable")))

    worker.oracle_actor = Oracle()
    worker._execute = lambda action: (_ for _ in ()).throw(AssertionError("matcher failure must not execute a candidate"))

    result = asyncio.run(
        worker.step_candidate_group(
            ["A", "B", "C", ""],
            group_metadata={"test_group": "matcher"},
        )
    )

    candidate_results, selected_index, _, selected_reward, done, info = result
    assert selected_index == -1
    assert selected_reward == 0.0
    assert done is True
    assert info["matcher_failure"] is True
    assert info["terminal_reason"] == "matcher_failure"
    assert all(row[1] == 0.0 for row in candidate_results)
    assert all(row[2] is False for row in candidate_results)
    assert all(row[3]["semantic_train_mask"] is False for row in candidate_results)
    assert all(row[3]["state_group_advanced"] is False for row in candidate_results)
    assert all(row[3]["test_group"] == "matcher" for row in candidate_results)


def test_tau_context_overflow_terminates_without_student_action():
    worker = _tau_preflight_worker(lambda **kwargs: [{"kind": "message", "content": "unused"} for _ in range(3)])

    info = worker.terminate_context_overflow({"context_prompt_tokens": 30000, "context_excess_tokens": 2096})

    assert worker._done is True
    assert info["action_kind"] == "context_overflow"
    assert info["semantic_train_mask"] is False
    assert info["runtime_train_mask"] is False
    assert info["terminal_reason"] == "context_budget_exceeded"
    assert info["context_prompt_tokens"] == 30000
    assert info["context_excess_tokens"] == 2096
    assert worker._env.actions == [{"name": "done", "arguments": {}}]


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
        user_reasoning_enabled=True,
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


@pytest.mark.parametrize("mode", ["frequency_weighted", "appearance"])
@pytest.mark.parametrize("enabled,transferred,expected_guard", [(True, False, True), (True, True, False), (False, False, False)])
def test_transfer_guard_penalizes_only_final_reward_not_validity_or_votes(mode, enabled, transferred, expected_guard):
    worker = _tau_scoring_worker(mode)
    worker.transfer_reward_guard_enabled = enabled
    worker._transfer_succeeded = transferred
    results, selected, _, _, done, _ = asyncio.run(worker.step_candidate_group([TRANSFER_HANDOFF_MESSAGE, "B", "C", ""]))
    info = results[0][3]
    assert info["transfer_without_tool"] is expected_guard
    assert info["teacher_match_count"] == 2
    assert info["matcher_matrix"] == [True, True, False]
    assert info["raw_semantic_reward"] > 0
    assert info["semantic_train_mask"] and info["runtime_train_mask"] and info["is_action_valid"]
    assert not done
    if expected_guard:
        assert results[0][1] == info["selection_score"] == -1.0
        assert selected == 1
        assert results[1][3]["appearance_counterfactual_selected"]
    else:
        assert results[0][1] == info["raw_semantic_reward"]


def test_all_negative_transfer_group_keeps_existing_execution_and_equal_reward_handling():
    worker = _tau_scoring_worker("frequency_weighted")
    worker.transfer_reward_guard_enabled = True
    executed = []
    worker._execute = lambda action: (executed.append(action) or "next", 0.0, False, {"protocol_reward": 0.0})
    results, selected, _, _, done, _ = asyncio.run(worker.step_candidate_group([TRANSFER_HANDOFF_MESSAGE] * 3))
    assert [row[1] for row in results] == [-1.0] * 3
    assert all(row[3]["semantic_train_mask"] and row[3]["is_action_valid"] for row in results)
    assert len(executed) == 1 and selected in range(3) and not done
    # No task quarantine or special execution veto; the common zero-std rule skips loss.


def test_transfer_success_is_recorded_only_after_execution_and_cleared_on_reset():
    from types import SimpleNamespace

    worker = _tau_scoring_worker("frequency_weighted")
    worker.transfer_reward_guard_enabled = True
    native_call = SimpleNamespace(role="assistant", content="", tool_calls=[SimpleNamespace(id="call-1", name="transfer_to_human_agents", arguments={"summary": "help"})])
    native_result = SimpleNamespace(role="tool", id="call-1", requestor="assistant", content="Transfer successful")
    history = []

    def step(action):
        history.extend([native_call, native_result])
        return "tool: Transfer successful", 0.0, False, False, {}

    worker._env = SimpleNamespace(step=step, close=lambda: None, _agent=SimpleNamespace(observation=history))
    assert worker._transfer_succeeded is False
    native_execute = TauBenchWorker.__ray_metadata__.modified_class._execute
    native_execute(worker, ParsedAction(kind="tool", name="transfer_to_human_agents", arguments={"summary": "help"}))
    assert worker._transfer_succeeded is True
    # Budget trimming cannot erase a successfully executed handoff.
    history.clear()
    worker._env.step = lambda action: ("next", 0.0, False, False, {})
    native_execute(worker, ParsedAction(kind="message", content=TRANSFER_HANDOFF_MESSAGE))
    assert worker._transfer_succeeded is True
    worker._make_env = lambda task_id: SimpleNamespace(reset=lambda **kwargs: ("reset", {}))
    worker._observation_info = lambda: (worker._last_observation, worker._last_info)
    worker.reset(task_id="new-task")
    assert worker._transfer_succeeded is False


def test_transfer_guard_metric_counts_only_nonpadding_candidates():
    manager = TauBenchEnvironmentManager(None, None, None)
    metrics = manager.success_evaluator(total_infos=[[]], total_batch_list=[[{"transfer_without_tool": True}, {"transfer_without_tool": False}, {"transfer_without_tool": True, "is_padding": True}]])
    assert metrics["env/transfer_without_tool_candidate_rate"].tolist() == [0.5]
