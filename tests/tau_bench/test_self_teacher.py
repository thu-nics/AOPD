"""CPU contract tests: real DataProto/tokenization; no model API or GPU needed."""

import asyncio
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.environments.env_package.tau_bench.actions import parse_action
from agent_system.environments.env_package.tau_bench.oracle import TauTeacherClient
from agent_system.environments.env_package.tau_bench.paired_training import build_plan
from agent_system.environments.env_package.tau_bench.self_teacher import (
    SelfTeacherRollout,
    build_self_teacher_messages,
    inference_batch,
    parse_teacher_vote,
)
from verl import DataProto

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
CALL = '<tool_call>{"name":"lookup","arguments":{"id":"a"}}</tool_call>'
PUBLIC = [{"role": "system", "content": "Public policy"}, {"role": "user", "content": "Help me"}]


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True) + json.dumps(kwargs.get("tools"))

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def batch_decode(self, tokens, **kwargs):
        return ["".join(chr(int(c)) for c in row if c) for row in tokens]


def config(tmp_path):
    from agent_system.environments.env_package.tau_bench.customer_briefs import BRIEF_PROTOCOL, digest

    scenario = {"instructions": {"known_info": "SECRET"}}
    briefs = tmp_path / "briefs.json"
    briefs.write_text(
        json.dumps(
            {
                "protocol": BRIEF_PROTOCOL,
                "status": "frozen",
                "records": [{"domain": "airline", "task_id": "test", "user_scenario": scenario, "scenario_sha256": digest(scenario), "review_status": "approved", "facts": [{"category": "known", "text": "SECRET", "source_field": "instructions.known_info", "quote": "SECRET"}]}],
            }
        )
    )
    return OmegaConf.create(
        {
            "env": {
                "rollout": {"n": 4},
                "tau": {
                    "source_root": "/tau",
                    "oracle": {
                        "samples": 3,
                        "teacher_cache_import_paths": [],
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "max_tokens": 4096,
                        "enable_thinking": True,
                        "self_extra_prompt_tokens": 4096,
                        "self_customer_briefs_path": str(briefs),
                        "self_privilege_mode": "customer",
                        "teacher_validity_max_retries": 2,
                        "cache_path": str(tmp_path / "teacher.jsonl"),
                        "use_privileged_context": True,
                    },
                },
            },
            "actor_rollout_ref": {
                "model": {"path": "/model"},
                "rollout": {
                    "name": "vllm",
                    "mode": "sync",
                    "do_sample": True,
                    "response_length": 4096,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "n": 1,
                    "max_model_len": 32768,
                },
            },
            "data": {"max_response_length": 4096, "max_prompt_length": 24576, "apply_chat_template_kwargs": {"enable_thinking": True}},
        }
    )


def request():
    return {"public_chat": copy.deepcopy(PUBLIC), "messages": build_self_teacher_messages(PUBLIC, {"user_scenario": "SECRET"}), "tools": TOOLS, "domain": "airline", "task_id": "train-0", "state_fingerprint": "fingerprint"}


class Actor:
    world_size = 2

    def __init__(self, teacher_text=None):
        self.calls = []
        self.teacher_text = teacher_text or (lambda slot, attempt: ("</think>" + CALL, "stop"))

    def teacher_response(self, name):
        _, _, slot, attempt = name.split(":")
        return self.teacher_text(int(slot), int(attempt))

    def generate_sequences(self, batch):
        names = list(batch.non_tensor_batch["self_request_id"])
        self.calls.append(names)
        values, reasons = [], []
        for name in names:
            if name.startswith("student:"):
                text, reason = "student" + name.split(":")[1], "stop"
            elif name.startswith("padding:"):
                text, reason = "THIS PADDING MUST NEVER TRAIN OR VOTE", "stop"
            else:
                text, reason = self.teacher_response(name)
            values.append([ord(char) for char in text])
            reasons.append(reason)
        responses = torch.zeros((len(batch), max(map(len, values))), dtype=torch.long)
        for i, tokens in enumerate(values):
            responses[i, : len(tokens)] = torch.tensor(tokens)
        # Reorder ALL rows, including padding, to exercise identity recovery.
        mask = torch.cat([batch.batch["attention_mask"], (responses != 0).long()], dim=1)
        result = DataProto.from_dict(
            tensors={"responses": responses, "attention_mask": mask, "rollout_log_probs": responses.float() / 1000},
            non_tensors={"self_request_id": np.asarray(names, dtype=object), "self_finish_reason": np.asarray(reasons, dtype=object)},
        )
        return result.select_idxs(list(reversed(range(len(names)))))


class Envs:
    def install_self_teacher_supervision(self, **kwargs):
        self.installed = kwargs
        return ["prepared"]


def generate(helper, actor, envs):
    _, prepared, overflow = helper.prepare([request()])
    assert not overflow
    public = prepared[0]["public_prompt_ids"]
    # Deliberately pad the student training prompt wider than its raw prompt.
    student = inference_batch([public] * 4, ["original"] * 4, helper.tokenizer, {})
    for field in ("input_ids", "attention_mask", "position_ids"):
        student.batch[field] = torch.cat([torch.zeros((4, 17), dtype=torch.long), student.batch[field]], dim=1)
    student.non_tensor_batch.pop("self_request_id")
    output, pending, _ = helper.generate(student, actor, prepared, envs, [0], [PUBLIC])
    assert pending == ["prepared"]
    assert len(output) == 4
    assert not output.non_tensor_batch  # teacher metadata cannot enter loss
    assert torch.equal(output.batch["prompts"], student.batch["input_ids"])
    assert output.batch["input_ids"].shape[1] == student.batch["input_ids"].shape[1] + output.batch["responses"].shape[1]
    assert torch.equal(output.batch["rollout_log_probs"], output.batch["responses"].float() / 1000)
    assert helper.tokenizer.batch_decode(output.batch["responses"]) == [f"student{i}" for i in range(4)]
    assert all("SECRET" not in text for text in helper.tokenizer.batch_decode(output.batch["prompts"]))
    return envs.installed["samples"][0]


def test_teacher_only_privilege_and_common_guidance():
    original = copy.deepcopy(PUBLIC)
    payload = {"customer": {"known": ["SECRET"]}}
    s1 = build_self_teacher_messages(PUBLIC)
    s2 = build_self_teacher_messages(PUBLIC, payload)
    assert PUBLIC == original
    assert s1[1:] == s2[1:] == PUBLIC[1:]
    assert s2[0]["content"].startswith(s1[0]["content"])
    assert "db" not in payload and "user_scenario" not in payload and "evaluation_criteria" not in payload
    assert payload["customer"]["known"] == ["SECRET"]
    assert set(payload) == {"customer"}
    assert "one available tool call" in s1[0]["content"]
    assert "do not impersonate them" in s1[0]["content"]


@pytest.mark.parametrize("protocol", ["tau-self-aopd-v1", "tau-self-aopd-v2", "tau-self-aopd-customer-info-v3"])
def test_old_self_teacher_protocol_cannot_be_resumed(tmp_path, protocol):
    cfg = config(tmp_path)
    SelfTeacherRollout(cfg, Tokenizer())
    path = tmp_path / "self_teacher_manifest.json"
    old = json.loads(path.read_text())
    old["settings"]["protocol"] = protocol
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="resume protocol changed"):
        SelfTeacherRollout(cfg, Tokenizer())


def test_aborted_teacher_output_is_not_a_valid_vote():
    assert parse_teacher_vote(CALL, "abort", TOOLS).kind == "invalid"


@pytest.mark.parametrize(
    "text,reason,kind",
    [
        ("<think>private</think>" + CALL + CALL, "stop", "tool"),
        ("private</think>" + CALL + "<tool_call>", "length", "tool"),
        ("<think>unfinished", "length", "invalid"),
        ("unfinished reasoning", "length", "invalid"),
        ("</think>message", "stop", "message"),
        ('</think><tool_call>{"name":"lookup","arguments":{"id":5}}</tool_call>', "stop", "invalid"),
        ('</think><tool_call>{"name":"missing","arguments":{}}</tool_call>', "stop", "invalid"),
    ],
)
def test_teacher_parsing_first_call_and_strict_schema(text, reason, kind):
    assert parse_teacher_vote(text, reason, TOOLS).kind == kind
    assert parse_action(CALL + CALL).kind == "invalid"  # student unchanged


def test_joint_generation_isolated_loss_rows_cache_and_snapshot(tmp_path):
    helper = SelfTeacherRollout(config(tmp_path), Tokenizer())
    helper.begin_step(1)
    actor, envs = Actor(), Envs()
    votes = generate(helper, actor, envs)
    assert len(votes) == 3 and votes[0] == votes[1] == votes[2]  # keep duplicates
    assert len(actor.calls) == 1 and len(actor.calls[0]) == 8  # 4 + 3 + pad
    before = helper.revision
    helper.begin_step(1)
    assert helper.revision == before
    generate(helper, actor, envs)
    assert len(actor.calls[-1]) == 4  # student always resamples, teacher memo hits
    assert helper.stats["cache_hits"] == 1
    helper.begin_step(2)
    assert before != helper.revision and not helper.cache
    generate(helper, actor, envs)
    assert len(actor.calls[-1]) == 8
    records = [json.loads(line) for line in helper.path.read_text().splitlines()]
    assert len(records) == 2 and records[0]["revision"] != records[1]["revision"]
    restored = SelfTeacherRollout(config(tmp_path), Tokenizer())
    restored.begin_step(2)
    assert not restored.cache and restored.revision != helper.revision


@pytest.mark.parametrize("valid", [0, 1, 2, 3])
def test_partial_votes_retry_only_missing_slots(tmp_path, valid):
    helper = SelfTeacherRollout(config(tmp_path), Tokenizer())
    actor = Actor(lambda slot, attempt: ("</think>" + CALL, "stop") if slot < valid else ("reasoning", "length"))
    votes = generate(helper, actor, Envs())
    assert len(votes) == valid
    assert helper.stats["validity_retries"] == 2 * (3 - valid)
    assert len(actor.calls) == (1 if valid == 3 else 3)
    for retry in actor.calls[1:]:
        assert all(name.startswith("padding:") or (name.startswith("teacher:") and int(name.split(":")[-2]) >= valid) for name in retry)


def test_multiple_states_variable_prompts_reordered_padding_and_partial_failures(tmp_path):
    helper = SelfTeacherRollout(config(tmp_path), Tokenizer())
    requests = []
    for i in range(3):
        value = request()
        value["public_chat"][1]["content"] += " more context" * i
        value["messages"] = build_self_teacher_messages(value["public_chat"], {"user_scenario": f"SECRET-{i}"})
        value["state_fingerprint"] = str(i)
        requests.append(value)
    _, prepared, failures = helper.prepare(requests)
    assert not failures
    state_ids = {req["cache_key"]: i for i, req in enumerate(prepared)}

    class MultiStateActor(Actor):
        world_size = 8

        def teacher_response(self, name):
            _, key, slot, _ = name.split(":")
            state = state_ids[key]
            if state == 0 or (state == 1 and int(slot) > 0):
                return "unfinished reasoning", "length"
            return '</think><tool_call>{"name":"lookup","arguments":{"id":"' + str(state) + '"}}</tool_call>', "stop"

    student = inference_batch([req["public_prompt_ids"] for req in prepared for _ in range(4)], ["original"] * 12, helper.tokenizer, {})
    student.non_tensor_batch.pop("self_request_id")
    envs, actor = Envs(), MultiStateActor()
    output, _, _ = helper.generate(student, actor, prepared, envs, [2, 4, 7], [req["public_chat"] for req in prepared])
    assert envs.installed["active_indices"] == [2, 4, 7]
    assert [len(votes) for votes in envs.installed["samples"]] == [0, 1, 3]
    for state, votes in enumerate(envs.installed["samples"]):
        assert all(vote["arguments"] == {"id": str(state)} for vote in votes)
    assert helper.stats["samples"] == 19  # no padding votes
    assert [len(call) for call in actor.calls] == [24, 8, 8]
    assert helper.tokenizer.batch_decode(output.batch["responses"]) == [f"student{i}" for i in range(12)]
    assert torch.equal(output.batch["prompts"], student.batch["input_ids"])
    assert torch.equal(output.batch["attention_mask"][:, : student.batch["input_ids"].shape[1]], student.batch["attention_mask"])
    assert torch.equal(output.batch["position_ids"][:, : student.batch["input_ids"].shape[1]], student.batch["position_ids"])
    assert torch.equal(output.batch["rollout_log_probs"], output.batch["responses"].float() / 1000)
    assert not output.non_tensor_batch


@pytest.mark.parametrize("violation", ["missing", "duplicate"])
def test_generation_identity_errors_fail_before_installing_votes(tmp_path, violation):
    class BadActor(Actor):
        def generate_sequences(self, batch):
            output = super().generate_sequences(batch)
            if violation == "missing":
                return output.select_idxs(list(range(len(output) - 1)))
            output.non_tensor_batch["self_request_id"][0] = output.non_tensor_batch["self_request_id"][1]
            return output

    envs = Envs()
    with pytest.raises(RuntimeError, match="request identities"):
        generate(SelfTeacherRollout(config(tmp_path), Tokenizer()), BadActor(), envs)
    assert not hasattr(envs, "installed")


@pytest.mark.parametrize("field,value,message", [("mode", "async", "synchronous vLLM"), ("name", "hf", "synchronous vLLM"), ("response_length", 8192, "actual rollout response_length"), ("do_sample", False, "stochastic"), ("temperature", 0, "stochastic")])
def test_reject_unsupported_inference_configuration(tmp_path, field, value, message):
    cfg = config(tmp_path)
    cfg.actor_rollout_ref.rollout[field] = value
    with pytest.raises(ValueError, match=message):
        SelfTeacherRollout(cfg, Tokenizer())
    assert not (tmp_path / "self_teacher_manifest.json").exists()


def test_validity_retry_can_recover_without_changing_student_samples(tmp_path):
    helper = SelfTeacherRollout(config(tmp_path), Tokenizer())
    actor = Actor(lambda slot, attempt: ("</think>" + CALL, "stop") if attempt == 2 else ("", "length"))
    assert len(generate(helper, actor, Envs())) == 3
    assert len({name for name in actor.calls[0] if name.startswith("student:")}) == 4
    assert all(not name.startswith("student:") for call in actor.calls[1:] for name in call)


def test_teacher_overflow_does_not_trim_public_history(tmp_path):
    cfg = config(tmp_path)
    cfg.env.tau.oracle.self_extra_prompt_tokens = 1
    helper = SelfTeacherRollout(cfg, Tokenizer())
    value = request()
    before = copy.deepcopy(value)
    ready, prepared, failures = helper.prepare([value])
    assert not len(ready) and not prepared and len(failures) == 1
    assert failures[0][1]["context_overflow_component"] == "self_teacher_extra_context"
    assert value == before


def test_protocol_guard_and_no_persistent_vote_import(tmp_path):
    cfg = config(tmp_path)
    SelfTeacherRollout(cfg, Tokenizer())
    cfg.env.tau.oracle.use_privileged_context = False
    with pytest.raises(ValueError, match="resume protocol changed"):
        SelfTeacherRollout(cfg, Tokenizer())
    cfg.env.tau.oracle.teacher_cache_import_paths = ["other.jsonl"]
    with pytest.raises(ValueError, match="cannot import"):
        SelfTeacherRollout(cfg, Tokenizer())


def test_real_qwen_template_preserves_the_complete_selected_public_history(tmp_path):
    from pathlib import Path

    from transformers import AutoTokenizer

    from agent_system.multi_turn_rollout.rollout_loop import _render_tau_prompt_with_budget

    model = Path("/mnt/public2/yuanhuining/models/Qwen3-4B")
    if not model.is_dir():
        pytest.skip("requires the local Qwen3-4B tokenizer, not model weights")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cfg = config(tmp_path)
    helper = SelfTeacherRollout(cfg, tokenizer)
    chat = copy.deepcopy(PUBLIC)
    for i in range(12):
        chat += [{"role": "assistant", "tool_calls": [{"id": f"call-{i}", "type": "function", "function": {"name": "lookup", "arguments": {"id": str(i)}}}]}, {"role": "tool", "tool_call_id": f"call-{i}", "content": "long public observation " * 1000}]
    _, visible = _render_tau_prompt_with_budget(tokenizer, chat, {"enable_thinking": True}, tools=TOOLS, max_prompt_tokens=24576)
    assert len(visible) < len(chat)  # exercise real public-history budget
    value = request()
    value["public_chat"] = visible
    value["messages"] = build_self_teacher_messages(visible, {"user_scenario": "HIDDEN"})
    ready, rows, failures = helper.prepare([value])
    assert ready.tolist() == [0] and not failures
    assert value["messages"][1:] == visible[1:]
    assert len(rows[0]["public_prompt_ids"]) <= 24576
    assert len(rows[0]["prompt_ids"]) + 4096 <= 32768


def test_self_client_only_uses_fixed_matcher_and_never_teacher_api(tmp_path, monkeypatch):
    monkeypatch.delenv("SELF_NO_KEY", raising=False)
    monkeypatch.setenv("FIXED_MATCHER_KEY", "test-key")
    client = TauTeacherClient(model="self-rollout-policy", api_base="self://rollout", api_key_env="SELF_NO_KEY", teacher_source="self", cache_path=str(tmp_path / "teacher.jsonl"), matcher_model="matcher", matcher_api_base="http://matcher/v1", matcher_api_key_env="FIXED_MATCHER_KEY")
    assert client.cache_path is None
    with pytest.raises(RuntimeError, match="current rollout weights"):
        client.sample_multiset(state_fingerprint="state", messages=PUBLIC, tools=TOOLS)
    assert not (tmp_path / "teacher.jsonl").exists()


def test_programmatic_only_self_client_requires_neither_teacher_nor_matcher_key(tmp_path, monkeypatch):
    monkeypatch.delenv("SELF_NO_KEY", raising=False)
    client = TauTeacherClient(model="self-rollout-policy", api_base="self://rollout", api_key_env="SELF_NO_KEY", teacher_source="self", matcher_enabled=False, cache_path=str(tmp_path / "teacher.jsonl"))
    assert client.cache_path is None and client.matcher_cache_path is None
    with pytest.raises(RuntimeError, match="matcher is disabled"):
        client._post_matcher({})
    with pytest.raises(RuntimeError, match="current rollout weights"):
        client.sample_multiset(state_fingerprint="state", messages=PUBLIC, tools=TOOLS)


@pytest.mark.parametrize("masked", [True, False])
def test_self_launcher_matcher_identity_requirement(masked):
    script = Path(__file__).resolve().parents[2] / "examples/tau_bench/train/run.sh"
    # Only source the configuration header; never probe endpoints or launch jobs.
    header = script.read_text().split('TRAIN_STEPS="', 1)[0]
    result = subprocess.run(["bash", "-c", header], env={"PATH": "/usr/bin:/bin", "TAU_TEACHER_SOURCE": "self", "TAU_MASK_MATCHER_REQUIRED_GROUPS": str(masked).lower()}, text=True, capture_output=True)
    assert (result.returncode == 0) == masked
    if not masked:
        assert "requires an explicit fixed matcher model" in result.stderr


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_self_preflight_partial_fixed_denominator_and_zero_vote_mask(count):
    from test_protocol import _tau_preflight_worker

    worker = _tau_preflight_worker(lambda **kw: pytest.fail("no external teacher API"))
    worker.teacher_source = "self"
    worker._student_chat = lambda: copy.deepcopy(PUBLIC)
    diagnostics = {"self_teacher_revision": "step-1"}
    ready, info = asyncio.run(worker.prepare_teacher_supervision(PUBLIC, supplied_samples=[{"kind": "message", "content": "hello"}] * count, self_diagnostics=diagnostics))
    assert ready == (count > 0)
    assert info["teacher_sample_count"] == 3
    assert info["teacher_valid_sample_count"] == count
    assert not info["semantic_train_mask"]  # only scored candidate rows train
    if not count:
        assert worker._done and info["teacher_failure"]
    else:
        assert not worker._done and info["self_teacher_revision"] == "step-1"


@pytest.mark.parametrize("smoke", [False, True])
def test_paired_self_plan_has_no_teacher_service_and_preserves_budgets(tmp_path, smoke):
    args = SimpleNamespace(pair="self-s1-s2", run_dir=str(tmp_path), steps=100, shared_port=8180, teacher_port=8181, cpus_per_run=48, smoke=smoke)
    env = {"MODEL_PATH": "/models/student", "TAU_SHARED_MODEL_PATH": "/models/q38", "TAU2_ROOT": "/repos/tau"}
    plan = build_plan(args, env)
    assert len(plan["services"]) == 1 and plan["services"][0]["gpus"] == ["0", "1"]
    for name, gpus in (("s1", "2,3"), ("s2", "4,5")):
        settings = plan["experiments"][name]
        assert settings["CUDA_VISIBLE_DEVICES"] == gpus
        assert settings["TAU_TEACHER_SOURCE"] == "self"
        assert settings["TAU_USE_PRIVILEGED_TEACHER_CONTEXT"] == ("true" if name == "s2" else "false")
        assert settings["MAX_PROMPT"] == "24576" and settings["MAX_MODEL_LEN"] == "32768"
        assert settings["TAU_TEACHER_MAX_TOKENS"] == settings["MAX_RESPONSE"] == "4096"
        assert settings["TAU_MATCHER_PROFILE"] == "qwen38_concise"
        assert settings["TAU_USER_REASONING_ENABLED"] == "false"
        assert settings["SMOKE_TRAIN_STEPS"] == "2" and settings["SMOKE_SAVE_FREQ"] == "1"
    env["TAU_TEACHER_CACHE_IMPORT_PATHS"] = '["old.jsonl"]'
    with pytest.raises(ValueError, match="cannot import"):
        build_plan(args, env)


@pytest.mark.parametrize("fail_second", [False, True])
def test_real_collector_keeps_only_student_rows_and_preserves_prior_groups(tmp_path, fail_second):
    from agent_system.environments.env_package.tau_bench.manager import TauBenchEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    class Environment(Envs):
        decisions = 0

        def reset(self, **kw):
            return {"text": ["state"]}, [{"tau_domain": "airline"}]

        def describe_self_teacher_states(self, **kw):
            # Exact visible state changes across turns, so this cannot hit memo.
            value = request()
            value["state_fingerprint"] = f"state-{self.decisions}"
            return [value]

        def start_teacher_preflight(self, **kw):
            pytest.fail("self rollout must never start external teacher preflight")

        def finish_teacher_preflight(self, pending):
            samples = self.installed["samples"][0]
            return [(bool(samples), {"tau_domain": "airline", "teacher_failure": not samples, "action_kind": "teacher_failure", "semantic_train_mask": False})]

        def state_group_step(self, groups, **kwargs):
            assert len(groups) == 1 and groups[0] == [f"student{i}" for i in range(4)]
            self.decisions += 1
            info = {"tau_domain": "airline", "vpr_game": "tau_airline", "action_kind": "message", "semantic_train_mask": True, "runtime_train_mask": True, "state_group_advanced": True, **self.installed["diagnostics"][0]}
            rows = [("next", float(rank), False, info) for rank in range(4)]
            return [rows], [3], {"text": ["next"]}, np.asarray([3.0]), np.asarray([False]), [info]

        success_evaluator = TauBenchEnvironmentManager(None, None, None).success_evaluator

    env = Environment()
    actor = Actor(lambda slot, attempt: ("", "length") if fail_second and env.decisions else ("</think>" + CALL, "stop"))
    collector = object.__new__(TrajectoryCollector)
    collector.config = config(tmp_path)
    collector.config.env.env_name = "tau_agentic_opd"
    collector.config.env.max_steps = 2
    collector.config.env.tau.train_max_steps = 2
    collector.config.env.tau.oracle.source = "self"
    collector.config.env.rollout.current_step = 1
    collector.tokenizer = Tokenizer()
    collector.preprocess_teacher_preflight_states = lambda **kwargs: (np.asarray([0]), [copy.deepcopy(PUBLIC)], [])

    def preprocess(gen_batch, obs):
        text = collector.tokenizer.apply_chat_template(PUBLIC, tools=TOOLS)
        tokens = collector.tokenizer.encode(text)
        student = inference_batch([tokens] * len(gen_batch), ["original"] * len(gen_batch), collector.tokenizer, {})
        student.non_tensor_batch.pop("self_request_id")
        student.non_tensor_batch["teacher_visible_chat"] = np.asarray([json.dumps(PUBLIC)] * len(gen_batch), dtype=object)
        return student

    collector.preprocess_batch = preprocess
    episodes, rewards, lengths, _, _, _ = collector._state_group_multi_turn_loop_once(DataProto.from_dict(tensors={"input_ids": torch.ones((1, 1))}), actor, env)
    expected = 1 if fail_second else 2
    assert env.decisions == expected and len(episodes[0]) == expected * 4
    assert lengths.tolist() == [expected] and rewards.tolist() == [3 * expected]
    assert len({row["self_teacher_revision"] for row in episodes[0]}) == 1
    assert all(row["semantic_train_mask"] and "self_request_id" not in row for row in episodes[0])
    assert "self_joint_generation" in collector._last_state_group_timing
    assert "teacher_hidden_by_student_work" not in collector._last_state_group_timing
