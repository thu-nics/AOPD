"""Paired public/answer-conditioned diagnostic, not a leaderboard evaluation.

Uses native Tau construction, execution and evaluation, and the actual
Self-AOPD answer projection. No student candidates or matcher.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType

import httpx
import litellm
from loguru import logger
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.data_model.simulation import TerminationReason, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Role
from tau2.registry import registry
from tau2.runner import get_tasks
from tau2.runner.build import build_text_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.utils.llm_utils import to_litellm_messages
from transformers import AutoTokenizer

from agent_system.environments.env_package.tau_bench.customer_briefs import digest, load_briefs
from agent_system.environments.env_package.tau_bench.self_teacher import (
    PROTOCOL,
    TEACHER_INSTRUCTION,
    audit_task_budgets,
    build_self_teacher_messages,
    parse_teacher_vote,
)
from agent_system.environments.env_package.tau_bench.self_teacher_answers import ANSWER_PROTOCOL, reference_source_fingerprint
from agent_system.environments.env_package.tau_bench.self_teacher_context import context_projection_fingerprint
from agent_system.environments.env_package.tau_bench.self_teacher_privilege import privileged_context
from agent_system.environments.env_package.tau_bench.user_simulator import USER_NAME, register_validated_user_simulator
from agent_system.environments.prompts.agentic_opd import tau_system_prompt
from agent_system.multi_turn_rollout.rollout_loop import _render_tau_prompt_with_budget
from examples.tau_bench.eval.deterministic_evaluator import EVALUATION_PROTOCOL, install_deterministic_evaluator

OUT = None
MODEL = None
BRIEFS = None
MODEL_NAME = None
MODES = ("public", "answer_conditioned")
DOMAINS = ("airline", "retail", "telecom")
TOKENIZER = None
CLIENT = None
TOKENIZER_LOCK = threading.Lock()
MAX_DECISIONS = 20


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class InvalidTeacherAction(RuntimeError):
    """Policy failure after validity retries, not an API failure."""


class DiagnosticAgent(LLMAgent):
    def __init__(self, *, task, mode, domain, seed, **kwargs):
        super().__init__(**kwargs)
        # Only the answer arm receives reference fields, through the allowlist
        # projection shared with training; never serialize the whole task.
        self.customer_task = {"id": str(task.id), "user_scenario": task.user_scenario.model_dump(mode="json")}
        if mode == "answer_conditioned":
            self.customer_task["evaluation_criteria"] = task.evaluation_criteria.model_dump(mode="json") if task.evaluation_criteria else None
        self.mode = mode
        self.domain = domain
        self.task_id = str(task.id)
        self.seed = seed
        self.decisions = 0
        self.audit_path = OUT / "requests" / mode / domain / f"{digest(str(task.id))[:20]}-{uuid.uuid4().hex}.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def system_prompt(self):
        return tau_system_prompt(self.domain_policy)

    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        public = to_litellm_messages(state.system_messages + state.messages)
        tools = [tool.openai_schema for tool in self.tools]
        with TOKENIZER_LOCK:
            public_prompt, visible = _render_tau_prompt_with_budget(TOKENIZER, public, {"enable_thinking": True}, tools=tools, max_prompt_tokens=24576)
            private = privileged_context(self.domain, self.customer_task, visible, briefs_path=BRIEFS, mode=self.mode, tools=tools) if self.mode != "public" else None
            messages = build_self_teacher_messages(visible, private)
            assert messages[1:] == visible[1:], "Teacher projection changed public history"
            prompt = TOKENIZER.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=True)
            public_tokens = len(TOKENIZER.encode(public_prompt, add_special_tokens=False))
        if len(prompt) > 28672 or len(prompt) - public_tokens > 4096:
            raise RuntimeError("Teacher context overflow: refusing to discard extra history")
        self.decisions += 1
        started = time.monotonic()
        attempts = []
        for attempt in range(3):
            payload = dict(
                model=MODEL_NAME,
                prompt=prompt,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=4096,
                n=1,
                seed=self.seed + self.decisions * 31 + attempt,
            )
            response = CLIENT.post("/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            action = parse_teacher_vote(choice["text"], choice["finish_reason"], tools)
            attempts.append(dict(raw_response=choice["text"], finish_reason=choice["finish_reason"], action=action.to_dict(), usage=data.get("usage")))
            if action.kind != "invalid":
                break
        audit = dict(
            mode=self.mode,
            domain=self.domain,
            task_id=self.task_id,
            decision=self.decisions,
            public_prompt_tokens=public_tokens,
            prompt_tokens=len(prompt),
            public_history_messages=len(public) - 1,
            retained_history_messages=len(visible) - 1,
            messages=messages,
            tools=tools,
            attempts=attempts,
            elapsed_s=time.monotonic() - started,
        )
        with self.audit_path.open("a") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
        if action.kind == "invalid":
            raise InvalidTeacherAction(action.error)
        tool_calls = [ToolCall(id="call_" + uuid.uuid4().hex, name=action.name, arguments=action.arguments)] if action.kind == "tool" else None
        return AssistantMessage(role="assistant", content=None if tool_calls else action.content, tool_calls=tool_calls, cost=0, usage=data.get("usage"), raw_data=data, generation_time_seconds=time.monotonic() - started)


def factory(tools, domain_policy, task, llm, llm_args, **_):
    settings = dict(llm_args)
    return DiagnosticAgent(tools=tools, domain_policy=domain_policy, task=task, llm=llm, llm_args={}, mode=settings["mode"], domain=settings["domain"], seed=settings["seed"])


def run_one(domain, task, mode, trial):
    seed = 200920 + int(hashlib.sha256(f"{domain}:{task.id}:{trial}".encode()).hexdigest()[:7], 16)
    output = OUT / "results" / mode / domain / f"{digest(str(task.id))[:20]}-trial{trial}.json"
    if output.exists():
        old = json.loads(output.read_text())
        if old["status"] == "completed":
            return old
    user_args = dict(
        api_base=os.environ["TAU_USER_API_BASE"],
        api_key=os.environ["TAU_USER_API_KEY"],
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=8192,
        num_retries=2,
        timeout=300,
        _validation_retries=2,
        extra_body=dict(top_k=20, min_p=0.0, repetition_penalty=1.0, chat_template_kwargs={"enable_thinking": True}),
    )
    config = TextRunConfig(
        domain=domain,
        task_set_name=domain,
        task_split_name="train",
        agent="self_teacher_diagnostic",
        llm_agent="openai/self-teacher-diagnostic",
        llm_args_agent=dict(mode=mode, domain=domain, seed=seed),
        user=USER_NAME,
        llm_user=os.environ["TAU_USER_MODEL"],
        llm_args_user=user_args,
        max_steps=200,
        max_errors=10,
        timeout=3600,
        seed=seed,
        num_trials=1,
        enforce_communication_protocol=False,
    )
    failures = []
    for infrastructure_attempt in range(3):
        started = time.monotonic()
        result = dict(mode=mode, domain=domain, task_id=str(task.id), seed=seed, trial=trial)
        try:
            orch = build_text_orchestrator(config, task, seed=seed)
            native_check = orch._check_termination
            native_step = orch.step

            def check(_, orch=orch, native_check=native_check):
                native_check()
                if not orch.done and orch.to_role == Role.AGENT and orch.agent.decisions >= MAX_DECISIONS:
                    orch.done = True
                    orch.termination_reason = TerminationReason.MAX_STEPS

            def step(_, orch=orch, native_step=native_step):
                try:
                    native_step()
                except InvalidTeacherAction:
                    orch.done = True
                    orch.termination_reason = TerminationReason.AGENT_ERROR

            orch._check_termination = MethodType(check, orch)
            orch.step = MethodType(step, orch)
            simulation = run_simulation(orch, evaluation_type=EvaluationType.ALL)
            audit_rows = [json.loads(line) for line in orch.agent.audit_path.read_text().splitlines()] if orch.agent.audit_path.exists() else []
            attempts = [attempt for row in audit_rows for attempt in row["attempts"]]
            result.update(
                invalid_attempts=sum(a["action"]["kind"] == "invalid" for a in attempts),
                clipped_attempts=sum(a["finish_reason"] == "length" for a in attempts),
                generation_attempts=len(attempts),
                transferred=any(row["attempts"][-1]["action"].get("name") == "transfer_to_human_agents" for row in audit_rows),
                max_prompt_tokens=max((row["prompt_tokens"] for row in audit_rows), default=0),
                status="completed",
                success=bool(simulation.reward_info.reward >= 1.0),
                reward=float(simulation.reward_info.reward),
                decisions=orch.agent.decisions,
                internal_steps=orch.step_count,
                termination_reason=simulation.termination_reason.value,
                simulation=simulation.model_dump(mode="json"),
                requests=str(orch.agent.audit_path),
            )
        except Exception as exc:
            failures.append(dict(type=type(exc).__name__, message=str(exc), attempt=infrastructure_attempt))
            result.update(status="infrastructure_pending", success=None)
        result.update(infrastructure_failures=failures, elapsed_s=time.monotonic() - started)
        save(output, result)
        if result["status"] == "completed":
            return result
        print("RETRY", mode, domain, task.id, failures[-1]["type"], failures[-1]["message"][:240], flush=True)
    return result


def pilot_ids(tasks):
    forced = {"airline": ["0", "5", "7", "14", "28", "34", "38", "49"], "retail": ["0", "13", "15", "24", "31", "57", "72", "99"], "telecom": ["[service_issue]airplane_mode_on|break_apn_settings|unseat_sim_card[PERSONA:None]"]}
    result = {}
    for domain, rows in tasks.items():
        available = {str(t.id) for t in rows}
        selected = [x for x in forced[domain] if x in available]
        selected.extend(x for x in sorted(available, key=digest) if x not in selected)
        result[domain] = selected[:8]
    return result


def summarize(tasks, *, bootstrap=False):
    import numpy as np

    results = [json.loads(p.read_text()) for p in (OUT / "results").glob("*/*/*.json")]
    pilot = pilot_ids(tasks)
    report = {"expected": sum(map(len, tasks.values())) * 3 * len(MODES), "completed": sum(r["status"] == "completed" for r in results), "infrastructure_pending": sum(r["status"] != "completed" for r in results), "groups": {}, "paired": {}}
    for population in ("all", "development24", "heldout154"):
        subset = [r for r in results if population == "all" or ((r["task_id"] in pilot[r["domain"]]) == (population == "development24"))]
        for mode in MODES:
            rates, total, success = [], 0, 0
            for domain in DOMAINS:
                rows = [r for r in subset if r["mode"] == mode and r["domain"] == domain and r["status"] == "completed"]
                n, wins = len(rows), sum(r["success"] for r in rows)
                rate = wins / n if n else None
                report["groups"][f"{population}/{mode}/{domain}"] = dict(
                    completed=n,
                    successes=wins,
                    success_rate=rate,
                    mean_decisions=sum(r["decisions"] for r in rows) / n if n else None,
                    internal_cap=sum(r.get("internal_steps", 0) >= 200 for r in rows),
                    invalid_attempts=sum(r.get("invalid_attempts", 0) for r in rows),
                    clipped_attempts=sum(r.get("clipped_attempts", 0) for r in rows),
                    generation_attempts=sum(r.get("generation_attempts", 0) for r in rows),
                    max_prompt_tokens=max((r.get("max_prompt_tokens", 0) for r in rows), default=0),
                )
                rates.append(rate)
                total += n
                success += wins
            report["groups"][f"{population}/{mode}/aggregate"] = dict(macro_success_rate=sum(rates) / 3 if all(r is not None for r in rates) else None, micro_success_rate=success / total if total else None, completed=total)
        indexed = {(r["mode"], r["domain"], r["task_id"], r["trial"]): r for r in subset if r["status"] == "completed"}
        for mode in MODES[1:]:
            by_domain = {d: collections.defaultdict(list) for d in DOMAINS}
            pairs = collections.Counter()
            for (m, d, t, trial), public in indexed.items():
                other = indexed.get((mode, d, t, trial))
                if m == "public" and other:
                    delta = int(other["success"]) - int(public["success"])
                    by_domain[d][t].append(delta)
                    pairs[f"public_{int(public['success'])}_private_{int(other['success'])}"] += 1
            value = {"counts": dict(pairs)}
            if all(by_domain[d] for d in DOMAINS):
                clusters = [np.array([np.mean(v) for v in by_domain[d].values()]) for d in DOMAINS]
                value["paired_macro_net_gain"] = float(np.mean([v.mean() for v in clusters]))
                if bootstrap:
                    rng = np.random.default_rng(920)
                    draws = np.mean([rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1) for v in clusters], axis=0)
                    value["task_cluster_bootstrap_95ci"] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
            report["paired"][f"{population}/{mode}"] = value
    save(OUT / "summary.json", report)
    return report


def main():
    global CLIENT, TOKENIZER, OUT, MODEL, BRIEFS, MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--briefs-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8112/v1")
    parser.add_argument("--served-model-name", default="self-teacher-diagnostic")
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    OUT = Path(args.output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL, BRIEFS, MODEL_NAME = args.model_path, str(Path(args.briefs_path).resolve()), args.served_model_name
    import fcntl

    run_lock = (OUT / "diagnostic.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from agent_system.environments.env_package.tau_bench.envs import validate_tau_source

    source_identity = validate_tau_source(args.source_root)
    os.environ["TAU2_DATA_DIR"] = str(Path(args.source_root).resolve() / "data")
    store = load_briefs(BRIEFS)
    logger.remove()
    logger.add(OUT / "native.log", level="WARNING")
    install_deterministic_evaluator()
    register_validated_user_simulator()
    registry.register_agent_factory(factory, "self_teacher_diagnostic")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tasks = {d: get_tasks(d, task_split_name="train") for d in DOMAINS}
    context_audit = audit_task_budgets(args.source_root, MODEL, briefs_path=BRIEFS, privilege_mode="answer_conditioned")
    manifest = dict(
        protocol="paired-direct-answer-conditioned-self-teacher-v1",
        conditions=list(MODES),
        answer_protocol=ANSWER_PROTOCOL,
        train_reference_sha256=reference_source_fingerprint(args.source_root),
        context_audit=context_audit,
        customer_briefs_sha256=store.fingerprint,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        teacher_protocol=PROTOCOL,
        teacher_instruction_sha256=digest(TEACHER_INSTRUCTION),
        source_identity=source_identity,
        tasks_sha256=digest({d: [t.model_dump(mode="json") for t in ts] for d, ts in tasks.items()}),
        projection_hash=context_projection_fingerprint(),
        evaluation_protocol=EVALUATION_PROTOCOL,
        model=MODEL,
        split="train",
        domains={d: [str(t.id) for t in ts] for d, ts in tasks.items()},
        trials=3,
        max_agent_decisions=MAX_DECISIONS,
        internal_max_steps=200,
        teacher_sampling=dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0, enable_thinking=True, max_tokens=4096, validity_retries=2),
        public_prompt_budget=24576,
        extra_teacher_budget=4096,
        max_model_len=32768,
        user_model=os.environ["TAU_USER_MODEL"],
        user_endpoint=os.environ["TAU_USER_API_BASE"],
        user_sampling=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0, enable_thinking=True, max_tokens=8192),
        benchmark_eligible=False,
        best_of_n=False,
    )
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Diagnostic manifest mismatch: use a new output directory")
    save(manifest_path, manifest)
    for domain, domain_tasks in tasks.items():
        for task in domain_tasks:
            store.get(domain, {"id": str(task.id), "user_scenario": task.user_scenario.model_dump(mode="json")})
    print("TASKS", {d: len(ts) for d, ts in tasks.items()}, "total trajectories", sum(map(len, tasks.values())) * 3 * len(MODES), flush=True)
    if args.check:
        return
    CLIENT = httpx.Client(base_url=args.api_base, timeout=600, trust_env=False, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
    for _ in range(180):
        try:
            response = CLIENT.get("/models", timeout=5)
            response.raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(5)
    else:
        raise RuntimeError("Local teacher server did not become ready")
    litellm.client_session = httpx.Client(trust_env=False, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
    if args.phase == "full":
        gate = OUT / "pilot_passed.json"
        if not gate.exists() or json.loads(gate.read_text())["manifest_sha256"] != digest(manifest):
            raise RuntimeError("Full diagnostic requires a passed pilot under exactly this protocol")
    selected = pilot_ids(tasks)
    jobs = [(d, tasks[d][i], mode, trial) for trial in range(3) for i in range(max(map(len, tasks.values()))) for d in DOMAINS if i < len(tasks[d]) for mode in MODES if args.phase == "full" or (trial == 0 and str(tasks[d][i].id) in selected[d])]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            report = summarize(tasks)
            print("DONE", report["completed"], "/", report["expected"], result["mode"], result["domain"], result["task_id"], result["trial"], result["status"], result["success"], flush=True)
    report = summarize(tasks, bootstrap=True)
    if args.phase == "pilot":
        pilot_expected = sum(map(len, selected.values())) * len(MODES)
        if report["infrastructure_pending"] or report["completed"] != pilot_expected:
            raise RuntimeError("Pilot not technically complete; full diagnostic remains gated")
        save(OUT / "pilot_passed.json", dict(manifest_sha256=digest(manifest), selected=selected, completed=pilot_expected))
    elif report["infrastructure_pending"]:
        raise RuntimeError("Some diagnostic trajectories remain pending; resume without counting them as policy failures")
    print("SUMMARY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
