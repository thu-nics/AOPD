"""Snapshot-local Tau action supervision using the rollout policy's own weights.

Privileged prompts are inference-only. They never become policy training rows.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.model import compute_position_id_with_mask

from .actions import ParsedAction, parse_action
from .self_teacher_context import context_projection_fingerprint, teacher_public_policy

PROTOCOL = "tau-self-aopd-answer-conditioned-v6"
TEACHER_INSTRUCTION = (
    "Choose exactly one next action: one available tool call OR one user-facing message. "
    "Assist the customer as a person; do not impersonate them. Device operations are "
    "performed by the customer: explain them in ordinary language, not tool calls. "
    "Follow the public policy, tool preconditions and required user consent. "
    "Private notes, targets and reference steps aid planning; they are not public disclosure, consent "
    "or proof that an action was executed. Customer preferences never override policy. "
    "Obtain missing lookup identifiers through the public conversation or tool results; "
    "do not use hidden lookup values to bypass verification or copy private aliases. Ask specific questions when needed. "
    "Reference steps describe one possible solution, not actions already executed or a mandatory script. "
    "Use the actual history to identify what remains; never repeat a completed step just because it is listed. "
    "If a required confirmation or fact is missing, obtain it before the reference operation. "
    "Use observations to choose a useful next step. Your final action must implement your reasoning."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def self_brief_fingerprint(oracle):
    if not oracle.use_privileged_context:
        return None
    from .customer_briefs import load_briefs

    return load_briefs(str(getattr(oracle, "self_customer_briefs_path", ""))).fingerprint


def build_self_teacher_messages(public_chat, privileged_context=None):
    if not public_chat or public_chat[0].get("role") != "system":
        raise ValueError("self teacher requires a public system message")
    messages = copy.deepcopy(public_chat)
    addition = "\n\nSELF-TEACHER ACTION GUIDANCE:\n" + TEACHER_INSTRUCTION
    if privileged_context is not None:
        addition += "\n\nPRIVATE PLANNING NOTES (NOT EXECUTION EVIDENCE):\n" + json.dumps(privileged_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    messages[0]["content"] = teacher_public_policy(str(messages[0].get("content") or "")) + addition
    return messages


def parse_teacher_vote(text, finish_reason, tools):
    """Mirror the API teacher: first complete call wins; truncated messages fail."""
    from .oracle import TauTeacherClient

    if finish_reason not in {"stop", "length"}:
        return ParsedAction(kind="invalid", error=f"incomplete teacher generation: {finish_reason}")
    # Qwen's generation prompt may already include the opening thinking tag.
    public = text.split("</think>", 1)[-1] if "</think>" in text else text
    if "<think>" in public:
        return ParsedAction(kind="invalid", error="unfinished teacher reasoning")
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", public, re.DOTALL)
    if calls:
        action = parse_action("<tool_call>" + calls[0] + "</tool_call>")
    elif finish_reason == "length":
        return ParsedAction(kind="invalid", error="truncated teacher message")
    else:
        action = parse_action(public)
    return TauTeacherClient._validate_teacher_action(action, tools)


def inference_batch(prompt_ids, request_ids, tokenizer, meta_info):
    """Build temporary inference tensors; prompt length is independent of training."""
    width = max(map(len, prompt_ids))
    ids = torch.full((len(prompt_ids), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(prompt_ids):
        ids[row, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long)
        mask[row, -len(tokens) :] = 1
    raw = np.empty(len(prompt_ids), dtype=object)
    raw[:] = prompt_ids
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask)},
        non_tensors={"raw_prompt_ids": raw, "self_request_id": np.asarray(request_ids, dtype=object)},
        meta_info=dict(meta_info),
    )


def restore_student_output(output, student_batch, student_ids):
    """Remove all auxiliary rows AND extra inference padding before policy loss."""
    positions = {value: i for i, value in enumerate(output.non_tensor_batch["self_request_id"])}
    selected = output.select_idxs([positions[value] for value in student_ids])
    response = selected.batch["responses"]
    response_mask = selected.batch["attention_mask"][:, -response.shape[1] :]
    ids = student_batch.batch["input_ids"]
    position_ids = student_batch.batch["position_ids"]
    delta = torch.arange(1, response.shape[1] + 1, device=position_ids.device).unsqueeze(0)
    return DataProto.from_dict(
        tensors={
            "prompts": ids,
            "responses": response,
            "input_ids": torch.cat([ids, response], dim=-1),
            "attention_mask": torch.cat([student_batch.batch["attention_mask"], response_mask], dim=-1),
            "position_ids": torch.cat([position_ids, position_ids[:, -1:] + delta], dim=-1),
            "rollout_log_probs": selected.batch["rollout_log_probs"],
        }
    )


class SelfTeacherRollout:
    """One collector owns the cache; no serving process or weight copy is needed."""

    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        oracle = config.env.tau.oracle
        rollout = config.actor_rollout_ref.rollout
        if rollout.name != "vllm" or rollout.mode != "sync":
            raise ValueError("self AOPD requires synchronous vLLM with per-request finish reasons")
        if not rollout.do_sample or float(rollout.temperature) <= 0:
            raise ValueError("self AOPD requires stochastic student/teacher generation")
        if int(oracle.samples) != 3 or int(config.env.rollout.n) != 4:
            raise ValueError("self AOPD requires independent N=4 / K=3 samples")
        if int(rollout.n) != 1:
            raise ValueError("self AOPD supplies explicit inference rows; rollout.n must be 1")
        if list(oracle.teacher_cache_import_paths):
            raise ValueError("self teacher cannot import external/snapshot teacher caches")
        for field in ("temperature", "top_p", "top_k", "min_p"):
            if float(getattr(oracle, field)) != float(getattr(rollout, field)):
                raise ValueError(f"shared self generation requires matching student/teacher {field}")
        if int(oracle.max_tokens) != int(config.data.max_response_length):
            raise ValueError("self teacher output budget must match student output budget")
        if int(rollout.response_length) != int(oracle.max_tokens):
            raise ValueError("self teacher output budget must match the actual rollout response_length")
        if bool(oracle.enable_thinking) != bool(config.data.apply_chat_template_kwargs.get("enable_thinking", True)):
            raise ValueError("self teacher and student thinking modes must match")
        self.extra_budget = int(oracle.self_extra_prompt_tokens)
        if self.extra_budget < 1 or int(config.data.max_prompt_length) + self.extra_budget + int(oracle.max_tokens) > int(rollout.max_model_len):
            raise ValueError("self teacher prompt + output exceeds the rollout context budget")
        self.retries = int(oracle.teacher_validity_max_retries)
        if self.retries < 0:
            raise ValueError("teacher validity retries must be non-negative")
        self.path = Path(oracle.cache_path).parent / "self_teacher.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Separate metadata from audit records: audit outputs are never imported.
        manifest = self.path.with_name("self_teacher_manifest.json")
        from omegaconf import OmegaConf

        settings = {
            "protocol": PROTOCOL,
            "template_sha256": digest(TEACHER_INSTRUCTION),
            "context_projection_sha256": context_projection_fingerprint(),
            "privilege_mode": getattr(oracle, "self_privilege_mode", "answer_conditioned") if oracle.use_privileged_context else None,
            "customer_briefs_sha256": self_brief_fingerprint(oracle),
            "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)),
            "model": str(config.actor_rollout_ref.model.path),
            "context_mode": "privileged" if oracle.use_privileged_context else "student_visible",
            "source_root": str(config.env.tau.source_root),
            "extra_budget": self.extra_budget,
            "public_budget": int(config.data.max_prompt_length),
            "output_budget": int(oracle.max_tokens),
            "decoding": {field: getattr(oracle, field) for field in ("temperature", "top_p", "top_k", "min_p", "enable_thinking")},
            "template_kwargs": OmegaConf.to_container(config.data.apply_chat_template_kwargs, resolve=True),
            "validity_retries": self.retries,
        }
        if oracle.use_privileged_context and settings["privilege_mode"] == "answer_conditioned":
            from .self_teacher_answers import ANSWER_PROTOCOL, reference_source_fingerprint

            settings["answer_protocol"] = ANSWER_PROTOCOL
            settings["train_reference_sha256"] = reference_source_fingerprint(config.env.tau.source_root)
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["settings"] != settings:
                raise ValueError("self teacher resume protocol changed; use a new run directory")
            self.lineage = saved["lineage"]
        else:
            self.lineage = uuid.uuid4().hex
            with manifest.open("x") as stream:
                json.dump({"settings": settings, "lineage": self.lineage}, stream, indent=2)
        self.begin_step(-1)

    def begin_step(self, step):
        # Even a resume at the same global step starts with an empty memo. Thus
        # stale audit rows can never be mistaken for current-weight supervision.
        if getattr(self, "step", None) == step:
            return
        self.step = step
        self.revision = f"{self.lineage}:pre-update-{step}:{uuid.uuid4().hex}"
        self.cache = {}
        self.stats = {"states": 0, "cache_hits": 0, "prompt_tokens": 0, "extra_tokens": 0, "samples": 0, "clipped": 0, "validity_retries": 0}

    def prepare(self, requests):
        from agent_system.multi_turn_rollout.rollout_loop import _render_agentic_prompt

        ready, prepared, overflows = [], [], []
        kwargs = dict(self.config.data.apply_chat_template_kwargs)
        for index, request in enumerate(requests):
            public = _render_agentic_prompt(self.tokenizer, request["public_chat"], "chatml", kwargs, tools=request["tools"])
            teacher = _render_agentic_prompt(self.tokenizer, request["messages"], "chatml", kwargs, tools=request["tools"])
            ids = self.tokenizer.encode(teacher, add_special_tokens=False)
            public_ids = self.tokenizer.encode(public, add_special_tokens=False)
            public_len = len(public_ids)
            if public_len > int(self.config.data.max_prompt_length):
                raise RuntimeError("self teacher received a public history not bounded by the student preflight")
            extra = len(ids) - public_len
            limit = int(self.config.data.max_prompt_length) + self.extra_budget
            if extra > self.extra_budget or len(ids) > limit:
                overflows.append((index, {"context_prompt_tokens": len(ids), "context_max_prompt_tokens": limit, "context_excess_tokens": max(extra - self.extra_budget, len(ids) - limit), "context_overflow_component": "self_teacher_extra_context"}))
                continue
            key = digest({"revision": self.revision, "request": request, "prompt_ids": ids})
            prepared.append(dict(request, prompt_ids=ids, public_prompt_ids=public_ids, cache_key=key, extra_tokens=extra))
            ready.append(index)
        return np.asarray(ready, dtype=np.int64), prepared, overflows

    def generate(self, student_batch, actor, requests, envs, active_indices, visible_chats):
        start = time.perf_counter()
        # One generation call holds the same rollout weights for both roles.
        student_ids = [f"student:{i}" for i in range(len(student_batch))]
        ids = [list(row) for row in student_batch.non_tensor_batch["raw_prompt_ids"]]
        if len(ids) != 4 * len(requests):
            raise RuntimeError("self supervision requires four student rows per state")
        for index, request in enumerate(requests):
            if any(tokens != request["public_prompt_ids"] for tokens in ids[4 * index : 4 * index + 4]):
                raise RuntimeError("self teacher and student do not share the same public prompt")
        if student_batch.meta_info.get("validate", False) or not student_batch.meta_info.get("do_sample", True):
            raise ValueError("self AOPD requires stochastic training generation")
        keys = list(student_ids)
        unique = {request["cache_key"]: request for request in requests}
        votes = {key: dict(self.cache.get(key, {})) for key in unique}
        attempts = {key: [] for key in unique}
        first_output = None
        for attempt in range(self.retries + 1):
            lookup = {}
            for key, request in unique.items():
                for slot in range(3):
                    if slot in votes[key]:
                        continue
                    request_id = f"teacher:{key}:{slot}:{attempt}"
                    keys.append(request_id)
                    ids.append(request["prompt_ids"])
                    lookup[request_id] = (key, slot)
            if not keys:
                break
            batch = inference_batch(ids, keys, self.tokenizer, student_batch.meta_info)
            padded, pad_size = pad_dataproto_to_divisor(batch, actor.world_size)
            # Padding generates auxiliary copies, not additional teacher votes.
            # Give them unique identities so even a reordered worker result
            # cannot substitute a padding sample for a real student/teacher row.
            if pad_size:
                padded.non_tensor_batch["self_request_id"][-pad_size:] = [f"padding:{attempt}:{i}" for i in range(pad_size)]
            expected = list(padded.non_tensor_batch["self_request_id"])
            output = actor.generate_sequences(padded)
            returned = list(output.non_tensor_batch["self_request_id"])
            if len(returned) != len(expected) or len(set(returned)) != len(expected) or set(returned) != set(expected):
                raise RuntimeError("self-teacher generation lost/duplicated request identities")
            positions = {value: i for i, value in enumerate(returned)}
            output = output.select_idxs([positions[key] for key in keys])
            returned = keys
            if attempt == 0:
                first_output = restore_student_output(output, student_batch, student_ids)
            teacher_rows = [row for row, request_id in enumerate(returned) if request_id in lookup]
            # Student responses are decoded once by the normal rollout path.
            texts = self.tokenizer.batch_decode(output.batch["responses"][teacher_rows], skip_special_tokens=True)
            for row, text in zip(teacher_rows, texts, strict=True):
                request_id = returned[row]
                key, slot = lookup[request_id]
                reason = output.non_tensor_batch["self_finish_reason"][row]
                action = parse_teacher_vote(text, reason, unique[key]["tools"])
                attempts[key].append({"sample_index": slot, "attempt": attempt, "raw_response": text, "finish_reason": reason, "action": action.to_dict()})
                self.stats["samples"] += 1
                self.stats["clipped"] += int(reason == "length")
                self.stats["validity_retries"] += int(attempt > 0)
                if action.kind != "invalid":
                    votes[key][slot] = action.to_dict()
            ids, keys = [], []
        diagnostics, actions = [], []
        for request in requests:
            key = request["cache_key"]
            self.stats["states"] += 1
            self.stats["cache_hits"] += int(key in self.cache)
            self.stats["prompt_tokens"] += len(request["prompt_ids"])
            self.stats["extra_tokens"] += request["extra_tokens"]
            diagnostics.append({"self_teacher_revision": self.revision, "self_teacher_prompt_tokens": len(request["prompt_ids"]), "self_teacher_extra_tokens": request["extra_tokens"]})
            actions.append([votes[key][slot] for slot in sorted(votes[key])])
        with self.path.open("a") as stream:
            for key, request in unique.items():
                if attempts[key]:
                    record = {k: v for k, v in request.items() if k not in {"prompt_ids", "public_prompt_ids"}}
                    stream.write(json.dumps(dict(record, protocol=PROTOCOL, revision=self.revision, attempts=attempts[key], votes=list(votes[key].values())), ensure_ascii=False) + "\n")
                self.cache[key] = votes[key]
        pending = envs.install_self_teacher_supervision(active_indices=active_indices, visible_chats=visible_chats, samples=actions, diagnostics=diagnostics)
        return first_output, pending, time.perf_counter() - start

    def metrics(self):
        n = max(1, self.stats["states"])
        return {
            "env/self_teacher_prompt_tokens_mean": np.asarray([self.stats["prompt_tokens"] / n]),
            "env/self_teacher_extra_tokens_mean": np.asarray([self.stats["extra_tokens"] / n]),
            "env/self_teacher_cache_hit_rate": np.asarray([self.stats["cache_hits"] / n]),
            "env/self_teacher_clip_rate": np.asarray([self.stats["clipped"] / max(1, self.stats["samples"])]),
            "env/self_teacher_validity_retry_count": np.asarray([self.stats["validity_retries"]]),
        }


def audit_task_budgets(source_root, model_path, extra_budget=4096, *, briefs_path="", privilege_mode="answer_conditioned", use_privileged=True):
    """Fail before GPU allocation if any official train reference cannot fit."""
    import os

    from .envs import validate_tau_source

    validate_tau_source(source_root)
    policy_root = Path(source_root) / "data/tau2/domains/telecom"
    manual = (policy_root / "tech_support_manual.md").read_text()
    main = (policy_root / "main_policy.md").read_text()
    teacher_public_policy("<main_policy>\n" + main + "\n</main_policy><tech_support_policy>\n" + manual + "\n</tech_support_policy>")
    os.environ["TAU2_DATA_DIR"] = str(Path(source_root).resolve() / "data")
    from tau2.runner.build import build_environment
    from tau2.runner.helpers import load_tasks
    from transformers import AutoTokenizer

    from .self_teacher_privilege import privileged_context

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    result = {}
    for domain in ("airline", "retail", "telecom"):
        lengths = []
        reference_unavailable = []
        tools = [tool.openai_schema for tool in build_environment(domain).get_tools()] if privilege_mode == "answer_conditioned" else None
        for task in load_tasks(domain, "train"):
            native_task = task
            task = task.model_dump(mode="json")
            private_context = None
            if use_privileged:
                environment = None
                if privilege_mode == "customer_and_state":
                    environment = build_environment(domain)
                    initial = native_task.initial_state
                    environment.set_state(
                        initialization_data=initial.initialization_data if initial else None,
                        initialization_actions=initial.initialization_actions if initial else None,
                        message_history=(initial.message_history or []) if initial else [],
                    )
                before = (environment.get_db_hash(), environment.get_user_db_hash()) if environment else None
                private_context = privileged_context(domain, task, [], briefs_path=briefs_path, mode=privilege_mode, environment=environment, tools=tools)
                if privilege_mode == "answer_conditioned" and not private_context["reference"]["available"]:
                    reference_unavailable.append(str(task["id"]))
                after = (environment.get_db_hash(), environment.get_user_db_hash()) if environment else None
                if before != after:
                    raise RuntimeError("private state projection mutated the environment")
            chat = [{"role": "system", "content": "Public policy."}, {"role": "user", "content": "Help me."}]
            public = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, enable_thinking=True)
            private = tokenizer.apply_chat_template(build_self_teacher_messages(chat, private_context), tokenize=True, add_generation_prompt=True, enable_thinking=True)
            length = len(private) - len(public)
            if length > extra_budget:
                raise ValueError(f"{domain}:{task['id']} self-teacher extra context {length} > {extra_budget}; no task was filtered")
            lengths.append(length)
        result[domain] = {"tasks": len(lengths), "max_extra_tokens": max(lengths), "mean_extra_tokens": sum(lengths) / len(lengths), "reference_unavailable": reference_unavailable}
    return result


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Audit every Tau train task's self-teacher token overhead (no API/GPU).")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--extra-budget", type=int, default=4096)
    parser.add_argument("--briefs-path", default="")
    parser.add_argument("--privilege-mode", choices=("public", "customer", "customer_and_state", "answer_conditioned"), default="public")
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = str(Path(args.source_root) / "data")
    print(json.dumps(audit_task_budgets(args.source_root, args.model_path, args.extra_budget, briefs_path=args.briefs_path, privilege_mode=args.privilege_mode, use_privileged=args.privilege_mode != "public"), indent=2))
