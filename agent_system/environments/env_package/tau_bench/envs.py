"""Ray environments for Tau Bench Airline and Retail training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from types import MethodType
from typing import Any, Mapping

import numpy as np
import ray

from .actions import (
    ParsedAction,
    canonical_action,
    parse_action,
    state_fingerprint,
    tau_messages_to_openai,
    to_tau_action,
    validate_tau_action,
)
from .oracle import ORACLE_PROTOCOL_VERSION, build_expert_messages

DOMAIN_ORDER = ("airline", "retail")
QUALIFICATION_PROTOCOL_VERSION = 1
TAU2_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"
TERMINAL_REWARD_PROTOCOL = "tau_db_x_communicate"


def compatibility_patch_sha256() -> str:
    patch = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "tau_bench"
        / "tau2_v1_optional_voice.patch"
    )
    if not patch.is_file():
        raise RuntimeError(f"Tau compatibility patch not found: {patch}")
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def interleave_domains(counts: Mapping[str, int]) -> list[str]:
    normalized = {domain: int(counts.get(domain, 0)) for domain in DOMAIN_ORDER}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("Tau trajectory counts must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Tau trajectory counts must be positive")
    used = {domain: 0 for domain in DOMAIN_ORDER}
    output = []
    for slot in range(total):
        candidates = [domain for domain in DOMAIN_ORDER if used[domain] < normalized[domain]]
        selected = max(
            candidates,
            key=lambda domain: (
                normalized[domain] * (slot + 1) / total - used[domain],
                -DOMAIN_ORDER.index(domain),
            ),
        )
        output.append(selected)
        used[selected] += 1
    return output


def interleave_grouped_domains(counts: Mapping[str, int], group_n: int) -> list[str]:
    if isinstance(group_n, bool) or not isinstance(group_n, int) or group_n <= 0:
        raise ValueError("group_n must be a positive integer")
    return [domain for domain in interleave_domains(counts) for _ in range(group_n)]


def select_uniform_argmax(rewards: list[float], rng: random.Random) -> int:
    if not rewards:
        raise ValueError("cannot select from an empty reward group")
    maximum = max(rewards)
    return rng.choice([index for index, reward in enumerate(rewards) if reward == maximum])


def load_qualification_manifest(
    path: str | Path,
    *,
    minimum_airline: int = 20,
    minimum_retail: int = 50,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tau qualification manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != QUALIFICATION_PROTOCOL_VERSION:
        raise RuntimeError(
            "Tau qualification manifest protocol_version mismatch: "
            f"expected {QUALIFICATION_PROTOCOL_VERSION}, "
            f"got {manifest.get('protocol_version')!r}"
        )
    if manifest.get("terminal_reward_protocol") != TERMINAL_REWARD_PROTOCOL:
        raise RuntimeError(
            "Tau qualification manifest must use terminal_reward_protocol="
            f"{TERMINAL_REWARD_PROTOCOL}"
        )
    if manifest.get("tau2_commit") != TAU2_COMMIT:
        raise RuntimeError(
            f"Tau qualification manifest must use pinned commit {TAU2_COMMIT}"
        )
    if manifest.get("tau2_compatibility_patch_sha256") != compatibility_patch_sha256():
        raise RuntimeError("Tau qualification manifest compatibility patch mismatch")
    if manifest.get("user_reasoning_enabled") is not False:
        raise RuntimeError(
            "Tau qualification manifest must disable user-simulator reasoning"
        )
    if manifest.get("oracle_samples_per_state") != 3:
        raise RuntimeError(
            "Tau qualification manifest must use three oracle samples per state"
        )
    if manifest.get("oracle_protocol_version") != ORACLE_PROTOCOL_VERSION:
        raise RuntimeError(
            "Tau qualification manifest oracle protocol mismatch: "
            f"expected {ORACLE_PROTOCOL_VERSION}, "
            f"got {manifest.get('oracle_protocol_version')!r}"
        )
    if manifest.get("trials_per_task") != 4:
        raise RuntimeError(
            "Tau qualification manifest must use four trials per task"
        )
    stable = manifest.get("stable_tasks") or {}
    test_tasks = manifest.get("test_tasks") or {}
    for domain in DOMAIN_ORDER:
        stable_ids = list(stable.get(domain) or [])
        test_ids = list(test_tasks.get(domain) or [])
        if len(stable_ids) != len(set(stable_ids)):
            raise RuntimeError(f"Tau manifest has duplicate stable {domain} task IDs")
        if len(test_ids) != len(set(test_ids)):
            raise RuntimeError(f"Tau manifest has duplicate test {domain} task IDs")
        if set(stable_ids) & set(test_ids):
            raise RuntimeError(
                f"Tau manifest train/test task IDs overlap for {domain}"
            )
    expected_test_counts = {"airline": 20, "retail": 40}
    actual_test_counts = {
        domain: len(test_tasks.get(domain) or []) for domain in DOMAIN_ORDER
    }
    if actual_test_counts != expected_test_counts:
        raise RuntimeError(
            "Tau qualification manifest must contain the complete test split: "
            f"expected {expected_test_counts}, got {actual_test_counts}"
        )
    counts = {domain: len(stable.get(domain) or []) for domain in DOMAIN_ORDER}
    if counts["airline"] < minimum_airline or counts["retail"] < minimum_retail:
        raise RuntimeError(
            "Tau expert qualification failed: "
            f"stable Airline={counts['airline']} (need {minimum_airline}), "
            f"Retail={counts['retail']} (need {minimum_retail})"
        )
    return manifest


def validate_tau_runtime_protocol(
    manifest: Mapping[str, Any],
    tau_config,
    *,
    require_oracle: bool,
) -> None:
    expected = {
        "user_llm": str(manifest.get("user_llm")),
        "user_temperature": float(manifest.get("user_temperature")),
        "user_reasoning_enabled": False,
    }
    actual = {
        "user_llm": str(tau_config.user_llm),
        "user_temperature": float(tau_config.user_temperature),
        "user_reasoning_enabled": bool(tau_config.user_reasoning_enabled),
    }
    if require_oracle:
        expected.update(
            {
                "expert_model": str(manifest.get("expert_model")),
                "oracle_samples_per_state": int(
                    manifest.get("oracle_samples_per_state")
                ),
                "oracle_reasoning_effort": str(
                    manifest.get("oracle_reasoning_effort")
                ),
                "oracle_max_tokens": int(manifest.get("oracle_max_tokens")),
            }
        )
        actual.update(
            {
                "expert_model": str(tau_config.oracle.model),
                "oracle_samples_per_state": int(tau_config.oracle.samples),
                "oracle_reasoning_effort": str(tau_config.oracle.reasoning_effort),
                "oracle_max_tokens": int(tau_config.oracle.max_tokens),
            }
        )
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={expected[key]!r}, runtime={actual[key]!r}"
            for key in mismatches
        )
        raise RuntimeError(f"Tau runtime protocol differs from qualification: {details}")


def _db_communicate_reward(self) -> tuple[float, str]:
    """Evaluate the reproducible Tau DB/COMMUNICATE reward components."""
    if self._simulation_run is None:
        return 0.0, json.dumps({}, indent=2)

    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    task = self._get_task()
    env_result = evaluate_simulation(
        simulation=self._simulation_run,
        task=task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=self.solo_mode,
        domain=self.domain,
    )
    reward = float(env_result.reward)
    components = {"env": env_result.model_dump(mode="json")}
    reward_basis = set(
        task.evaluation_criteria.reward_basis
        if task.evaluation_criteria is not None
        else []
    )
    if RewardType.COMMUNICATE in reward_basis:
        communicate_result = evaluate_simulation(
            simulation=self._simulation_run,
            task=task,
            evaluation_type=EvaluationType.COMMUNICATE,
            solo_mode=self.solo_mode,
            domain=self.domain,
        )
        reward *= float(communicate_result.reward)
        components["communicate"] = communicate_result.model_dump(mode="json")
    reward_info = {
        "reward": reward,
        "protocol": TERMINAL_REWARD_PROTOCOL,
        "components": components,
    }
    return reward, json.dumps(reward_info, indent=2)


def make_tau_agent_gym_env(**kwargs):
    """Create AgentGym with the shared reproducible terminal reward protocol."""
    try:
        from tau2.gym.gym_agent import AgentGymEnv
    except ImportError as exc:
        raise RuntimeError(
            "Tau Bench is not installed. Install the pinned tau2[gym] dependency "
            "with examples/tau_bench/install_tau2.sh."
        ) from exc
    env = AgentGymEnv(**kwargs)
    env._get_reward = MethodType(_db_communicate_reward, env)
    return env


@ray.remote
class TauBenchWorker:
    def __init__(
        self,
        *,
        domain: str,
        max_steps: int,
        user_llm: str,
        user_temperature: float,
        user_reasoning_enabled: bool,
        oracle_actor=None,
        seed: int = 0,
    ):
        if domain not in DOMAIN_ORDER:
            raise ValueError(f"unsupported Tau domain: {domain}")
        self.domain = domain
        self.max_steps = int(max_steps)
        self.user_llm = user_llm
        self.user_temperature = float(user_temperature)
        self.user_reasoning_enabled = bool(user_reasoning_enabled)
        self.oracle_actor = oracle_actor
        self.seed = int(seed)
        self._env = None
        self._task_id = None
        self._step = 0
        self._done = False
        self._rng = random.Random(seed)
        self._last_observation = ""
        self._last_info: dict[str, Any] = {}

    def _make_env(self, task_id: str):
        return make_tau_agent_gym_env(
            domain=self.domain,
            task_id=task_id,
            # Tau counts low-level message transitions; the adapter enforces
            # the public limit in agent decisions and reserves internal headroom.
            max_steps=self.max_steps * 3 + 4,
            user_llm=self.user_llm,
            user_llm_args={
                "temperature": self.user_temperature,
                "reasoning": {"enabled": self.user_reasoning_enabled},
            },
            all_messages_as_observation=False,
        )

    def _history(self) -> list[dict[str, Any]]:
        if self._env is None or self._env._agent is None:
            return []
        return tau_messages_to_openai(self._env._agent.observation)

    def _tools(self) -> list[dict[str, Any]]:
        if self._env is None:
            return []
        return [tool.openai_schema for tool in self._env._get_tools()]

    def _task(self) -> dict[str, Any]:
        return self._env._get_task().model_dump(mode="json")

    def _policy(self) -> str:
        return self._env._get_policy()

    def _annotate(self, info: dict[str, Any], **updates) -> dict[str, Any]:
        result = {
            "tau_domain": self.domain,
            "vpr_game": f"tau_{self.domain}",
            "tau_task_id": self._task_id,
            "step": self._step,
            "max_steps": self.max_steps,
            "terminal_success": bool(info.get("protocol_reward", 0.0) > 0) if self._done else None,
            **info,
        }
        result.update(updates)
        if self._env is not None:
            # Tau may include native Tool objects in step info. Keep the public
            # boundary stable for tokenizer/chat-template consumers.
            result["observation"] = self._last_observation
            result["chat"] = self._student_chat()
            result["tools"] = self._tools()
        return result

    def _observation_info(self) -> tuple[str, dict[str, Any]]:
        info = self._annotate(
            self._last_info,
            observation=self._last_observation,
            chat=self._student_chat(),
            tools=self._tools(),
        )
        return self._last_observation, info

    def _student_chat(self) -> list[dict[str, Any]]:
        system = (
            "You are a customer-service agent. Follow the domain policy and use the "
            "available tools when needed. At each turn, produce exactly one current "
            "action: either one tool call or one message to the user.\n\nDOMAIN POLICY:\n"
            f"{self._policy()}"
        )
        return [{"role": "system", "content": system}, *self._history()]

    def reset(self, *, task_id: str, seed: int | None = None):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._task_id = str(task_id)
        actual_seed = self.seed if seed is None else int(seed)
        self._rng.seed(actual_seed)
        self._env = self._make_env(self._task_id)
        observation, info = self._env.reset(seed=actual_seed)
        self._step = 0
        self._done = False
        self._last_observation = observation
        self._last_info = dict(info)
        self._last_info["protocol_reward"] = 0.0
        return self._observation_info()

    def _validate(self, action: ParsedAction) -> ParsedAction:
        return validate_tau_action(action, self._env._get_tools())

    def _finalize_at_decision_limit(self, observation, reward, done, info):
        if done or self._step < self.max_steps:
            return observation, reward, done, info
        observation, reward, terminated, truncated, info = self._env.step(
            json.dumps({"name": "done", "arguments": {}})
        )
        return observation, float(reward), bool(terminated or truncated), info

    def _execute(self, action: ParsedAction):
        observation, reward, terminated, truncated, info = self._env.step(to_tau_action(action))
        self._step += 1
        observation, reward, done, info = self._finalize_at_decision_limit(
            observation, float(reward), bool(terminated or truncated), info
        )
        self._done = done
        self._last_observation = observation
        self._last_info = dict(info)
        self._last_info["protocol_reward"] = float(reward)
        return observation, float(reward), self._done, self._last_info

    def step(self, raw_action: str):
        if self._done:
            info = self._annotate(
                self._last_info,
                parse_ok=True,
                illegal_action=False,
                is_action_valid=1,
                raw_action=raw_action,
                parsed_action="",
                protocol_reward=0.0,
                terminal_success=None,
                terminal_reason="already_done",
            )
            return self._last_observation, 0.0, True, info
        action = self._validate(parse_action(raw_action))
        if action.kind == "invalid":
            self._step += 1
            observation, protocol_reward, done, base_info = self._finalize_at_decision_limit(
                self._last_observation, 0.0, False, self._last_info
            )
            self._done = done
            self._last_observation = observation
            self._last_info = dict(base_info)
            info = self._annotate(
                self._last_info,
                parse_ok=False,
                illegal_action=True,
                is_action_valid=0,
                raw_action=raw_action,
                parsed_action="",
                protocol_reward=protocol_reward,
                terminal_success=bool(protocol_reward > 0) if done else None,
                terminal_reason="decision_limit" if done else "invalid_action",
            )
            return self._last_observation, protocol_reward, done, info
        observation, reward, done, base_info = self._execute(action)
        info = self._annotate(
            base_info,
            parse_ok=True,
            illegal_action=False,
            is_action_valid=1,
            raw_action=raw_action,
            parsed_action=canonical_action(action),
            terminal_success=bool(reward > 0) if done else None,
            protocol_reward=reward,
            terminal_reason="environment_done" if done else None,
        )
        return observation, reward, done, info

    def step_candidate_group(self, raw_actions: list[str]):
        if self.oracle_actor is None:
            raise RuntimeError("state-group Tau rollout requires an oracle actor")
        candidates = [self._validate(parse_action(raw)) for raw in raw_actions]
        history = self._history()
        tools = self._tools()
        fingerprint = state_fingerprint(self.domain, self._task_id, history, tools)
        expert_messages = build_expert_messages(
            policy=self._policy(), task=self._task(), history=history
        )
        sampled = ray.get(
            self.oracle_actor.sample_oracle_set.remote(
                state_fingerprint=fingerprint,
                messages=expert_messages,
                tools=tools,
            )
        )
        oracle_actions = [self._validate(ParsedAction(**action)) for action in sampled]
        oracle_actions = [action for action in oracle_actions if action.kind != "invalid"]
        oracle_keys = {
            canonical_action(action) for action in oracle_actions if action.kind == "tool"
        }
        oracle_messages = [
            action.content or "" for action in oracle_actions if action.kind == "message"
        ]
        candidate_message_positions = [
            index for index, action in enumerate(candidates) if action.kind == "message"
        ]
        candidate_messages = [candidates[index].content or "" for index in candidate_message_positions]
        semantic_matches = ray.get(
            self.oracle_actor.match_messages.remote(oracle_messages, candidate_messages)
        ) if candidate_messages and oracle_messages else [False] * len(candidate_messages)
        message_match_by_index = dict(zip(candidate_message_positions, semantic_matches))

        rewards = []
        for index, action in enumerate(candidates):
            if action.kind == "invalid":
                rewards.append(-1.0)
            elif action.kind == "tool":
                rewards.append(1.0 if canonical_action(action) in oracle_keys else 0.0)
            else:
                rewards.append(1.0 if message_match_by_index.get(index, False) else 0.0)
        selected_index = select_uniform_argmax(rewards, self._rng)
        selected_action = candidates[selected_index]

        if selected_action.kind == "invalid":
            self._step += 1
            observation, protocol_reward, done, base_info = self._finalize_at_decision_limit(
                self._last_observation, 0.0, False, self._last_info
            )
            self._done = done
            self._last_observation = observation
            self._last_info = dict(base_info)
            terminal_reason = "decision_limit" if done else "invalid_noop"
        else:
            observation, protocol_reward, done, base_info = self._execute(selected_action)
            terminal_reason = "environment_done" if done else None

        oracle_set_size = len(oracle_actions)
        candidate_results = []
        for index, (raw, action, reward) in enumerate(zip(raw_actions, candidates, rewards)):
            candidate_info = self._annotate(
                base_info if index == selected_index else self._last_info,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                raw_action=raw,
                parsed_action="" if action.kind == "invalid" else canonical_action(action),
                terminal_success=bool(protocol_reward > 0) if done and index == selected_index else None,
                terminal_reason=terminal_reason if index == selected_index else None,
                move_optimal=bool(reward > 0),
                legal_non_oracle=bool(action.kind != "invalid" and reward == 0),
                oracle_set_size=oracle_set_size,
                oracle_policy_tier="deepseek_v4_flash_samples",
                state_fingerprint=fingerprint,
                protocol_reward=protocol_reward if index == selected_index else 0.0,
                state_group_selection_type="uniform_argmax",
                state_group_random_select_prob=0.0,
            )
            candidate_results.append((observation, reward, done and index == selected_index, candidate_info))

        selected_info = candidate_results[selected_index][3]
        return (
            candidate_results,
            selected_index,
            observation,
            rewards[selected_index],
            done,
            selected_info,
        )

    def current_observation_info(self):
        return self._observation_info()

    def close(self):
        if self._env is not None:
            self._env.close()


class TauBenchVectorEnv:
    def __init__(self, workers, domains, seeds):
        if not (len(workers) == len(domains) == len(seeds)):
            raise ValueError("workers, domains, and seeds must align")
        self.workers = workers
        self.domains = domains
        self.seeds = seeds
        self._episode = 0

    def _validate_rows(self, kwargs):
        if kwargs is None or len(kwargs) != len(self.workers):
            raise ValueError(f"expected {len(self.workers)} Tau env kwargs")
        for domain, row in zip(self.domains, kwargs, strict=True):
            if str(row.get("domain")) != domain:
                raise ValueError(
                    f"Tau row domain {row.get('domain')!r} does not match slot {domain!r}"
                )
            if not row.get("task_id"):
                raise ValueError("Tau row is missing task_id")

    def reset(self, kwargs=None):
        self._validate_rows(kwargs)
        offset = self._episode * 100003
        self._episode += 1
        futures = [
            worker.reset.remote(task_id=row["task_id"], seed=seed + offset)
            for worker, row, seed in zip(self.workers, kwargs, self.seeds, strict=True)
        ]
        results = ray.get(futures)
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions):
        results = ray.get(
            [worker.step.remote(action) for worker, action in zip(self.workers, actions, strict=True)]
        )
        return (
            [result[0] for result in results],
            np.asarray([result[1] for result in results], dtype=np.float32),
            np.asarray([result[2] for result in results], dtype=bool),
            [result[3] for result in results],
        )

    def step_candidate_groups(self, candidate_action_groups, active_indices=None):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        indices = [int(index) for index in active_indices]
        if len(indices) != len(candidate_action_groups):
            raise ValueError("active_indices must align with candidate groups")
        results = ray.get(
            [
                self.workers[index].step_candidate_group.remote(group)
                for index, group in zip(indices, candidate_action_groups, strict=True)
            ]
        )
        return (
            [result[0] for result in results],
            np.asarray([result[1] for result in results], dtype=np.int32),
            [result[2] for result in results],
            np.asarray([result[3] for result in results], dtype=np.float32),
            np.asarray([result[4] for result in results], dtype=bool),
            [result[5] for result in results],
        )

    def close(self):
        for worker in self.workers:
            ray.kill(worker)


def build_tau_bench_envs(
    *,
    seed: int,
    counts: Mapping[str, int],
    env_config,
    is_train: bool,
    group_n: int,
    oracle_actor=None,
):
    domains = interleave_grouped_domains(counts, group_n)
    max_steps = int(env_config.tau.train_max_steps if is_train else env_config.tau.eval_max_steps)
    workers = []
    seeds = []
    for index, domain in enumerate(domains):
        worker_seed = int(seed) + index
        workers.append(
            TauBenchWorker.remote(
                domain=domain,
                max_steps=max_steps,
                user_llm=str(env_config.tau.user_llm),
                user_temperature=float(env_config.tau.user_temperature),
                user_reasoning_enabled=bool(env_config.tau.user_reasoning_enabled),
                oracle_actor=oracle_actor,
                seed=worker_seed,
            )
        )
        seeds.append(worker_seed)
    return TauBenchVectorEnv(workers, domains, seeds)
