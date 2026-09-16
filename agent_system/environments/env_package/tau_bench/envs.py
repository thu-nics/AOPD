"""Ray environments for Tau Bench Airline and Retail training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

import numpy as np
import ray

from agent_system.environments.prompts.agentic_opd import (
    TAU_PROMPT_PROTOCOL,
    prompt_hash,
    tau_system_prompt,
)
from agent_system.environments.teacher_reward import (
    DEFAULT_FREQUENCY_BONUS_SCALE,
    select_with_appearance_counterfactual,
    teacher_match_reward,
    validate_teacher_reward_config,
)

from .actions import (
    TRANSFER_TOOL_NAME,
    ParsedAction,
    canonical_action,
    is_transfer_notice,
    parse_action,
    state_fingerprint,
    successful_transfer_in_history,
    tau_messages_to_openai,
    to_tau_action,
    tool_schema_hash,
    validate_tau_action,
)
from .oracle import build_teacher_messages

DOMAIN_ORDER = ("airline", "retail")
TASK_MANIFEST_PROTOCOL_VERSION = 3
TAU2_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"
TERMINAL_REWARD_PROTOCOL = "tau_db_x_communicate"
TAU_DEFAULT_TEACHER_REWARD_MODE = "appearance"
TAU_DEFAULT_NATIVE_LOG_LEVEL = "WARNING"
DEFAULT_USER_API_BASE = "http://127.0.0.1:8000/v1"
REQUIRED_USER_SIMULATOR_DATA = (
    "data/tau2/user_simulator/simulation_guidelines.md",
    "data/tau2/user_simulator/simulation_guidelines_tools.md",
)

OFFICIAL_TASK_COUNTS = {
    "train": {"airline": 30, "retail": 74},
    "test": {"airline": 20, "retail": 40},
    "base": {"airline": 50, "retail": 114},
}


def configure_tau_native_logging(level: str = TAU_DEFAULT_NATIVE_LOG_LEVEL) -> str:
    """Keep Tau's native per-turn traces out of training worker logs."""
    normalized = str(level).strip().upper()
    supported = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if normalized not in supported:
        raise ValueError(f"unsupported Tau native log level {level!r}; expected one of " + ", ".join(sorted(supported)))

    # Tau uses Loguru and its default sink includes INFO/DEBUG records containing
    # complete message objects. A user-simulator message may embed raw provider
    # responses and reasoning, so one line can be hundreds of KiB. Each Ray Tau
    # worker is a dedicated process; replacing its Loguru sink is therefore
    # isolated from the driver and other environment families.
    from loguru import logger as tau_logger

    tau_logger.remove()
    tau_logger.add(
        sys.stderr,
        level=normalized,
        backtrace=False,
        diagnose=False,
        colorize=False,
    )
    return normalized


def compatibility_patch_path() -> Path:
    return Path(__file__).resolve().parents[4] / "examples" / "tau_bench" / "tau2_v1_optional_voice.patch"


def compatibility_patch_sha256() -> str:
    patch = compatibility_patch_path()
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


def frequency_sensitive_group(
    rewards: list[float],
    appearance_scores: list[float],
) -> bool:
    """Whether teacher frequency changes group advantages or argmax choices."""
    if len(rewards) != len(appearance_scores):
        raise ValueError("frequency and appearance scores must align")
    if len(rewards) < 2:
        return False
    frequency_values = np.asarray(rewards, dtype=np.float64)
    appearance_values = np.asarray(appearance_scores, dtype=np.float64)

    def normalized(values):
        if np.ptp(values) <= 1e-8:
            return np.zeros_like(values)
        return (values - values.mean()) / (values.std(ddof=1) + 1e-6)

    advantage_changed = not np.allclose(
        normalized(frequency_values),
        normalized(appearance_values),
        atol=1e-6,
    )
    argmax_changed = set(np.flatnonzero(frequency_values == frequency_values.max())) != set(np.flatnonzero(appearance_values == appearance_values.max()))
    return bool(advantage_changed or argmax_changed)


def tau_source_root() -> Path:
    try:
        import tau2
    except ImportError as exc:
        raise RuntimeError("Tau Bench is not installed; run examples/tau_bench/install_tau2.sh") from exc
    return Path(tau2.__file__).resolve().parents[2]


def validate_tau_source(expected_root: str | Path | None = None) -> dict[str, str]:
    """Fail loudly unless the editable Tau source matches the pinned protocol."""
    root = tau_source_root()
    if expected_root is not None:
        expected = Path(expected_root).expanduser().resolve()
        if root != expected:
            raise RuntimeError(f"Tau source root mismatch: expected {expected}, got {root}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve Tau source commit under {root}") from exc
    missing_data = [relative for relative in REQUIRED_USER_SIMULATOR_DATA if not (root / relative).is_file()]
    if missing_data:
        raise RuntimeError("Tau source is missing required user-simulator data: " + ", ".join(missing_data) + "; rerun examples/tau_bench/install_tau2.sh")
    if commit != TAU2_COMMIT:
        raise RuntimeError(f"Tau source must be pinned to {TAU2_COMMIT}, got {commit}")
    patch = compatibility_patch_path()
    try:
        subprocess.check_output(
            ["git", "apply", "--unidiff-zero", "--reverse", "--check", str(patch)],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Tau compatibility patch is not applied cleanly to the pinned source") from exc

    return {
        "source_root": str(root),
        "tau2_commit": commit,
        "compatibility_patch_sha256": compatibility_patch_sha256(),
    }


def validate_tau_runtime_config(tau_config, *, require_oracle: bool) -> None:
    if not bool(tau_config.user_reasoning_enabled):
        raise RuntimeError("Tau Qwen3.5 user simulator requires thinking enabled")
    if float(tau_config.user_temperature) != 1.0:
        raise RuntimeError("Tau requires user_temperature=1")
    if require_oracle:
        if int(tau_config.oracle.samples) != 3:
            raise RuntimeError("Tau agentic OPD training requires exactly three oracle samples")
        if not str(tau_config.oracle.model).strip():
            raise RuntimeError("Tau agentic OPD training requires a non-empty oracle model")


def tau_user_simulator_llm_args(
    user_llm: str,
    *,
    api_base: str,
    api_key_env: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_tokens: int,
    reasoning_enabled: bool,
    generation_retries: int,
) -> dict[str, Any]:
    """Build Qwen-compatible LiteLLM arguments for local or remote serving."""
    api_key = os.environ.get(str(api_key_env))
    if not api_key:
        raise RuntimeError(f"missing required Tau user API key environment variable {api_key_env}")
    return {
        "api_base": str(api_base).rstrip("/"),
        "api_key": api_key,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "presence_penalty": float(presence_penalty),
        "max_tokens": int(max_tokens),
        "_validation_retries": int(generation_retries),
        "extra_body": {
            "top_k": int(top_k),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "chat_template_kwargs": {
                "enable_thinking": bool(reasoning_enabled),
            },
        },
    }


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
    reward_basis = set(task.evaluation_criteria.reward_basis if task.evaluation_criteria is not None else [])
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
        from .user_simulator import ValidatedUserAgentGymEnv
    except ImportError as exc:
        raise RuntimeError("Tau Bench is not installed. Install the pinned tau2[gym] dependency with examples/tau_bench/install_tau2.sh.") from exc
    env = ValidatedUserAgentGymEnv(**kwargs)
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
        user_api_base: str = DEFAULT_USER_API_BASE,
        user_api_key_env: str = "TAU_USER_API_KEY",
        user_temperature: float = 1.0,
        user_top_p: float = 0.95,
        user_top_k: int = 20,
        user_min_p: float = 0.0,
        user_presence_penalty: float = 1.5,
        user_repetition_penalty: float = 1.0,
        user_max_tokens: int = 8192,
        user_reasoning_enabled: bool = True,
        user_generation_retries: int = 2,
        oracle_actor=None,
        teacher_reward_mode: str = TAU_DEFAULT_TEACHER_REWARD_MODE,
        native_log_level: str = TAU_DEFAULT_NATIVE_LOG_LEVEL,
        frequency_bonus_scale: float = DEFAULT_FREQUENCY_BONUS_SCALE,
        use_privileged_teacher_context: bool = False,
        transfer_reward_guard_enabled: bool = False,
        mask_matcher_required_groups: bool = False,
        seed: int = 0,
    ):
        if domain not in DOMAIN_ORDER:
            raise ValueError(f"unsupported Tau domain: {domain}")
        self.domain = domain
        self.max_steps = int(max_steps)
        self.native_log_level = configure_tau_native_logging(native_log_level)
        self.user_llm = user_llm
        self.user_api_base = str(user_api_base)
        self.user_api_key_env = str(user_api_key_env)
        self.user_temperature = float(user_temperature)
        self.user_top_p = float(user_top_p)
        self.user_top_k = int(user_top_k)
        self.user_min_p = float(user_min_p)
        self.user_presence_penalty = float(user_presence_penalty)
        self.user_repetition_penalty = float(user_repetition_penalty)
        self.user_max_tokens = int(user_max_tokens)
        self.user_reasoning_enabled = bool(user_reasoning_enabled)
        self.user_generation_retries = int(user_generation_retries)
        self.oracle_actor = oracle_actor
        (
            self.teacher_reward_mode,
            self.frequency_bonus_scale,
        ) = validate_teacher_reward_config(
            teacher_reward_mode,
            frequency_bonus_scale,
        )
        self.use_privileged_teacher_context = bool(use_privileged_teacher_context)
        self.transfer_reward_guard_enabled = bool(transfer_reward_guard_enabled)
        self.mask_matcher_required_groups = bool(mask_matcher_required_groups)
        self._transfer_succeeded = False
        self.seed = int(seed)
        self._env = None
        self._task_id = None
        self._step = 0
        self._done = False
        self._rng = random.Random(seed)
        self._last_observation = ""
        self._last_info: dict[str, Any] = {}
        self._last_step_hit_decision_limit = False
        self._prepared_teacher_supervision: dict[str, Any] | None = None

    def _make_env(self, task_id: str):
        return make_tau_agent_gym_env(
            domain=self.domain,
            task_id=task_id,
            # Tau counts low-level message transitions; the adapter enforces
            # the public limit in agent decisions and reserves internal headroom.
            max_steps=self.max_steps * 3 + 4,
            user_llm=self.user_llm,
            user_llm_args=tau_user_simulator_llm_args(
                self.user_llm,
                api_base=self.user_api_base,
                api_key_env=self.user_api_key_env,
                temperature=self.user_temperature,
                top_p=self.user_top_p,
                top_k=self.user_top_k,
                min_p=self.user_min_p,
                presence_penalty=self.user_presence_penalty,
                repetition_penalty=self.user_repetition_penalty,
                max_tokens=self.user_max_tokens,
                reasoning_enabled=self.user_reasoning_enabled,
                generation_retries=self.user_generation_retries,
            ),
            all_messages_as_observation=False,
        )

    def _history(self) -> list[dict[str, Any]]:
        if self._env is None or self._env._agent is None:
            return []
        return tau_messages_to_openai(self._env._agent.observation)

    def _tools(self) -> list[dict[str, Any]]:
        if self._env is None:
            return []
        from agent_system.environments.tool_matching_metadata import callable_tool_matching_metadata

        native_tools = self._env._get_tools()
        schemas = [tool.openai_schema for tool in native_tools]
        schema_key = (self.domain, tool_schema_hash(schemas))
        if getattr(self, "_matching_schema_key", None) != schema_key:
            self._tool_matching_metadata = callable_tool_matching_metadata(native_tools, family="tau", environment=self.domain)
            self._matching_schema_key = schema_key
        return schemas

    def _task(self) -> dict[str, Any]:
        return self._env._get_task().model_dump(mode="json")

    def _policy(self) -> str:
        return self._env._get_policy()

    def _annotate(self, info: dict[str, Any], **updates) -> dict[str, Any]:
        result = {
            "tau_domain": self.domain,
            "agent_prompt_protocol": TAU_PROMPT_PROTOCOL,
            "agent_prompt_hash": (prompt_hash(self._student_chat()[0]["content"]) if self._env is not None else ""),
            "agentic_env_family": "tau",
            "vpr_game": f"tau_{self.domain}",
            "tau_task_id": self._task_id,
            "tool_schema_hash": tool_schema_hash(self._tools()),
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
        system = tau_system_prompt(self._policy())
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
        self._transfer_succeeded = False
        self._env = self._make_env(self._task_id)
        observation, info = self._env.reset(seed=actual_seed)
        self._step = 0
        self._done = False
        self._last_observation = observation
        self._last_info = dict(info)
        self._last_info["protocol_reward"] = 0.0
        self._last_step_hit_decision_limit = False
        self._prepared_teacher_supervision = None
        return self._observation_info()

    def _validate(self, action: ParsedAction) -> ParsedAction:
        return validate_tau_action(action, self._env._get_tools())

    def _finalize_at_decision_limit(self, observation, reward, done, info):
        self._last_step_hit_decision_limit = False
        if done or self._step < self.max_steps:
            return observation, reward, done, info
        self._last_step_hit_decision_limit = True
        observation, reward, terminated, truncated, info = self._env.step(json.dumps({"name": "done", "arguments": {}}))
        return observation, float(reward), bool(terminated or truncated), info

    def _execute(self, action: ParsedAction):
        observation, reward, terminated, truncated, info = self._env.step(to_tau_action(action))
        if self.transfer_reward_guard_enabled and action.kind == "tool" and action.name == TRANSFER_TOOL_NAME:
            self._transfer_succeeded = self._transfer_succeeded or successful_transfer_in_history(self._history())
        self._step += 1
        observation, reward, done, info = self._finalize_at_decision_limit(observation, float(reward), bool(terminated or truncated), info)
        self._done = done
        self._last_observation = observation
        self._last_info = dict(info)
        self._last_info["protocol_reward"] = float(reward)
        return observation, float(reward), self._done, self._last_info

    def _terminate_without_action(
        self,
        *,
        action_kind: str,
        terminal_reason: str,
        **updates,
    ) -> dict[str, Any]:
        """Finalize one trajectory without creating a trainable student action."""
        self._prepared_teacher_supervision = None
        protocol_reward = 0.0
        terminal_outcome_valid = False
        termination_error = None
        try:
            observation, reward, terminated, truncated, info = self._env.step(json.dumps({"name": "done", "arguments": {}}))
            self._last_observation = observation
            self._last_info = dict(info)
            protocol_reward = float(reward)
            terminal_outcome_valid = bool(terminated or truncated)
        except Exception as exc:
            termination_error = f"{type(exc).__name__}: {exc}"
            self._last_info = dict(self._last_info)
        self._done = True
        self._last_step_hit_decision_limit = False
        self._last_info["protocol_reward"] = protocol_reward
        return self._annotate(
            self._last_info,
            action_kind=action_kind,
            semantic_train_mask=False,
            runtime_train_mask=False,
            protocol_reward=protocol_reward,
            terminal_success=(bool(protocol_reward > 0) if terminal_outcome_valid else None),
            terminal_outcome_valid=terminal_outcome_valid,
            terminal_reason=terminal_reason,
            termination_error=termination_error,
            tool_calling=0,
            state_group_advanced=False,
            **updates,
        )

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
                tool_calling=0,
                terminal_reason="already_done",
            )
            return self._last_observation, 0.0, True, info
        action = self._validate(parse_action(raw_action))
        if action.kind == "invalid":
            self._step += 1
            observation, protocol_reward, done, base_info = self._finalize_at_decision_limit(self._last_observation, 0.0, False, self._last_info)
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
                action_kind="invalid",
                protocol_reward=protocol_reward,
                terminal_success=bool(protocol_reward > 0) if done else None,
                tool_calling=0,
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
            action_kind=action.kind,
            terminal_success=bool(reward > 0) if done else None,
            protocol_reward=reward,
            tool_calling=int(action.kind == "tool"),
            terminal_reason=("decision_limit" if self._last_step_hit_decision_limit else "environment_done" if done else None),
            decision_limit_reached=self._last_step_hit_decision_limit,
        )
        return observation, reward, done, info

    def terminate_context_overflow(
        self,
        diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """End one oversized state without treating it as a student action."""
        return self._terminate_without_action(
            action_kind="context_overflow",
            terminal_reason="context_budget_exceeded",
            context_overflow=True,
            runtime_failure=False,
            **dict(diagnostics),
        )

    def _teacher_privileged_context(self) -> dict[str, Any]:
        task = self._task()
        criteria = task.get("evaluation_criteria") or {}
        return {
            "task_id": task.get("id"),
            "user_scenario": task.get("user_scenario"),
            "reference_resolution_actions": criteria.get("actions") or [],
            "evaluation_criteria": criteria,
        }

    def _validate_teacher_visible_chat(
        self,
        visible_chat: list[dict[str, Any]] | None,
    ) -> None:
        if visible_chat is None:
            return
        logical_chat = self._student_chat()
        first_user_index = next(
            (index for index, message in enumerate(logical_chat) if message.get("role") == "user"),
            None,
        )
        visible_first_user_index = next(
            (index for index, message in enumerate(visible_chat) if message.get("role") == "user"),
            None,
        )
        if first_user_index is None or visible_first_user_index is None or visible_chat[0] != logical_chat[0] or visible_chat[visible_first_user_index] != logical_chat[first_user_index]:
            raise ValueError("Tau teacher-visible chat must preserve system and initial user")
        logical = iter(logical_chat[1:])
        for expected in visible_chat[1:]:
            if not any(candidate == expected for candidate in logical):
                raise ValueError("Tau teacher-visible chat is not an ordered logical-history view")

    async def prepare_teacher_supervision(
        self,
        visible_chat: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if self.oracle_actor is None:
            raise RuntimeError("state-group Tau rollout requires an oracle actor")
        if self._done:
            raise RuntimeError("cannot prepare a Tau state group after termination")
        self._validate_teacher_visible_chat(visible_chat)
        student_visible_chat = self._student_chat() if visible_chat is None else visible_chat
        tools = self._tools()
        teacher_messages = build_teacher_messages(
            student_visible_chat,
            privileged_context=(self._teacher_privileged_context() if self.use_privileged_teacher_context else None),
            use_privileged_context=self.use_privileged_teacher_context,
        )
        teacher_context_mode = "privileged" if self.use_privileged_teacher_context else "student_visible"
        fingerprint = state_fingerprint(
            self.domain,
            self._task_id,
            (teacher_messages if self.use_privileged_teacher_context else student_visible_chat),
            tools,
        )
        sampled: list[dict[str, Any]] = []
        try:
            sampled = await self.oracle_actor.sample_multiset.remote(
                state_fingerprint=fingerprint,
                messages=teacher_messages,
                tools=tools,
                teacher_context_mode=teacher_context_mode,
            )
        except Exception as exc:
            return False, self._terminate_without_action(
                action_kind="teacher_failure",
                terminal_reason="teacher_failure",
                teacher_failure=True,
                teacher_error=f"{type(exc).__name__}: {exc}",
                teacher_frequency=0,
                teacher_multiset=[],
                teacher_multiset_size=0,
                teacher_sample_count=0,
                teacher_valid_sample_count=0,
                teacher_invalid_sample_count=0,
                teacher_unique_action_count=0,
                teacher_action_kind_disagreement=False,
                matcher_failure=False,
                matcher_matrix=[],
                oracle_set_size=0,
                state_fingerprint=fingerprint,
                teacher_context_mode=teacher_context_mode,
            )
        teacher_sample_count = 3
        if len(sampled) > teacher_sample_count:
            raise RuntimeError(f"Tau teacher returned {len(sampled)} samples; expected at most 3")
        try:
            validated_samples = [self._validate(ParsedAction(**action)) for action in sampled]
        except Exception as exc:
            return False, self._terminate_without_action(
                action_kind="teacher_failure",
                terminal_reason="teacher_failure",
                teacher_failure=True,
                teacher_error=f"{type(exc).__name__}: {exc}",
                teacher_frequency=0,
                teacher_multiset=[],
                teacher_multiset_size=0,
                teacher_sample_count=teacher_sample_count,
                teacher_valid_sample_count=0,
                teacher_invalid_sample_count=teacher_sample_count,
                teacher_unique_action_count=0,
                teacher_action_kind_disagreement=False,
                matcher_failure=False,
                matcher_matrix=[],
                oracle_set_size=0,
                state_fingerprint=fingerprint,
                teacher_context_mode=teacher_context_mode,
            )
        teacher_actions = [action for action in validated_samples if action.kind != "invalid"]
        teacher_invalid_sample_count = teacher_sample_count - len(teacher_actions)
        if not teacher_actions:
            return False, self._terminate_without_action(
                action_kind="teacher_failure",
                terminal_reason="teacher_failure",
                teacher_failure=True,
                teacher_error="RuntimeError: Tau teacher multiset has no valid action",
                teacher_frequency=0,
                teacher_multiset=[],
                teacher_multiset_size=0,
                teacher_sample_count=teacher_sample_count,
                teacher_valid_sample_count=0,
                teacher_invalid_sample_count=teacher_invalid_sample_count,
                teacher_unique_action_count=0,
                teacher_action_kind_disagreement=False,
                matcher_failure=False,
                matcher_matrix=[],
                oracle_set_size=0,
                state_fingerprint=fingerprint,
                teacher_context_mode=teacher_context_mode,
            )
        teacher_multiset = [action.to_dict() for action in teacher_actions]
        teacher_unique_action_count = len({canonical_action(action) for action in teacher_actions})
        teacher_action_kind_disagreement = len({action.kind for action in teacher_actions}) > 1
        self._prepared_teacher_supervision = {
            "state_fingerprint": fingerprint,
            "teacher_actions": teacher_actions,
            "teacher_multiset": teacher_multiset,
            "teacher_sample_count": teacher_sample_count,
            "teacher_invalid_sample_count": teacher_invalid_sample_count,
            "teacher_unique_action_count": teacher_unique_action_count,
            "teacher_action_kind_disagreement": teacher_action_kind_disagreement,
            "teacher_context_mode": teacher_context_mode,
        }
        return True, self._annotate(
            self._last_info,
            action_kind="teacher_preflight",
            semantic_train_mask=False,
            runtime_train_mask=False,
            teacher_failure=False,
            teacher_error=None,
            matcher_failure=False,
            matcher_matrix=[],
            teacher_frequency=0,
            teacher_multiset=teacher_multiset,
            teacher_multiset_size=len(teacher_multiset),
            oracle_set_size=teacher_unique_action_count,
            teacher_sample_count=teacher_sample_count,
            teacher_valid_sample_count=len(teacher_actions),
            teacher_invalid_sample_count=teacher_invalid_sample_count,
            teacher_unique_action_count=teacher_unique_action_count,
            teacher_action_kind_disagreement=teacher_action_kind_disagreement,
            teacher_reward_mode=self.teacher_reward_mode,
            frequency_bonus_scale=self.frequency_bonus_scale,
            state_fingerprint=fingerprint,
            teacher_context_mode=teacher_context_mode,
            state_group_advanced=False,
        )

    def _matcher_failure_group(
        self,
        *,
        raw_actions: list[str],
        candidates: list[ParsedAction],
        prepared: Mapping[str, Any],
        fingerprint: str,
        error: Exception,
        group_metadata: Mapping[str, Any] | None = None,
    ):
        """Mask an unsupervised group without executing a student candidate."""
        error_text = f"{type(error).__name__}: {error}"
        failure_info = self._terminate_without_action(
            action_kind="matcher_failure",
            terminal_reason="matcher_failure",
            teacher_failure=False,
            teacher_error=None,
            matcher_failure=True,
            matcher_error=error_text,
            teacher_frequency=0,
            teacher_multiset=prepared["teacher_multiset"],
            teacher_multiset_size=len(prepared["teacher_multiset"]),
            teacher_sample_count=prepared["teacher_sample_count"],
            teacher_valid_sample_count=len(prepared["teacher_actions"]),
            teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
            teacher_unique_action_count=prepared["teacher_unique_action_count"],
            teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
            matcher_matrix=[],
            oracle_set_size=prepared["teacher_unique_action_count"],
            state_fingerprint=fingerprint,
            teacher_context_mode=prepared["teacher_context_mode"],
            **dict(group_metadata or {}),
        )
        candidate_results = []
        for raw, action in zip(raw_actions, candidates, strict=True):
            info = self._annotate(
                self._last_info,
                raw_action=raw,
                parsed_action=("" if action.kind == "invalid" else canonical_action(action)),
                action_kind=action.kind,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                semantic_train_mask=False,
                runtime_train_mask=False,
                move_optimal=False,
                legal_non_oracle=False,
                selection_score=0.0,
                raw_selection_score=0.0,
                raw_semantic_reward=0.0,
                teacher_frequency=0,
                teacher_match_count=0,
                teacher_multiset=prepared["teacher_multiset"],
                teacher_multiset_size=len(prepared["teacher_multiset"]),
                teacher_sample_count=prepared["teacher_sample_count"],
                teacher_valid_sample_count=len(prepared["teacher_actions"]),
                teacher_invalid_sample_count=prepared["teacher_invalid_sample_count"],
                teacher_unique_action_count=prepared["teacher_unique_action_count"],
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=False,
                teacher_failure=False,
                teacher_error=None,
                matcher_failure=True,
                matcher_error=error_text,
                matcher_matrix=[],
                oracle_set_size=prepared["teacher_unique_action_count"],
                state_fingerprint=fingerprint,
                teacher_context_mode=prepared["teacher_context_mode"],
                protocol_reward=0.0,
                terminal_success=None,
                terminal_outcome_valid=False,
                terminal_reason=None,
                state_group_selection_type="none",
                state_group_random_select_prob=0.0,
                state_group_advanced=False,
                appearance_counterfactual_selected=False,
                **dict(group_metadata or {}),
            )
            candidate_results.append((self._last_observation, 0.0, False, info))
        return (
            candidate_results,
            -1,
            self._last_observation,
            0.0,
            True,
            failure_info,
        )

    async def step_candidate_group(
        self,
        raw_actions: list[str],
        visible_chat: list[dict[str, Any]] | None = None,
        group_metadata: Mapping[str, Any] | None = None,
    ):
        if self.oracle_actor is None:
            raise RuntimeError("state-group Tau rollout requires an oracle actor")
        self._validate_teacher_visible_chat(visible_chat)
        student_visible_chat = self._student_chat() if visible_chat is None else visible_chat
        tools = self._tools()
        teacher_messages = build_teacher_messages(
            student_visible_chat,
            privileged_context=(self._teacher_privileged_context() if self.use_privileged_teacher_context else None),
            use_privileged_context=self.use_privileged_teacher_context,
        )
        fingerprint = state_fingerprint(
            self.domain,
            self._task_id,
            (teacher_messages if self.use_privileged_teacher_context else student_visible_chat),
            tools,
        )
        prepared = self._prepared_teacher_supervision
        self._prepared_teacher_supervision = None
        if prepared is None or prepared["state_fingerprint"] != fingerprint:
            raise RuntimeError("Tau candidate scoring requires matching frozen teacher supervision")
        candidates = [self._validate(parse_action(raw)) for raw in raw_actions]
        teacher_actions = prepared["teacher_actions"]
        teacher_multiset = prepared["teacher_multiset"]
        teacher_sample_count = int(prepared["teacher_sample_count"])
        teacher_messages = [action.content or "" for action in teacher_actions if action.kind == "message"]
        candidate_message_positions = [index for index, action in enumerate(candidates) if action.kind == "message"]
        candidate_messages = [candidates[index].content or "" for index in candidate_message_positions]
        matcher_required_group = False
        try:
            from agent_system.environments.action_matching import build_tool_match_plan, finish_tool_match_plan, match_candidate_tools

            if self.mask_matcher_required_groups:
                # Detect unknowns without constructing API/source evidence.
                plan = build_tool_match_plan(teacher_actions, candidates, tools, student_visible_chat, tool_matching_metadata=getattr(self, "_tool_matching_metadata", {}), semantic_enabled=False)
                tool_matches = finish_tool_match_plan(plan, [])
                matrix = [[teacher.strip() == candidate.strip() for teacher in teacher_messages] for candidate in candidate_messages]
                matcher_required_group = bool(plan["unresolved_positions"]) or any(not value for row in matrix for value in row)
                matched = {"counts": [sum(row) for row in matrix], "matrix": matrix}
            else:
                tool_matches = await match_candidate_tools(
                    self.oracle_actor,
                    teacher_actions,
                    candidates,
                    tools,
                    student_visible_chat,
                    tool_matching_metadata=getattr(self, "_tool_matching_metadata", {}),
                )
                matched = (
                    await self.oracle_actor.match_message_pairs.remote(
                        teacher_messages,
                        candidate_messages,
                        student_visible_chat,
                        tools,
                    )
                    if candidate_messages and teacher_messages
                    else {
                        "counts": [0] * len(candidate_messages),
                        # Keep one (empty) row per candidate so the result obeys the
                        # same candidate x teacher matrix contract when the teacher
                        # multiset contains tool calls only.
                        "matrix": [[] for _ in candidate_messages],
                    }
                )
            if not isinstance(matched, Mapping):
                raise TypeError("Tau matcher result must be an object")
            match_counts = matched.get("counts")
            matcher_matrix = matched.get("matrix")
            if not isinstance(match_counts, list) or len(match_counts) != len(candidate_message_positions):
                raise ValueError("Tau matcher returned the wrong number of counts")
            if not isinstance(matcher_matrix, list) or len(matcher_matrix) != len(candidate_message_positions):
                raise ValueError("Tau matcher returned the wrong number of matrix rows")
            for count, row in zip(match_counts, matcher_matrix, strict=True):
                if isinstance(count, bool) or not isinstance(count, int) or not isinstance(row, list) or len(row) != len(teacher_messages) or any(not isinstance(value, bool) for value in row) or count != sum(row):
                    raise ValueError("Tau matcher returned an invalid pairwise Boolean matrix")
        except Exception as exc:
            return self._matcher_failure_group(
                raw_actions=raw_actions,
                candidates=candidates,
                prepared=prepared,
                fingerprint=fingerprint,
                error=exc,
                group_metadata=group_metadata,
            )
        message_match_by_index = dict(zip(candidate_message_positions, match_counts, strict=True))
        message_matrix_by_index = dict(zip(candidate_message_positions, matcher_matrix, strict=True)) if matcher_matrix else {}

        rewards = []
        teacher_match_counts = []
        for index, action in enumerate(candidates):
            if matcher_required_group:
                # Zero is only a masked placeholder, NOT an unmatched verdict.
                match_count, reward = 0, 0.0
            elif action.kind == "invalid":
                match_count = 0
                reward = -1.0
            elif action.kind == "tool":
                match_count = tool_matches["counts"][index]
                reward = teacher_match_reward(
                    match_count,
                    teacher_sample_count=teacher_sample_count,
                    mode=self.teacher_reward_mode,
                    frequency_bonus_scale=self.frequency_bonus_scale,
                )
            else:
                match_count = int(message_match_by_index.get(index, 0))
                reward = teacher_match_reward(
                    match_count,
                    teacher_sample_count=teacher_sample_count,
                    mode=self.teacher_reward_mode,
                    frequency_bonus_scale=self.frequency_bonus_scale,
                )
            teacher_match_counts.append(match_count)
            rewards.append(reward)
        appearance_scores = [
            -1.0 if action.kind == "invalid" else (1.0 if match_count > 0 else 0.0)
            for action, match_count in zip(
                candidates,
                teacher_match_counts,
                strict=True,
            )
        ]
        # Semantic equivalence and native action validity are unchanged. This
        # independent protocol penalty uses the reward floor: a zero cap could
        # still reinforce a violation when the other candidates get -1.
        # Apply before either selection policy, preserving raw teacher verdicts.
        raw_rewards = list(rewards)
        transfer_without_tool = [self.transfer_reward_guard_enabled and not self._transfer_succeeded and is_transfer_notice(action) for action in candidates]
        for index, violation in enumerate(transfer_without_tool):
            if violation:
                rewards[index] = -1.0
                appearance_scores[index] = -1.0
        frequency_sensitive = frequency_sensitive_group(rewards, appearance_scores)
        if matcher_required_group:
            valid_indices = [index for index, action in enumerate(candidates) if action.kind != "invalid"]
            # An unknown pair necessarily has a valid candidate. Do not stop at
            # clarification turns or use an incomplete reward argmax.
            selected_index = self._rng.choice(valid_indices)
            appearance_index = selected_index
            frequency_sensitive = False
        else:
            selected_index, appearance_index = select_with_appearance_counterfactual(rewards, appearance_scores, self._rng)
        selected_action = candidates[selected_index]
        appearance_action = candidates[appearance_index]
        frequency_changed_selection = canonical_action(selected_action) != canonical_action(appearance_action)

        if selected_action.kind == "invalid":
            self._step += 1
            observation, protocol_reward, done, base_info = self._finalize_at_decision_limit(
                self._last_observation,
                0.0,
                False,
                self._last_info,
            )
            self._done = done
            self._last_observation = observation
            self._last_info = dict(base_info)
            terminal_reason = "decision_limit" if done else "invalid_noop"
        else:
            observation, protocol_reward, done, base_info = self._execute(selected_action)
            terminal_reason = "decision_limit" if self._last_step_hit_decision_limit else "environment_done" if done else None

        oracle_set_size = int(prepared["teacher_unique_action_count"])
        candidate_results = []
        for index, (raw, action, reward, match_count) in enumerate(
            zip(
                raw_actions,
                candidates,
                rewards,
                teacher_match_counts,
                strict=True,
            )
        ):
            candidate_info = self._annotate(
                base_info if index == selected_index else self._last_info,
                parse_ok=action.kind != "invalid",
                illegal_action=action.kind == "invalid",
                is_action_valid=int(action.kind != "invalid"),
                raw_action=raw,
                parsed_action=("" if action.kind == "invalid" else canonical_action(action)),
                action_kind=action.kind,
                semantic_train_mask=not matcher_required_group,
                runtime_train_mask=True,
                matcher_required_group=matcher_required_group,
                selection_score=reward,
                raw_selection_score=raw_rewards[index],
                raw_semantic_reward=raw_rewards[index],
                transfer_without_tool=transfer_without_tool[index],
                terminal_success=(bool(protocol_reward > 0) if done and index == selected_index else None),
                tool_calling=int(action.kind == "tool"),
                terminal_reason=(terminal_reason if index == selected_index else None),
                decision_limit_reached=(self._last_step_hit_decision_limit and index == selected_index),
                move_optimal=bool(reward > 0),
                legal_non_oracle=bool(not matcher_required_group and action.kind != "invalid" and reward == 0),
                oracle_set_size=oracle_set_size,
                oracle_policy_tier="teacher_samples_multiset",
                teacher_sample_count=teacher_sample_count,
                teacher_valid_sample_count=len(teacher_actions),
                teacher_invalid_sample_count=int(prepared["teacher_invalid_sample_count"]),
                teacher_unique_action_count=oracle_set_size,
                teacher_frequency=match_count,
                teacher_match_count=match_count,
                tool_argument_semantic_match_count=tool_matches["added_counts"][index],
                tool_argument_normalized_match_count=tool_matches["normalized_counts"][index],
                teacher_multiset=teacher_multiset,
                teacher_multiset_size=len(teacher_multiset),
                teacher_action_kind_disagreement=prepared["teacher_action_kind_disagreement"],
                frequency_sensitive_group=frequency_sensitive,
                teacher_failure=False,
                teacher_error=None,
                matcher_failure=False,
                matcher_error=None,
                teacher_reward_mode=self.teacher_reward_mode,
                frequency_bonus_scale=self.frequency_bonus_scale,
                matcher_matrix=[] if matcher_required_group else message_matrix_by_index.get(index, []),
                state_fingerprint=fingerprint,
                teacher_context_mode=prepared["teacher_context_mode"],
                protocol_reward=(protocol_reward if index == selected_index else 0.0),
                state_group_selection_type="random" if matcher_required_group else "uniform_argmax",
                state_group_random_select_prob=1.0 if matcher_required_group else 0.0,
                state_group_advanced=(index == selected_index),
                appearance_counterfactual_selected=(not matcher_required_group and index == appearance_index),
                appearance_counterfactual_action_kind=None if matcher_required_group else appearance_action.kind,
                frequency_changed_selection=frequency_changed_selection,
                frequency_changed_selection_to_tool=bool(frequency_changed_selection and selected_action.kind == "tool"),
                frequency_changed_selection_to_message=bool(frequency_changed_selection and selected_action.kind == "message"),
                **dict(group_metadata or {}),
            )
            candidate_results.append(
                (
                    observation,
                    reward,
                    done and index == selected_index,
                    candidate_info,
                )
            )

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
                raise ValueError(f"Tau row domain {row.get('domain')!r} does not match slot {domain!r}")
            if not row.get("task_id"):
                raise ValueError("Tau row is missing task_id")

    def reset(self, kwargs=None):
        self._validate_rows(kwargs)
        offset = self._episode * 100003
        self._episode += 1
        futures = [
            worker.reset.remote(
                task_id=row["task_id"],
                seed=int(row.get("seed", seed + offset)),
            )
            for worker, row, seed in zip(self.workers, kwargs, self.seeds, strict=True)
        ]
        results = ray.get(futures)
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions):
        results = ray.get([worker.step.remote(action) for worker, action in zip(self.workers, actions, strict=True)])
        return (
            [result[0] for result in results],
            np.asarray([result[1] for result in results], dtype=np.float32),
            np.asarray([result[2] for result in results], dtype=bool),
            [result[3] for result in results],
        )

    def start_teacher_preflight(self, *, active_indices, visible_chats):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(visible_chats):
            raise ValueError("active_indices must align with Tau teacher-visible chats")
        return [
            self.workers[index].prepare_teacher_supervision.remote(visible_chat)
            for index, visible_chat in zip(
                indices,
                visible_chats,
                strict=True,
            )
        ]

    @staticmethod
    def finish_teacher_preflight(pending):
        return ray.get(pending)

    def terminate_context_overflows(self, *, active_indices, diagnostics):
        indices = [int(index) for index in active_indices]
        if len(indices) != len(diagnostics):
            raise ValueError("active_indices must align with Tau context diagnostics")
        return ray.get([self.workers[index].terminate_context_overflow.remote(item) for index, item in zip(indices, diagnostics, strict=True)])

    def step_candidate_groups(
        self,
        candidate_action_groups,
        active_indices=None,
        visible_chats=None,
        group_metadata=None,
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        indices = [int(index) for index in active_indices]
        if len(indices) != len(candidate_action_groups):
            raise ValueError("active_indices must align with candidate groups")
        if any(index < 0 or index >= len(self.workers) for index in indices):
            raise ValueError("active_indices reference unknown Tau workers")
        if visible_chats is None:
            visible_chats = [None] * len(indices)
        if len(visible_chats) != len(indices):
            raise ValueError("visible chats must align with Tau candidate groups")
        if group_metadata is None:
            group_metadata = [None] * len(indices)
        if len(group_metadata) != len(indices):
            raise ValueError("group metadata must align with Tau candidate groups")
        results = ray.get(
            [
                self.workers[index].step_candidate_group.remote(
                    group,
                    visible_chat=visible_chat,
                    group_metadata=metadata,
                )
                for index, group, visible_chat, metadata in zip(
                    indices,
                    candidate_action_groups,
                    visible_chats,
                    group_metadata,
                    strict=True,
                )
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
    group_n: int,
    env_config,
    is_train: bool,
    oracle_actor=None,
):
    domains = interleave_grouped_domains(counts, group_n)
    teacher_reward = env_config.teacher_reward
    max_steps = int(env_config.tau.train_max_steps if is_train else env_config.tau.eval_max_steps)
    worker_options = dict(getattr(env_config, "resources_per_worker", {}) or {})
    worker_factory = TauBenchWorker.options(**worker_options) if worker_options else TauBenchWorker

    workers = []
    seeds = []
    for index, domain in enumerate(domains):
        worker_seed = int(seed) + index
        workers.append(
            worker_factory.remote(
                domain=domain,
                max_steps=max_steps,
                user_llm=str(env_config.tau.user_llm),
                user_api_base=str(env_config.tau.user_api_base),
                user_api_key_env=str(env_config.tau.user_api_key_env),
                user_temperature=float(env_config.tau.user_temperature),
                user_top_p=float(env_config.tau.user_top_p),
                user_top_k=int(env_config.tau.user_top_k),
                user_min_p=float(env_config.tau.user_min_p),
                user_presence_penalty=float(env_config.tau.user_presence_penalty),
                user_repetition_penalty=float(env_config.tau.user_repetition_penalty),
                user_max_tokens=int(env_config.tau.user_max_tokens),
                user_reasoning_enabled=bool(env_config.tau.user_reasoning_enabled),
                user_generation_retries=int(env_config.tau.user_generation_retries),
                oracle_actor=oracle_actor,
                transfer_reward_guard_enabled=bool(getattr(env_config.tau, "transfer_reward_guard_enabled", False)) if oracle_actor is not None and is_train else False,
                mask_matcher_required_groups=bool(getattr(env_config.tau, "mask_matcher_required_groups", False)) if oracle_actor is not None and is_train else False,
                teacher_reward_mode=str(teacher_reward.mode),
                native_log_level=str(
                    getattr(
                        env_config.tau,
                        "native_log_level",
                        TAU_DEFAULT_NATIVE_LOG_LEVEL,
                    )
                ),
                frequency_bonus_scale=float(teacher_reward.frequency_bonus_scale),
                use_privileged_teacher_context=bool(
                    getattr(
                        env_config.tau.oracle,
                        "use_privileged_context",
                        False,
                    )
                )
                if oracle_actor is not None
                else False,
                seed=worker_seed,
            )
        )
        seeds.append(worker_seed)
    return TauBenchVectorEnv(workers, domains, seeds)
