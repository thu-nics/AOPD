# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase as EnvironmentManagerBase


def _trainer_validation_enabled(config) -> bool:
    """Whether this run can invoke validation at any point."""
    trainer = config.trainer
    return bool(trainer.get("val_only", False) or trainer.get("val_before_train", True) or int(trainer.get("test_freq", -1)) > 0)


def _validate_awm_context_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    model_length = int(config.actor_rollout_ref.rollout.max_model_len)
    lengths = {
        "data.max_prompt_length": prompt_length,
        "data.max_response_length": response_length,
        "actor_rollout_ref.rollout.max_model_len": model_length,
    }
    non_positive = [name for name, value in lengths.items() if value <= 0]
    if non_positive:
        raise ValueError("AWM context lengths must be positive: " + ", ".join(non_positive))
    if prompt_length + response_length > model_length:
        raise ValueError("AWM context budget requires data.max_prompt_length + data.max_response_length <= actor_rollout_ref.rollout.max_model_len")


def _validate_teacher_reward(config):
    from agent_system.environments.teacher_reward import (
        validate_teacher_reward_config,
    )

    reward = config.env.teacher_reward
    validate_teacher_reward_config(
        str(reward.mode),
        float(reward.frequency_bonus_scale),
    )


def _validate_mixed_decision_budget(config):
    budgets = (int(config.env.awm.train_max_steps), int(config.env.envscaler.train_max_steps))
    if config.env.agentic_mix.get("bounded_smoke", False):
        if budgets != (2, 2) or int(config.env.max_steps) != 2 or int(config.trainer.total_training_steps) != 1:
            raise ValueError("bounded mixed smoke requires one training step and two decisions per task")
    elif budgets != (20, 40):
        raise ValueError("mixed training requires 20 AWM and 40 EnvScaler decisions")


def make_envs(config):
    """
    Create enviroments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    rollout_mode = getattr(config.env.rollout, "mode", "vanilla")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if rollout_mode == "state_group":
        group_n = 1

    mixed_env_name = config.env.env_name.lower()
    if mixed_env_name == "awm_envscaler_agentic_opd":
        if bool(getattr(config.env.awm.oracle, "use_privileged_context", False)):
            raise ValueError("AWM does not support privileged teacher context; set env.awm.oracle.use_privileged_context=false")
        if rollout_mode != "state_group":
            raise ValueError("awm_envscaler_agentic_opd requires env.rollout.mode=state_group")
        if str(config.algorithm.adv_estimator) != "dapo":
            raise ValueError("awm_envscaler_agentic_opd requires algorithm.adv_estimator=dapo")
        if int(config.env.rollout.n) != 4:
            raise ValueError("mixed agentic OPD protocol requires four candidates per state")
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError("mixed agentic OPD protocol requires rollout.n=1 at inference")
        if not bool(config.actor_rollout_ref.rollout.multi_turn.enable):
            raise ValueError("mixed agentic OPD training requires multi-turn rollout")
        _validate_awm_context_budget(config)
        context = config.env.context
        if str(context.history_policy) != "token_budget":
            raise ValueError("mixed agentic OPD training requires token-budget context")
        if context.max_history_exchanges is not None and int(context.max_history_exchanges) < 0:
            raise ValueError("max_history_exchanges must be non-negative")
        counts = OmegaConf.to_container(config.env.agentic_mix.trajectory_counts, resolve=True)
        if sum(int(value) for value in counts.values()) != int(config.data.train_batch_size):
            raise ValueError("mixed AWM/EnvScaler counts must sum to train batch size")
        if set(counts) != {"awm", "envscaler"}:
            raise ValueError("mixed agentic OPD counts must contain awm and envscaler")
        if str(config.env.awm.verifier_mode) != "sql":
            raise ValueError("mixed AWM training requires SQL+LLM verification")
        if str(config.env.awm.reward_mode) != "semantic":
            raise ValueError("mixed AWM training requires semantic rewards")
        _validate_teacher_reward(config)
        _validate_mixed_decision_budget(config)
        terminal_judge = config.env.awm.terminal_judge
        runtime_failures = config.env.awm.runtime_failures
        if not bool(terminal_judge.enabled) or not bool(runtime_failures.enabled) or not bool(runtime_failures.judge.enabled):
            raise ValueError("mixed agentic OPD training requires AWM terminal/runtime judges")
        envscaler_runtime = config.env.envscaler.runtime_failures
        envscaler_judge = envscaler_runtime.judge
        if not bool(envscaler_runtime.enabled) or not bool(envscaler_judge.enabled):
            raise ValueError("mixed agentic OPD training requires the EnvScaler tool-exception judge")
        envscaler_confidence = int(envscaler_judge.confidence_threshold)
        if not 0 <= envscaler_confidence <= 100:
            raise ValueError("EnvScaler runtime judge confidence threshold must be in [0, 100]")

        from agent_system.environments.env_package.awm.runtime.oracle import (
            DeepSeekAWMOracleActor,
        )
        from agent_system.environments.env_package.envscaler.envs import (
            build_mixed_agentic_envs,
        )
        from agent_system.environments.env_package.envscaler.manager import (
            MixedAgenticEnvironmentManager,
            awm_projection,
        )
        from agent_system.environments.env_package.envscaler.source import (
            validate_envscaler_source,
        )

        validate_envscaler_source(config.env.envscaler.source_root)
        val_only = bool(config.trainer.get("val_only", False))
        oracle_actor = None
        if not val_only:
            oracle = config.env.awm.oracle
            runtime_judge = runtime_failures.judge
            oracle_actor = DeepSeekAWMOracleActor.remote(
                model=str(oracle.model),
                provider=str(oracle.provider),
                api_base=str(oracle.api_base),
                api_key_env=str(oracle.api_key_env),
                samples=int(oracle.samples),
                enable_thinking=bool(oracle.enable_thinking),
                reasoning_effort=(str(oracle.reasoning_effort) if oracle.reasoning_effort is not None else None),
                thinking_budget=(int(oracle.thinking_budget) if oracle.thinking_budget is not None else None),
                temperature=(float(oracle.temperature) if oracle.temperature is not None else None),
                top_p=(float(oracle.top_p) if oracle.top_p is not None else None),
                presence_penalty=(float(oracle.presence_penalty) if oracle.presence_penalty is not None else None),
                max_tokens=int(oracle.max_tokens),
                matcher_provider=str(oracle.matcher_provider),
                matcher_model=str(oracle.matcher_model),
                matcher_api_base=str(oracle.matcher_api_base),
                matcher_api_key_env=str(oracle.matcher_api_key_env),
                matcher_enable_thinking=oracle.get("matcher_enable_thinking"),
                matcher_reasoning_effort=oracle.get("matcher_reasoning_effort"),
                matcher_max_tokens=oracle.get("matcher_max_tokens"),
                matcher_max_concurrent_requests=oracle.get("matcher_max_concurrent_requests", 32),
                cache_path=str(oracle.cache_path),
                teacher_cache_import_paths=list(oracle.get("teacher_cache_import_paths", [])),
                teacher_cache_import_prompt_hashes=list(oracle.get("teacher_cache_import_prompt_hashes", [])),
                matcher_cache_path=str(oracle.matcher_cache_path),
                timeout_seconds=float(oracle.timeout_seconds),
                max_retries=int(oracle.max_retries),
                teacher_validity_max_retries=int(oracle.teacher_validity_max_retries),
                max_concurrent_requests=int(oracle.max_concurrent_requests),
                teacher_multi_call_fallback_enabled=bool(config.env.rollout.teacher_multi_call_fallback.enabled),
                teacher_multi_call_fallback_min_repeat_streak=int(config.env.rollout.teacher_multi_call_fallback.min_repeat_streak),
                runtime_judge_enabled=True,
                runtime_judge_data_dir=str(runtime_judge.data_dir),
                runtime_judge_reference_trials_path=(str(runtime_judge.reference_trials_path) if runtime_judge.reference_trials_path else None),
                runtime_judge_cache_path=str(runtime_judge.cache_path),
                runtime_judge_provider=str(runtime_judge.provider),
                runtime_judge_model=str(runtime_judge.model),
                runtime_judge_api_base=str(runtime_judge.api_base),
                runtime_judge_api_key_env=str(runtime_judge.api_key_env),
                runtime_judge_reasoning_effort=str(runtime_judge.reasoning_effort),
                runtime_judge_max_tokens=int(runtime_judge.max_tokens),
                runtime_judge_max_format_retries=int(runtime_judge.max_format_retries),
            )
        envs = None
        if not val_only:
            vector = build_mixed_agentic_envs(
                seed=int(config.env.seed),
                counts=counts,
                env_config=config.env,
                oracle_actor=oracle_actor,
            )
            envs = MixedAgenticEnvironmentManager(
                vector,
                awm_projection,
                config,
                oracle_actor=oracle_actor,
            )

        if not _trainer_validation_enabled(config):
            return envs, None

        if str(config.env.validation.env_name).lower() != "tau":
            raise ValueError("mixed AWM/EnvScaler periodic validation must use Tau")
        from agent_system.environments.env_package.tau_bench.envs import (
            build_tau_bench_envs,
            validate_tau_runtime_config,
            validate_tau_source,
        )
        from agent_system.environments.env_package.tau_bench.manager import (
            TauBenchEnvironmentManager,
            tau_projection,
        )

        validate_tau_source(config.env.tau.source_root)
        validate_tau_runtime_config(config.env.tau, require_oracle=False)
        validation_counts = OmegaConf.to_container(config.env.tau.validation_counts, resolve=True)
        if sum(int(value) for value in validation_counts.values()) != int(config.data.val_batch_size):
            raise ValueError("Tau validation counts must sum to data.val_batch_size")
        val_vector = build_tau_bench_envs(
            seed=int(config.env.tau.eval_seed),
            counts=validation_counts,
            env_config=config.env,
            group_n=1,
            is_train=False,
            oracle_actor=None,
        )
        val_envs = TauBenchEnvironmentManager(val_vector, tau_projection, config)
        return envs, val_envs
    if mixed_env_name in {"awm_agentic_opd", "awm_outcome"}:
        if bool(getattr(config.env.awm.oracle, "use_privileged_context", False)):
            raise ValueError("AWM does not support privileged teacher context; set env.awm.oracle.use_privileged_context=false")
        expected_mode = "state_group" if mixed_env_name == "awm_agentic_opd" else "vanilla"
        if rollout_mode != expected_mode:
            raise ValueError(f"{mixed_env_name} requires env.rollout.mode={expected_mode}")
        expected_reward_mode = "semantic" if mixed_env_name == "awm_agentic_opd" else "outcome"
        if str(config.env.awm.reward_mode) != expected_reward_mode:
            raise ValueError(f"{mixed_env_name} requires env.awm.reward_mode={expected_reward_mode}")
        if str(config.env.awm.verifier_mode) != "sql":
            raise ValueError("AWM training requires env.awm.verifier_mode=sql")
        terminal_judge = getattr(config.env.awm, "terminal_judge", None)
        if terminal_judge is None or not bool(terminal_judge.enabled):
            raise ValueError("AWM training requires the terminal SQL+LLM judge")
        for field in ("model", "api_base", "api_key_env"):
            if not str(getattr(terminal_judge, field, "") or "").strip():
                raise ValueError(f"AWM terminal judge {field} must be non-empty")
        if str(terminal_judge.reasoning_effort) != "max":
            raise ValueError("AWM terminal judge requires reasoning_effort=max")
        if int(terminal_judge.max_tokens) < 8192:
            raise ValueError("AWM terminal judge requires max_tokens >= 8192")
        if float(terminal_judge.timeout_seconds) <= 0:
            raise ValueError("AWM terminal judge timeout_seconds must be positive")
        if int(terminal_judge.max_retries) < 0:
            raise ValueError("AWM terminal judge max_retries must be non-negative")
        context = getattr(config.env, "context", None)
        if context is None or str(context.history_policy) != "token_budget":
            raise ValueError("AWM training requires env.context.history_policy=token_budget")
        if context.max_history_exchanges is not None and int(context.max_history_exchanges) < 0:
            raise ValueError("max_history_exchanges must be non-negative")
        if int(config.env.awm.train_max_steps) != 20:
            raise ValueError("AWM training protocol requires env.awm.train_max_steps=20")
        if int(config.env.max_steps) != 20:
            raise ValueError("AWM training protocol requires env.max_steps=20")
        if int(config.env.rollout.n) != 4:
            raise ValueError("AWM training protocol requires env.rollout.n=4")
        _validate_awm_context_budget(config)
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError("AWM protocol requires actor_rollout_ref.rollout.n=1; env.rollout.n controls the four candidates")
        if not bool(config.actor_rollout_ref.rollout.multi_turn.enable):
            raise ValueError("AWM protocol requires actor_rollout_ref.rollout.multi_turn.enable=true")
        expected_estimator = "dapo" if mixed_env_name == "awm_agentic_opd" else "grpo"
        if str(config.algorithm.adv_estimator) != expected_estimator:
            raise ValueError(f"{mixed_env_name} requires algorithm.adv_estimator={expected_estimator}")
        if mixed_env_name == "awm_agentic_opd":
            _validate_teacher_reward(config)
            runtime_failures = getattr(config.env.awm, "runtime_failures", None)
            if runtime_failures is None or not bool(runtime_failures.enabled):
                raise ValueError("AWM agentic OPD training requires runtime-failure handling")
            if int(runtime_failures.protocol_version) != 2:
                raise ValueError("AWM runtime-failure protocol mismatch")
            if not str(runtime_failures.path).strip():
                raise ValueError("AWM runtime-failure path must be non-empty")
            runtime_judge = getattr(runtime_failures, "judge", None)
            if runtime_judge is None or not bool(runtime_judge.enabled):
                raise ValueError("AWM agentic OPD training requires runtime 5xx judge")
            confidence_threshold = int(runtime_judge.confidence_threshold)
            if not 0 <= confidence_threshold <= 100:
                raise ValueError("AWM runtime judge confidence threshold must be in [0, 100]")
            if str(runtime_judge.reasoning_effort) != "max":
                raise ValueError("AWM runtime judge requires reasoning_effort=max")
            if int(runtime_judge.max_tokens) < 8192:
                raise ValueError("AWM runtime judge requires max_tokens >= 8192")
            for field in ("data_dir", "cache_path"):
                if not str(getattr(runtime_judge, field, "") or "").strip():
                    raise ValueError(f"AWM runtime judge {field} must be non-empty")

        from agent_system.environments.env_package.awm.runtime.envs import build_awm_envs
        from agent_system.environments.env_package.awm.runtime.manager import (
            AWMEnvironmentManager,
            awm_projection,
        )

        oracle_actor = None
        val_only = bool(config.trainer.get("val_only", False))
        if mixed_env_name == "awm_agentic_opd" and not val_only:
            from agent_system.environments.env_package.awm.runtime.oracle import (
                DeepSeekAWMOracleActor,
            )

            oracle_actor = DeepSeekAWMOracleActor.remote(
                model=str(config.env.awm.oracle.model),
                provider=str(config.env.awm.oracle.provider),
                api_base=str(config.env.awm.oracle.api_base),
                api_key_env=str(config.env.awm.oracle.api_key_env),
                samples=int(config.env.awm.oracle.samples),
                enable_thinking=bool(config.env.awm.oracle.enable_thinking),
                reasoning_effort=(str(config.env.awm.oracle.reasoning_effort) if config.env.awm.oracle.reasoning_effort is not None else None),
                thinking_budget=(int(config.env.awm.oracle.thinking_budget) if config.env.awm.oracle.thinking_budget is not None else None),
                temperature=(float(config.env.awm.oracle.temperature) if config.env.awm.oracle.temperature is not None else None),
                top_p=(float(config.env.awm.oracle.top_p) if config.env.awm.oracle.top_p is not None else None),
                presence_penalty=(float(config.env.awm.oracle.presence_penalty) if config.env.awm.oracle.presence_penalty is not None else None),
                max_tokens=int(config.env.awm.oracle.max_tokens),
                matcher_provider=str(config.env.awm.oracle.matcher_provider),
                matcher_model=str(config.env.awm.oracle.matcher_model),
                matcher_api_base=str(config.env.awm.oracle.matcher_api_base),
                matcher_api_key_env=str(config.env.awm.oracle.matcher_api_key_env),
                matcher_enable_thinking=config.env.awm.oracle.get("matcher_enable_thinking"),
                matcher_reasoning_effort=config.env.awm.oracle.get("matcher_reasoning_effort"),
                matcher_max_tokens=config.env.awm.oracle.get("matcher_max_tokens"),
                matcher_max_concurrent_requests=config.env.awm.oracle.get("matcher_max_concurrent_requests", 32),
                cache_path=str(config.env.awm.oracle.cache_path),
                teacher_cache_import_paths=list(config.env.awm.oracle.get("teacher_cache_import_paths", [])),
                teacher_cache_import_prompt_hashes=list(config.env.awm.oracle.get("teacher_cache_import_prompt_hashes", [])),
                matcher_cache_path=str(config.env.awm.oracle.matcher_cache_path),
                timeout_seconds=float(config.env.awm.oracle.timeout_seconds),
                max_retries=int(config.env.awm.oracle.max_retries),
                teacher_validity_max_retries=int(config.env.awm.oracle.teacher_validity_max_retries),
                max_concurrent_requests=int(config.env.awm.oracle.max_concurrent_requests),
                teacher_multi_call_fallback_enabled=bool(config.env.rollout.teacher_multi_call_fallback.enabled),
                teacher_multi_call_fallback_min_repeat_streak=int(config.env.rollout.teacher_multi_call_fallback.min_repeat_streak),
                runtime_judge_enabled=bool(runtime_failures.judge.enabled),
                runtime_judge_data_dir=str(runtime_failures.judge.data_dir),
                runtime_judge_reference_trials_path=(str(runtime_failures.judge.reference_trials_path) if runtime_failures.judge.reference_trials_path else None),
                runtime_judge_cache_path=str(runtime_failures.judge.cache_path),
                runtime_judge_provider=str(runtime_failures.judge.provider),
                runtime_judge_model=str(runtime_failures.judge.model),
                runtime_judge_api_base=str(runtime_failures.judge.api_base),
                runtime_judge_api_key_env=str(runtime_failures.judge.api_key_env),
                runtime_judge_reasoning_effort=str(runtime_failures.judge.reasoning_effort),
                runtime_judge_max_tokens=int(runtime_failures.judge.max_tokens),
                runtime_judge_max_format_retries=int(runtime_failures.judge.max_format_retries),
            )
        _envs = None
        if not val_only:
            _envs = build_awm_envs(
                seed=int(config.env.seed),
                count=int(config.data.train_batch_size),
                env_config=config.env,
                is_train=True,
                group_n=group_n,
                oracle_actor=oracle_actor,
            )
        envs = (
            None
            if val_only
            else AWMEnvironmentManager(
                _envs,
                awm_projection,
                config,
                oracle_actor=oracle_actor,
            )
        )
        if not _trainer_validation_enabled(config):
            return envs, None
        validation_env_name = str(config.env.validation.env_name).lower()
        if validation_env_name == "awm":
            _val_envs = build_awm_envs(
                seed=int(config.env.awm.eval_seed),
                count=int(config.data.val_batch_size),
                env_config=config.env,
                is_train=False,
                group_n=1,
                oracle_actor=None,
            )
            val_envs = AWMEnvironmentManager(_val_envs, awm_projection, config)
        elif validation_env_name == "tau":
            from agent_system.environments.env_package.tau_bench.envs import (
                build_tau_bench_envs,
                validate_tau_runtime_config,
                validate_tau_source,
            )
            from agent_system.environments.env_package.tau_bench.manager import (
                TauBenchEnvironmentManager,
                tau_projection,
            )

            validate_tau_source(config.env.tau.source_root)
            validate_tau_runtime_config(config.env.tau, require_oracle=False)
            validation_counts = OmegaConf.to_container(config.env.tau.validation_counts, resolve=True)
            if sum(int(value) for value in validation_counts.values()) != int(config.data.val_batch_size):
                raise ValueError("Tau validation counts must sum to data.val_batch_size")
            _val_envs = build_tau_bench_envs(
                seed=int(config.env.tau.eval_seed),
                counts=validation_counts,
                env_config=config.env,
                group_n=1,
                is_train=False,
                oracle_actor=None,
            )
            val_envs = TauBenchEnvironmentManager(_val_envs, tau_projection, config)
        else:
            raise ValueError(f"unsupported AWM validation environment: {validation_env_name}")
        return envs, val_envs
    elif mixed_env_name in {"tau_agentic_opd", "tau_outcome"}:
        _validate_teacher_reward(config)
        expected_mode = "state_group" if mixed_env_name == "tau_agentic_opd" else "vanilla"
        if rollout_mode != expected_mode:
            raise ValueError(f"{mixed_env_name} requires env.rollout.mode={expected_mode}")
        from agent_system.environments.env_package.tau_bench.envs import (
            build_tau_bench_envs,
            validate_tau_runtime_config,
            validate_tau_source,
        )
        from agent_system.environments.env_package.tau_bench.manager import (
            TauBenchEnvironmentManager,
            tau_projection,
        )

        validate_tau_source(config.env.tau.source_root)
        validate_tau_runtime_config(
            config.env.tau,
            require_oracle=mixed_env_name == "tau_agentic_opd",
        )
        train_counts = OmegaConf.to_container(config.env.tau.trajectory_counts, resolve=True)
        validation_counts = OmegaConf.to_container(config.env.tau.validation_counts, resolve=True)
        if sum(int(value) for value in train_counts.values()) != int(config.data.train_batch_size):
            raise ValueError("Tau training counts must sum to data.train_batch_size")
        if sum(int(value) for value in validation_counts.values()) != int(config.data.val_batch_size):
            raise ValueError("Tau validation counts must sum to data.val_batch_size")

        oracle_actor = None
        val_only = bool(config.trainer.get("val_only", False))
        if mixed_env_name == "tau_agentic_opd" and not val_only:
            from agent_system.environments.env_package.tau_bench.oracle import (
                TauTeacherActor,
            )

            oracle_actor = TauTeacherActor.remote(
                provider=str(config.env.tau.oracle.get("provider", "vllm")),
                teacher_source=str(config.env.tau.oracle.get("source", "external")),
                model=str(config.env.tau.oracle.model),
                api_base=str(config.env.tau.oracle.api_base),
                api_key_env=str(config.env.tau.oracle.api_key_env),
                samples=int(config.env.tau.oracle.samples),
                temperature=float(config.env.tau.oracle.temperature),
                top_p=float(config.env.tau.oracle.top_p),
                top_k=int(config.env.tau.oracle.top_k),
                min_p=float(config.env.tau.oracle.min_p),
                enable_thinking=bool(config.env.tau.oracle.enable_thinking),
                reasoning_effort=config.env.tau.oracle.get("reasoning_effort"),
                thinking_budget=config.env.tau.oracle.get("thinking_budget"),
                max_tokens=int(config.env.tau.oracle.max_tokens),
                cache_path=str(config.env.tau.oracle.cache_path),
                teacher_cache_import_paths=list(config.env.tau.oracle.get("teacher_cache_import_paths", [])),
                matcher_enabled=not bool(config.env.tau.get("mask_matcher_required_groups", False)),
                matcher_cache_path=str(config.env.tau.oracle.matcher_cache_path),
                matcher_provider=str(config.env.tau.oracle.get("matcher_provider", "openai-compatible")),
                matcher_profile=str(config.env.tau.oracle.get("matcher_profile", "default")),
                matcher_model=config.env.tau.oracle.get("matcher_model"),
                matcher_api_base=config.env.tau.oracle.get("matcher_api_base"),
                matcher_api_key_env=config.env.tau.oracle.get("matcher_api_key_env"),
                matcher_enable_thinking=config.env.tau.oracle.get("matcher_enable_thinking"),
                matcher_max_tokens=config.env.tau.oracle.get("matcher_max_tokens"),
                matcher_reasoning_effort=config.env.tau.oracle.get("matcher_reasoning_effort"),
                matcher_max_concurrent_requests=int(config.env.tau.oracle.get("matcher_max_concurrent_requests", 32)),
                timeout_seconds=float(config.env.tau.oracle.timeout_seconds),
                max_retries=int(config.env.tau.oracle.max_retries),
                teacher_validity_max_retries=int(config.env.tau.oracle.teacher_validity_max_retries),
                max_concurrent_requests=int(config.env.tau.oracle.max_concurrent_requests),
            )
        _envs = None
        if not val_only:
            _envs = build_tau_bench_envs(
                seed=int(config.env.seed),
                counts=train_counts,
                group_n=group_n,
                env_config=config.env,
                is_train=True,
                oracle_actor=oracle_actor,
            )
        _val_envs = build_tau_bench_envs(
            seed=int(config.env.tau.eval_seed),
            counts=validation_counts,
            env_config=config.env,
            group_n=1,
            is_train=False,
            oracle_actor=None,
        )
        envs = (
            None
            if val_only
            else TauBenchEnvironmentManager(
                _envs,
                tau_projection,
                config,
                oracle_actor=oracle_actor,
            )
        )
        val_envs = TauBenchEnvironmentManager(_val_envs, tau_projection, config)
        return envs, val_envs
    else:
        raise ValueError(f"Unsupported release environment: {mixed_env_name}")
