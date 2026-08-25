from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

import agent_system.environments.env_package.awm.runtime.envs as awm_envs
from agent_system.environments.env_manager import (
    _validate_awm_context_budget,
    _validate_teacher_reward,
)


def _compose(config_name):
    config_dir = Path(__file__).parents[2] / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=config_name)


def test_awm_uses_low_memory_sampled_entropy_monitoring():
    for config_name in ("awm_agentic_opd", "awm_outcome"):
        config = _compose(config_name)
        actor = config.actor_rollout_ref.actor
        assert actor.entropy_coeff == 0.0
        assert actor.log_entropy_metrics is False
        assert actor.log_sampled_entropy_metrics is True
        assert config.data.return_raw_chat is True
        assert config.data.shuffle is False
        assert config.data.apply_chat_template_kwargs.enable_thinking is True
        assert config.data.max_prompt_length == 27904
        assert config.data.max_response_length == 4096
        assert config.data.max_prompt_length + config.data.max_response_length == 32000
        assert config.env.context.history_policy == "token_budget"
        assert config.env.context.max_history_exchanges is None
        assert config.env.awm.verifier_mode == "sql"
        terminal = config.env.awm.terminal_judge
        assert terminal.enabled is True
        assert terminal.model == "deepseek-v4-flash"
        assert terminal.api_base == "https://api.deepseek.com"
        assert terminal.api_key_env == "DEEPSEEK_API_KEY"
        assert terminal.reasoning_effort == "max"
        assert terminal.max_tokens == 8192
        assert terminal.timeout_seconds == 300
        assert terminal.max_retries == 5
        assert config.actor_rollout_ref.rollout.n == 1
        assert config.actor_rollout_ref.rollout.multi_turn.enable is True
        rollout = config.actor_rollout_ref.rollout
        assert rollout.temperature == 0.6
        assert rollout.top_p == 0.95
        assert rollout.top_k == 20
        validation = rollout.val_kwargs
        if config_name == "awm_agentic_opd":
            assert config.env.teacher_reward.mode == "frequency_weighted"
            assert config.env.teacher_reward.frequency_bonus_scale == 0.5
            assert config.env.awm.oracle.use_privileged_context is False
            assert config.algorithm.state_group.compact_policy_rows is True
            assert config.algorithm.state_group.diagnostic_only is False
            assert validation.do_sample is False
            assert validation.temperature == 0.0
            assert validation.top_p == 1.0
            assert validation.top_k == -1
            assert validation.min_p == 0.0
        else:
            assert config.algorithm.state_group.compact_policy_rows is False
            assert validation.do_sample is True
            assert validation.temperature == 0.6
            assert validation.top_p == 0.95
            assert validation.top_k == 20
        assert validation.n == 1
        assert validation.seed == config.env.awm.eval_seed == 300


def test_tau_agentic_opd_teacher_context_and_compaction_defaults():
    config = _compose("tau_agentic_opd")

    assert config.env.tau.oracle.use_privileged_context is False
    assert config.env.teacher_reward.mode == "appearance"
    assert config.env.teacher_reward.frequency_bonus_scale == 0.5
    assert config.algorithm.state_group.compact_policy_rows is True
    assert _compose("tau_outcome").algorithm.state_group.compact_policy_rows is False


def test_awm_context_budget_is_configurable_but_must_fit_model():
    config = _compose("awm_agentic_opd")
    _validate_awm_context_budget(config)

    config.data.max_prompt_length = 28672
    config.data.max_response_length = 4096
    config.actor_rollout_ref.rollout.max_model_len = 32768
    _validate_awm_context_budget(config)

    config.data.max_prompt_length = 28673
    with pytest.raises(ValueError, match="AWM context budget requires"):
        _validate_awm_context_budget(config)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_awm_frequency_bonus_scale_accepts_supported_ablation_range(scale):
    config = _compose("awm_agentic_opd")
    config.env.teacher_reward.frequency_bonus_scale = scale
    _validate_teacher_reward(config)


@pytest.mark.parametrize("scale", [-0.1, float("inf"), float("nan")])
def test_awm_frequency_bonus_scale_must_be_finite_and_non_negative(scale):
    config = _compose("awm_agentic_opd")
    config.env.teacher_reward.frequency_bonus_scale = scale
    with pytest.raises(ValueError, match="frequency_bonus_scale"):
        _validate_teacher_reward(config)


def test_teacher_reward_mode_is_shared_and_strictly_validated():
    config = _compose("awm_agentic_opd")
    config.env.teacher_reward.mode = "appearance"
    _validate_teacher_reward(config)

    config.env.teacher_reward.mode = "unknown"
    with pytest.raises(ValueError, match="teacher reward mode"):
        _validate_teacher_reward(config)


def test_formal_agentic_opd_config_uses_tau_airline_validation():
    config = _compose("awm_agentic_opd")
    assert config.data.train_batch_size == 64
    assert config.data.val_batch_size == 16
    assert config.env.rollout.n == 4
    assert config.env.validation.env_name == "tau"
    assert list(config.env.tau.validation_domains) == ["airline"]
    assert dict(config.env.tau.validation_counts) == {"airline": 16, "retail": 0}
    assert config.env.tau.validation_task_split == "base"
    assert config.env.tau.validation_trials == 1
    runtime = config.env.awm.runtime_failures
    assert runtime.protocol_version == 2
    assert runtime.judge.enabled is True
    assert runtime.judge.confidence_threshold == 80
    assert runtime.judge.reasoning_effort == "max"
    assert runtime.judge.max_tokens == 8192
    assert runtime.judge.reference_trials_path is None
    assert runtime.judge.cache_path.endswith("runtime_judge.jsonl")


@pytest.mark.parametrize("max_history_exchanges", [None, 0, 3, 10])
def test_awm_worker_accepts_configurable_max_history_exchanges(max_history_exchanges):
    worker_class = awm_envs.AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        max_history_exchanges=max_history_exchanges,
        verifier_mode="sql",
        reward_mode="semantic",
    )

    assert worker.max_history_exchanges == max_history_exchanges


def test_awm_worker_rejects_negative_max_history_exchanges():
    worker_class = awm_envs.AWMWorker.__ray_metadata__.modified_class
    with pytest.raises(ValueError, match="max_history_exchanges must be non-negative"):
        worker_class(
            base_url="unused",
            max_steps=20,
            max_history_exchanges=-1,
            verifier_mode="sql",
            reward_mode="semantic",
        )


def test_awm_builder_honors_fractional_ray_worker_resources(monkeypatch):
    options = []
    created = []

    def fake_remote(**kwargs):
        created.append(kwargs)
        return object()

    def fake_options(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(remote=fake_remote)

    monkeypatch.setattr(
        awm_envs,
        "AWMWorker",
        SimpleNamespace(options=fake_options),
    )
    env_config = SimpleNamespace(
        teacher_reward=SimpleNamespace(
            mode="frequency_weighted",
            frequency_bonus_scale=0.5,
        ),
        context=SimpleNamespace(max_history_exchanges=None),
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
        awm=SimpleNamespace(
            train_max_steps=20,
            eval_max_steps=20,
            reward_mode="semantic",
            runtime_failures=None,
            base_url="http://127.0.0.1:8000",
            verifier_mode="sql",
            terminal_judge=SimpleNamespace(
                api_base="https://api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
                model="deepseek-v4-flash",
            ),
        ),
    )

    env = awm_envs.build_awm_envs(
        seed=3,
        count=2,
        group_n=4,
        env_config=env_config,
        is_train=True,
    )

    assert options == [{"num_cpus": 0.1, "num_gpus": 0}]
    assert len(env.workers) == 8
    assert env.seeds == list(range(3, 11))
    assert len(created) == 8


def test_base_trainer_preserves_entropy_metric_default():
    actor = _compose("ppo_trainer").actor_rollout_ref.actor
    assert actor.log_entropy_metrics is True
    assert actor.log_sampled_entropy_metrics is False


def test_training_launcher_scopes_artifacts_and_forwards_overrides():
    launcher = (Path(__file__).parents[2] / "examples" / "awm" / "train" / "run_training.sh").read_text(encoding="utf-8")

    assert 'RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"' in launcher
    assert 'RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_STAMP}"' in launcher
    assert 'TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"' in launcher
    assert 'VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"' in launcher
    assert 'trainer.val_before_train="$VAL_BEFORE_TRAIN"' in launcher
    assert 'TRAIN_TASK_COUNT="${TRAIN_TASK_COUNT:-}"' in launcher
    assert 'TRAIN_TASK_FRACTION="${TRAIN_TASK_FRACTION:-}"' in launcher
    assert 'FINAL_POOL_DIR="${FINAL_POOL_DIR:-$REPO_ROOT/runs/awm_data_processing/03_static_feasibility_judge}"' in launcher
    assert 'TRAIN_SELECTION_MANIFEST="$FINAL_POOL_DIR/health_manifest.json"' in launcher
    assert 'ENVSCALER_POOL="${ENVSCALER_POOL:-$REPO_ROOT/runs/envscaler_data_processing/02_static_feasibility_judge/envscaler_training_pool.parquet}"' in launcher
    assert 'ENVSCALER_MANIFEST="${ENVSCALER_MANIFEST:-$REPO_ROOT/runs/envscaler_data_processing/02_static_feasibility_judge/health_manifest.json}"' in launcher
    assert '"$SCRIPT_DIR/../data/slice_training_pool.py"' in launcher
    assert '"$SCRIPT_DIR/../data/materialize_training_schedule.py"' in launcher
    assert 'EXPERT_CACHE_DIR="${EXPERT_CACHE_DIR:-$RUN_DIR/cache}"' in launcher
    assert 'RUNTIME_JUDGE_REFERENCE_TRIALS="${RUNTIME_JUDGE_REFERENCE_TRIALS:-}"' in launcher
    assert 'RUNTIME_JUDGE_CACHE_PATH="${RUNTIME_JUDGE_CACHE_PATH:-$EXPERT_CACHE_DIR/runtime_judge.jsonl}"' in launcher
    assert 'RUNTIME_JUDGE_CONFIDENCE_THRESHOLD="${RUNTIME_JUDGE_CONFIDENCE_THRESHOLD:-80}"' in launcher
    assert 'RUNTIME_JUDGE_MAX_TOKENS="${RUNTIME_JUDGE_MAX_TOKENS:-8192}"' in launcher
    assert 'TEACHER_REWARD_MODE="${TEACHER_REWARD_MODE:-frequency_weighted}"' in launcher
    assert 'FREQUENCY_BONUS_SCALE="${FREQUENCY_BONUS_SCALE:-0.5}"' in launcher
    assert 'STATE_GROUP_ADVANTAGE_MODE="${STATE_GROUP_ADVANTAGE_MODE:-mean_then_batch_whiten}"' in launcher
    assert 'MIN_EFFECTIVE_STATE_GROUPS="${MIN_EFFECTIVE_STATE_GROUPS:-1}"' in launcher
    assert 'STATE_GROUP_DIAGNOSTIC_ONLY="${STATE_GROUP_DIAGNOSTIC_ONLY:-0}"' in launcher
    assert 'RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"' in launcher
    assert 'trainer.resume_from_path="${RESUME_FROM_PATH:-null}"' in launcher
    assert "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" in launcher
    assert 'MANAGE_AWM_SERVER="${MANAGE_AWM_SERVER:-1}"' in launcher
    assert 'TAU_USER_LLM="${TAU_USER_LLM:-openrouter/qwen/qwen3.6-27b}"' in launcher
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"' in launcher
    assert 'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"' in launcher
    assert "PPO_MAX_TOKENS_PER_GPU LOGPROB_MAX_TOKENS_PER_GPU" in launcher
    assert "${!budget_name} * SP_SIZE < MAX_MODEL_LEN" in launcher
    assert "$budget_name * SP_SIZE must be at least MAX_MODEL_LEN=$MAX_MODEL_LEN" in launcher
    assert "MAX_HISTORY_EXCHANGES=" in launcher
    assert 'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"' in launcher
    assert "MAX_PROMPT_LENGTH=$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))" in launcher
    assert 'data.max_prompt_length="$MAX_PROMPT_LENGTH"' in launcher
    assert 'data.max_response_length="$MAX_RESPONSE_LENGTH"' in launcher
    assert 'actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"' in launcher
    assert "env.context.max_history_exchanges=" in launcher
    assert '"$TAU_USER_LLM" == openrouter/*' in launcher
    assert '"env.tau.user_llm=$TAU_USER_LLM"' in launcher
    assert 'AWM_SERVER_LOG="$RUN_DIR/awm_server.log"' in launcher
    assert 'AWM_SERVER_RUN_ID="$(basename "$RUN_DIR")-$$"' in launcher
    assert "setsid env \\" in launcher
    assert "trap stop_managed_awm_server EXIT" in launcher
    assert '--expected-run-id "$AWM_SERVER_RUN_ID"' in launcher
    assert '"$RUN_DIR/awm_server_manifest.json"' in launcher
    assert '"terminal_judge": server_protocol["terminal_judge"]' in launcher
    assert "export AWM_DATA_DIR TAU2_DATA_DIR TENSORBOARD_DIR" in launcher
    assert 'trainer.save_before_validation="$SAVE_BEFORE_VALIDATION"' in launcher
    assert "pd.read_parquet(sys.argv[1])" in launcher
    assert "pd.read_parquet('$TRAIN_FILE')" not in launcher
    assert '    "$@" 2>&1 | tee "$RUN_DIR/train.log"' in launcher
    assert '"env.awm.runtime_failures.path=$RUN_DIR/runtime_failures.jsonl"' in launcher
    assert '"env.awm.runtime_failures.judge.data_dir=$AWM_DATA_DIR"' in launcher
    assert '"env.awm.runtime_failures.judge.reference_trials_path=$RUNTIME_JUDGE_REFERENCE_TRIALS"' in launcher
    assert '"env.awm.runtime_failures.judge.cache_path=$RUNTIME_JUDGE_CACHE_PATH"' in launcher
    assert 'manifest.get("trials_sha256")' not in launcher
    assert "hashlib.file_digest" not in launcher
    assert 'TERMINAL_JUDGE_MODEL="${TERMINAL_JUDGE_MODEL:-deepseek-v4-flash}"' in launcher
    assert "env.awm.verifier_mode=sql" in launcher
    assert 'env.awm.terminal_judge.model="$TERMINAL_JUDGE_MODEL"' in launcher
    assert '"env.teacher_reward.mode=$TEACHER_REWARD_MODE"' in launcher
    assert '"env.teacher_reward.frequency_bonus_scale=$FREQUENCY_BONUS_SCALE"' in launcher
    assert 'algorithm.state_group.advantage_mode="$STATE_GROUP_ADVANTAGE_MODE"' in launcher
    assert 'algorithm.state_group.min_effective_groups="$MIN_EFFECTIVE_STATE_GROUPS"' in launcher
    assert 'algorithm.state_group.compact_policy_rows="$COMPACT_STATE_GROUP_ROWS"' in launcher
    assert 'algorithm.state_group.diagnostic_only="$STATE_GROUP_DIAGNOSTIC_ONLY_HYDRA"' in launcher
    assert '--expected-terminal-model "$TERMINAL_JUDGE_MODEL"' in launcher


def test_awm_server_exposes_run_identity():
    root = Path(__file__).parents[2]
    server = (root / "agent_system/environments/env_package/awm/runtime/server.py").read_text(encoding="utf-8")
    checker = (root / "examples/awm/runtime/check_server.py").read_text(encoding="utf-8")

    assert 'RUN_ID = os.environ.get("AWM_SERVER_RUN_ID", "standalone")' in server
    assert '@app.get("/awm-run-identity", tags=["protocol"])' in server
    assert 'return {"run_id": RUN_ID}' in server
    assert '@app.get("/awm-terminal-judge", tags=["protocol"])' in server
    assert 'parser.add_argument("--expected-run-id")' in checker
    assert 'parser.add_argument("--expected-terminal-model")' in checker
    assert "require_server_run_id(args.base_url, args.expected_run_id, args.timeout)" in checker
