from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2]


def test_training_launcher_requires_runtime_identity_and_keeps_protocol_defaults():
    launcher = (ROOT / "examples/tau_bench/train/run.sh").read_text()
    assert 'METHOD="${METHOD:-agentic_opd}"' in launcher
    assert 'MODEL_PATH="${MODEL_PATH:-}"' in launcher
    assert 'TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/../tau2-bench}"' in launcher
    assert 'AIRLINE_TRAJ="${AIRLINE_TRAJ:-5}"' in launcher
    assert 'RETAIL_TRAJ="${RETAIL_TRAJ:-11}"' in launcher
    assert 'TEST_FREQ="${TEST_FREQ:--1}"' in launcher
    assert 'VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"' in launcher
    assert 'TAU_USER_API_BASE="${TAU_USER_API_BASE:-}"' in launcher
    assert 'TAU_NATIVE_LOG_LEVEL="${TAU_NATIVE_LOG_LEVEL:-WARNING}"' in launcher
    assert 'env.tau.native_log_level="$TAU_NATIVE_LOG_LEVEL"' in launcher
    assert 'TAU_TEACHER_API_BASE="${TAU_TEACHER_API_BASE:-}"' in launcher
    assert 'ORACLE_MATCHER_CACHE="${ORACLE_MATCHER_CACHE:-$RUN_DIR/cache/matcher.jsonl}"' in launcher
    assert 'TAU_TEACHER_VALIDITY_MAX_RETRIES="${TAU_TEACHER_VALIDITY_MAX_RETRIES:-2}"' in launcher
    assert "env.tau.oracle.matcher_cache_path=$ORACLE_MATCHER_CACHE" in launcher
    assert "env.tau.oracle.teacher_validity_max_retries=$TAU_TEACHER_VALIDITY_MAX_RETRIES" in launcher
    assert 'WARMUP_STEPS="${WARMUP_STEPS:-0}"' in launcher
    assert '"reward_model.reward_manager=turn"' in launcher
    assert '"reward_model.overlong_buffer.enable=False"' in launcher
    assert '"actor_rollout_ref.actor.optim.weight_decay=0.01"' in launcher
    assert '"actor_rollout_ref.actor.clip_ratio_high=0.2"' in launcher
    assert '"actor_rollout_ref.actor.clip_ratio_c=3.0"' in launcher
    assert '"actor_rollout_ref.actor.log_entropy_metrics=False"' in launcher
    assert '"actor_rollout_ref.actor.log_sampled_entropy_metrics=True"' in launcher
    assert "DEFAULT_DATA_TRUNCATION=left" in launcher
    assert 'if [[ "$METHOD" == "agentic_opd" ]]' in launcher
    assert "DEFAULT_DATA_TRUNCATION=error" in launcher
    assert 'data.truncation="$DATA_TRUNCATION"' in launcher
    assert "dapo_turn" not in launcher
    assert "Set MODEL_PATH" in launcher
    assert "Set TAU_USER_API_BASE" in launcher
    assert "Set TAU_TEACHER_API_BASE" in launcher
    assert "/mnt/public2/yuanhuining" not in launcher
    assert "172.27." not in launcher
    assert "OPENROUTER_API_KEY" not in launcher
    assert "qualification" not in launcher.lower()


def test_evaluation_launcher_has_portable_remote_and_local_interfaces():
    launcher = (ROOT / "examples/tau_bench/eval/run.sh").read_text()
    assert 'PYTHON="${PYTHON:-python}"' in launcher
    assert 'VLLM_BIN="${VLLM_BIN:-vllm}"' in launcher
    assert 'TAU_USER_MODEL="${TAU_USER_MODEL:-}"' in launcher
    assert 'TAU_USER_API_BASE="${TAU_USER_API_BASE:-}"' in launcher
    assert 'USER_MODEL_PATH="${USER_MODEL_PATH:-}"' in launcher
    assert "USER_MODEL_PATH is required for local user mode" in launcher
    assert launcher.count("setsid env -u VLLM_PORT") == 2
    assert "--disable-log-requests" not in launcher
    assert '"$process_state" == Z*' in launcher
    assert "/mnt/public2/yuanhuining" not in launcher
    assert "172.27." not in launcher


def test_training_launcher_has_validated_two_and_eight_gpu_profiles():
    launcher = (ROOT / "examples/tau_bench/train/run.sh").read_text()
    assert "2)" in launcher
    assert 'TP_SIZE="${TP_SIZE:-1}"' in launcher
    assert 'SP_SIZE="${SP_SIZE:-2}"' in launcher
    assert 'PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-16384}"' in launcher
    assert "8)" in launcher
    assert 'TP_SIZE="${TP_SIZE:-2}"' in launcher
    assert 'SP_SIZE="${SP_SIZE:-4}"' in launcher
    assert 'PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-8192}"' in launcher


def test_tau_configs_require_runtime_identity_and_share_official_schedule():
    for name in ("tau_agentic_opd", "tau_outcome"):
        config = OmegaConf.load(ROOT / f"verl/trainer/config/{name}.yaml")
        tau = config.env.tau
        assert (tau.trajectory_counts.airline, tau.trajectory_counts.retail) == (
            5,
            11,
        )
        assert tau.validation_task_split == "test"
        assert OmegaConf.is_missing(tau, "source_root")
        assert OmegaConf.is_missing(tau, "user_llm")
        assert OmegaConf.is_missing(tau, "user_api_base")
        assert tau.user_reasoning_enabled is True
        assert tau.user_generation_retries == 2
        assert tau.native_log_level == "WARNING"


def test_agentic_teacher_runtime_identity_is_required_but_sampling_is_fixed():
    config = OmegaConf.load(ROOT / "verl/trainer/config/tau_agentic_opd.yaml")
    oracle = config.env.tau.oracle
    assert OmegaConf.is_missing(oracle, "model")
    assert OmegaConf.is_missing(oracle, "api_base")
    assert oracle.samples == 3
    assert (oracle.temperature, oracle.top_p, oracle.top_k, oracle.min_p) == (
        0.6,
        0.95,
        20,
        0.0,
    )
    assert oracle.enable_thinking is True
    assert oracle.max_tokens == 8192
    assert OmegaConf.is_missing(oracle, "matcher_cache_path")
    assert oracle.teacher_validity_max_retries == 2
    assert config.reward_model.reward_manager == "turn"
    assert config.reward_model.overlong_buffer.enable is False
    assert config.actor_rollout_ref.actor.entropy_coeff == 0.0
    assert config.actor_rollout_ref.actor.log_entropy_metrics is False
    assert config.actor_rollout_ref.actor.log_sampled_entropy_metrics is True
