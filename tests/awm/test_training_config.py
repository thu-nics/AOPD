from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

import agent_system.environments.env_package.awm.runtime.envs as awm_envs
from agent_system.environments.env_manager import _validate_awm_context_budget


def _compose(config_name):
    config_dir = Path(__file__).parents[2] / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=config_name)


def test_awm_disables_unused_entropy_computation():
    for config_name in ("awm_semantic", "awm_outcome"):
        config = _compose(config_name)
        actor = config.actor_rollout_ref.actor
        assert actor.entropy_coeff == 0.0
        assert actor.log_entropy_metrics is False
        assert config.data.return_raw_chat is True
        assert config.data.shuffle is False
        assert config.data.apply_chat_template_kwargs.enable_thinking is True
        assert config.data.max_prompt_length == 27904
        assert config.data.max_response_length == 4096
        assert config.data.max_prompt_length + config.data.max_response_length == 32000
        assert config.env.awm.history_window == 6
        assert config.actor_rollout_ref.rollout.n == 1
        assert config.actor_rollout_ref.rollout.multi_turn.enable is True
        validation = config.actor_rollout_ref.rollout.val_kwargs
        assert validation.do_sample is True
        assert validation.temperature == config.actor_rollout_ref.rollout.temperature == 0.6
        assert validation.top_p == config.actor_rollout_ref.rollout.top_p == 0.95
        assert validation.top_k == config.actor_rollout_ref.rollout.top_k == 20
        assert validation.n == 1
        assert validation.seed == config.env.awm.eval_seed == 300


def test_awm_context_budget_is_configurable_but_must_fit_model():
    config = _compose("awm_semantic")
    _validate_awm_context_budget(config)

    config.data.max_prompt_length = 28672
    config.data.max_response_length = 4096
    config.actor_rollout_ref.rollout.max_model_len = 32768
    _validate_awm_context_budget(config)

    config.data.max_prompt_length = 28673
    with pytest.raises(ValueError, match="AWM context budget requires"):
        _validate_awm_context_budget(config)


def test_formal_semantic_config_uses_tau_airline_validation():
    config = _compose("awm_semantic")
    assert config.data.train_batch_size == 64
    assert config.data.val_batch_size == 16
    assert config.env.rollout.n == 4
    assert config.env.validation.env_name == "tau"
    assert list(config.env.tau.validation_domains) == ["airline"]
    assert dict(config.env.tau.validation_counts) == {"airline": 16, "retail": 0}
    assert config.env.tau.validation_task_split == "base"
    assert config.env.tau.validation_trials == 1


@pytest.mark.parametrize("history_window", [0, 3, 6, 10])
def test_awm_worker_accepts_configurable_history_window(history_window):
    worker_class = awm_envs.AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        history_window=history_window,
        verifier_mode="code",
        reward_mode="semantic",
    )

    assert worker.history_window == history_window


def test_awm_worker_rejects_negative_history_window():
    worker_class = awm_envs.AWMWorker.__ray_metadata__.modified_class
    with pytest.raises(ValueError, match="history_window must be non-negative"):
        worker_class(
            base_url="unused",
            max_steps=20,
            history_window=-1,
            verifier_mode="code",
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
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
        awm=SimpleNamespace(
            train_max_steps=20,
            eval_max_steps=20,
            reward_mode="semantic",
            runtime_failures=None,
            base_url="http://127.0.0.1:8000",
            history_window=6,
            verifier_mode="sql_then_code_judge",
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


def test_training_launcher_scopes_artifacts_and_forwards_overrides():
    launcher = (Path(__file__).parents[2] / "examples" / "awm" / "train" / "run_training.sh").read_text(encoding="utf-8")

    assert 'RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"' in launcher
    assert 'RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_STAMP}"' in launcher
    assert 'TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"' in launcher
    assert 'VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"' in launcher
    assert 'trainer.val_before_train="$VAL_BEFORE_TRAIN"' in launcher
    assert 'TRAIN_TASK_COUNT="${TRAIN_TASK_COUNT:-}"' in launcher
    assert 'TRAIN_TASK_FRACTION="${TRAIN_TASK_FRACTION:-}"' in launcher
    assert 'FINAL_POOL_DIR="${FINAL_POOL_DIR:-$REPO_ROOT/runs/awm_final_pool}"' in launcher
    assert 'TRAIN_SELECTION_MANIFEST="$FINAL_POOL_DIR/final_manifest.json"' in launcher
    assert '"$SCRIPT_DIR/../data/slice_training_pool.py"' in launcher
    assert '"$SCRIPT_DIR/../data/materialize_training_schedule.py"' in launcher
    assert 'EXPERT_CACHE_DIR="${EXPERT_CACHE_DIR:-$RUN_DIR/cache}"' in launcher
    assert 'MANAGE_AWM_SERVER="${MANAGE_AWM_SERVER:-1}"' in launcher
    assert 'TAU_USER_LLM="${TAU_USER_LLM:-openrouter/qwen/qwen3.6-27b}"' in launcher
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"' in launcher
    assert 'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"' in launcher
    assert 'HISTORY_WINDOW="${HISTORY_WINDOW:-6}"' in launcher
    assert 'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"' in launcher
    assert "MAX_PROMPT_LENGTH=$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))" in launcher
    assert 'data.max_prompt_length="$MAX_PROMPT_LENGTH"' in launcher
    assert 'data.max_response_length="$MAX_RESPONSE_LENGTH"' in launcher
    assert 'actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"' in launcher
    assert 'env.awm.history_window="$HISTORY_WINDOW"' in launcher
    assert '"$TAU_USER_LLM" == openrouter/*' in launcher
    assert '"env.tau.user_llm=$TAU_USER_LLM"' in launcher
    assert 'AWM_SERVER_LOG="$RUN_DIR/awm_server.log"' in launcher
    assert 'AWM_SERVER_RUN_ID="$(basename "$RUN_DIR")-$$"' in launcher
    assert "setsid env \\" in launcher
    assert "trap stop_managed_awm_server EXIT" in launcher
    assert '--expected-run-id "$AWM_SERVER_RUN_ID"' in launcher
    assert '"$RUN_DIR/awm_server_manifest.json"' in launcher
    assert "export AWM_DATA_DIR TAU2_DATA_DIR TENSORBOARD_DIR" in launcher
    assert 'trainer.save_before_validation="$SAVE_BEFORE_VALIDATION"' in launcher
    assert "pd.read_parquet(sys.argv[1])" in launcher
    assert "pd.read_parquet('$TRAIN_FILE')" not in launcher
    assert '    "$@" 2>&1 | tee "$RUN_DIR/train.log"' in launcher
    assert 'env.awm.runtime_failures.path="$RUN_DIR/runtime_failures.jsonl"' in launcher


def test_awm_server_exposes_run_identity():
    root = Path(__file__).parents[2]
    server = (root / "agent_system/environments/env_package/awm/runtime/server.py").read_text(encoding="utf-8")
    checker = (root / "examples/awm/runtime/check_server.py").read_text(encoding="utf-8")

    assert 'RUN_ID = os.environ.get("AWM_SERVER_RUN_ID", "standalone")' in server
    assert '@app.get("/awm-run-identity", tags=["protocol"])' in server
    assert 'return {"run_id": RUN_ID}' in server
    assert 'parser.add_argument("--expected-run-id")' in checker
    assert "require_server_run_id(args.base_url, args.expected_run_id, args.timeout)" in checker
