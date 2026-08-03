from pathlib import Path

from hydra import compose, initialize_config_dir


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
        assert config.actor_rollout_ref.rollout.n == 1
        assert config.actor_rollout_ref.rollout.multi_turn.enable is True
        validation = config.actor_rollout_ref.rollout.val_kwargs
        assert validation.do_sample is True
        assert validation.temperature == config.actor_rollout_ref.rollout.temperature == 0.6
        assert validation.top_p == config.actor_rollout_ref.rollout.top_p == 0.95
        assert validation.top_k == config.actor_rollout_ref.rollout.top_k == 20
        assert validation.n == 1
        assert validation.seed == config.env.awm.eval_seed == 300


def test_base_trainer_preserves_entropy_metric_default():
    actor = _compose("ppo_trainer").actor_rollout_ref.actor
    assert actor.log_entropy_metrics is True


def test_training_launcher_scopes_artifacts_and_forwards_overrides():
    launcher = (Path(__file__).parents[2] / "examples" / "awm" / "scripts" / "run_training.sh").read_text(encoding="utf-8")

    assert 'RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"' in launcher
    assert 'RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_STAMP}"' in launcher
    assert 'TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"' in launcher
    assert 'VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"' in launcher
    assert 'trainer.val_before_train="$VAL_BEFORE_TRAIN"' in launcher
    assert 'TRAIN_TASK_COUNT="${TRAIN_TASK_COUNT:-}"' in launcher
    assert 'TRAIN_TASK_FRACTION="${TRAIN_TASK_FRACTION:-}"' in launcher
    assert '"$SCRIPT_DIR/../cli/slice_training_pool.py"' in launcher
    assert "export AWM_DATA_DIR TENSORBOARD_DIR" in launcher
    assert '    "$@" 2>&1 | tee "$RUN_DIR/train.log"' in launcher
