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


def test_base_trainer_preserves_entropy_metric_default():
    actor = _compose("ppo_trainer").actor_rollout_ref.actor
    assert actor.log_entropy_metrics is True
