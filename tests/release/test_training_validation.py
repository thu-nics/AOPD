from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs


@pytest.mark.parametrize("enabled", [False, True])
def test_tau_validation_is_lazy_and_uses_its_own_user_without_oracle(monkeypatch, enabled):
    from agent_system.environments.env_package.tau_bench import envs, manager

    config = OmegaConf.create(
        {
            "trainer": {"val_before_train": enabled, "test_freq": -1},
            "data": {"train_batch_size": 1, "val_batch_size": 1},
            "env": {
                "env_name": "tau_outcome",
                "seed": 0,
                "teacher_reward": {"mode": "appearance", "frequency_bonus_scale": 0.5},
                "rollout": {"mode": "vanilla", "n": 4},
                "tau": {"source_root": "unused", "user_llm": "train-user", "eval_seed": 300, "trajectory_counts": {"airline": 1}, "validation_counts": {"airline": 1}, "validation_user": {"user_llm": "val-user"}},
            },
        }
    )
    calls = []
    monkeypatch.setattr(envs, "validate_tau_source", lambda *_: None)
    monkeypatch.setattr(envs, "validate_tau_runtime_config", lambda *_, **__: None)

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(envs, "build_tau_bench_envs", build)
    monkeypatch.setattr(manager, "TauBenchEnvironmentManager", lambda vector, projection, config, **kw: SimpleNamespace(config=config))
    train, val = make_envs(config)
    assert calls[0]["env_config"].tau.user_llm == "train-user"
    assert len(calls) == 1 + enabled
    if enabled:
        assert calls[1]["env_config"].tau.user_llm == "val-user"
        assert calls[1]["oracle_actor"] is None
        assert calls[1]["group_n"] == 1
        assert val.config.env.tau.user_llm == "val-user"
    else:
        assert val is None
    assert train.config.env.tau.user_llm == "train-user"
