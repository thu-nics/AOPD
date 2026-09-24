import pytest
from omegaconf import OmegaConf

from agent_system.environments.env_manager import _validate_mixed_decision_budget


def config(smoke=False, awm=20, envscaler=40, steps=1):
    return OmegaConf.create({"env": {"agentic_mix": {"bounded_smoke": smoke}, "awm": {"train_max_steps": awm}, "envscaler": {"train_max_steps": envscaler}, "max_steps": max(awm, envscaler)}, "trainer": {"total_training_steps": steps}})


def test_formal_budget_stays_20_and_40():
    _validate_mixed_decision_budget(config())
    with pytest.raises(ValueError):
        _validate_mixed_decision_budget(config(awm=2, envscaler=2))


def test_explicit_smoke_is_one_update_two_decisions():
    _validate_mixed_decision_budget(config(smoke=True, awm=2, envscaler=2))
    for cfg in (config(smoke=True), config(smoke=True, awm=2, envscaler=2, steps=2)):
        with pytest.raises(ValueError):
            _validate_mixed_decision_budget(cfg)
