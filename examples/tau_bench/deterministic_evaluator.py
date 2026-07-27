"""Deterministic Tau scoring that excludes experimental NL assertions."""

from __future__ import annotations

from typing import Any

from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator import evaluate_simulation as _tau_evaluate_simulation


EVALUATION_PROTOCOL = "tau_all_without_nl_assertions_v1"


def task_without_nl_reward_basis(task: Task) -> Task:
    """Copy a task with NL assertions removed only from its reward basis."""
    criteria = task.evaluation_criteria
    if criteria is None or RewardType.NL_ASSERTION not in criteria.reward_basis:
        return task

    deterministic_basis = [
        reward_type
        for reward_type in criteria.reward_basis
        if reward_type != RewardType.NL_ASSERTION
    ]
    deterministic_criteria = criteria.model_copy(
        update={"reward_basis": deterministic_basis}
    )
    return task.model_copy(
        update={"evaluation_criteria": deterministic_criteria}
    )


def evaluate_simulation_without_nl_assertions(
    *, task: Task, **kwargs: Any
):
    """Run Tau's evaluator after removing only NL assertions from scoring."""
    return _tau_evaluate_simulation(
        task=task_without_nl_reward_basis(task),
        **kwargs,
    )


def install_deterministic_evaluator() -> None:
    """Install the wrapper at Tau's simulation-runner call boundary."""
    from tau2.runner import simulation as runner_simulation

    runner_simulation.evaluate_simulation = (
        evaluate_simulation_without_nl_assertions
    )
