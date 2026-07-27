from tau2.data_model.tasks import (
    EvaluationCriteria,
    RewardType,
    Task,
    UserScenario,
)

from examples.tau_bench import deterministic_evaluator


def make_task(reward_basis):
    return Task(
        id="task-1",
        user_scenario=UserScenario(instructions="Help me."),
        evaluation_criteria=EvaluationCriteria(
            reward_basis=reward_basis,
            nl_assertions=["The agent was helpful."],
        ),
    )


def test_removes_only_nl_assertion_from_copied_reward_basis():
    task = make_task(
        [RewardType.DB, RewardType.NL_ASSERTION, RewardType.ACTION]
    )

    deterministic = deterministic_evaluator.task_without_nl_reward_basis(task)

    assert deterministic is not task
    assert deterministic.evaluation_criteria.reward_basis == [
        RewardType.DB,
        RewardType.ACTION,
    ]
    assert deterministic.evaluation_criteria.nl_assertions == [
        "The agent was helpful."
    ]
    assert RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis


def test_evaluator_delegates_with_deterministic_task(monkeypatch):
    task = make_task([RewardType.DB, RewardType.NL_ASSERTION])
    captured = {}
    sentinel = object()

    def fake_evaluate_simulation(*, task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        deterministic_evaluator,
        "_tau_evaluate_simulation",
        fake_evaluate_simulation,
    )
    result = deterministic_evaluator.evaluate_simulation_without_nl_assertions(
        task=task,
        simulation="simulation",
        evaluation_type="all",
        solo_mode=False,
        domain="retail",
    )

    assert result is sentinel
    assert captured["task"].evaluation_criteria.reward_basis == [RewardType.DB]
    assert captured["kwargs"] == {
        "simulation": "simulation",
        "evaluation_type": "all",
        "solo_mode": False,
        "domain": "retail",
    }


def test_task_without_nl_basis_is_reused_unchanged():
    task = make_task([RewardType.ENV_ASSERTION, RewardType.ACTION])
    assert deterministic_evaluator.task_without_nl_reward_basis(task) is task
