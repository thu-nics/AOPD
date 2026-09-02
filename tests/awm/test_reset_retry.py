import asyncio
import math

import pytest

from agent_system.environments.env_package.awm.runtime.envs import AWMWorker


class _FakeEnv:
    def __init__(self, *, reset_result=None, reset_error=None, tools=None, tools_error=None):
        self.reset_result = reset_result
        self.reset_error = reset_error
        self.tools = [] if tools is None else tools
        self.tools_error = tools_error
        self.closed = False

    async def reset(self, **kwargs):
        if self.reset_error is not None:
            raise self.reset_error
        return self.reset_result

    async def list_tools(self, *, use_cache):
        if self.tools_error is not None:
            raise self.tools_error
        return self.tools

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True


def _worker(*, reset_max_retries=2, reset_retry_backoff_seconds=0.0):
    worker_class = AWMWorker.__ray_metadata__.modified_class
    return worker_class(
        base_url="unused",
        max_steps=20,
        verifier_mode="sql",
        reward_mode="semantic",
        reset_max_retries=reset_max_retries,
        reset_retry_backoff_seconds=reset_retry_backoff_seconds,
    )


def test_reset_recreates_environment_after_transient_connection_failure():
    worker = _worker()
    first = _FakeEnv(reset_error=ConnectionResetError("connection reset by peer"))
    second = _FakeEnv(
        reset_result={
            "reward_type": "reset_ok",
            "scenario": "demo",
            "task_idx": 3,
            "task": "Complete the task",
        }
    )
    environments = iter([first, second])

    async def new_env():
        return next(environments)

    worker._new_env = new_env
    observation, info = asyncio.run(worker.reset(scenario="demo", task_idx=3, seed=17))

    assert observation == "Complete the task"
    assert info["reset_retry_count"] == 1
    assert first.closed is True
    assert second.closed is False


def test_reset_recreates_environment_after_mcp_tool_connection_failure():
    worker = _worker()
    first = _FakeEnv(
        reset_result={"reward_type": "reset_ok", "task": "Complete the task"},
        tools_error=RuntimeError("Server error: All connection attempts failed"),
    )
    second = _FakeEnv(reset_result={"reward_type": "reset_ok", "task": "Complete the task"})
    environments = iter([first, second])

    async def new_env():
        return next(environments)

    worker._new_env = new_env
    _, info = asyncio.run(worker.reset(scenario="demo", task_idx=3, seed=17))

    assert info["reset_retry_count"] == 1
    assert first.closed is True
    assert second.closed is False


def test_reset_raises_after_bounded_number_of_transient_failures():
    worker = _worker(reset_max_retries=2)
    created = []

    async def new_env():
        env = _FakeEnv(reset_error=ConnectionResetError("connection reset by peer"))
        created.append(env)
        return env

    worker._new_env = new_env
    with pytest.raises(ConnectionResetError, match="connection reset by peer"):
        asyncio.run(worker.reset(scenario="demo", task_idx=3, seed=17))

    assert len(created) == 3
    assert all(env.closed for env in created)


def test_reset_does_not_retry_deterministic_reset_response():
    worker = _worker()
    failed = _FakeEnv(
        reset_result={
            "reward_type": "server_error",
            "error": "unknown task id",
        }
    )
    created = 0

    async def new_env():
        nonlocal created
        created += 1
        return failed

    worker._new_env = new_env
    with pytest.raises(RuntimeError, match="unknown task id"):
        asyncio.run(worker.reset(scenario="demo", task_idx=999, seed=17))

    assert created == 1
    assert failed.closed is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reset_max_retries": -1}, "reset_max_retries"),
        ({"reset_retry_backoff_seconds": -0.1}, "reset_retry_backoff_seconds"),
        ({"reset_retry_backoff_seconds": math.inf}, "reset_retry_backoff_seconds"),
        ({"reset_retry_backoff_seconds": math.nan}, "reset_retry_backoff_seconds"),
    ],
)
def test_reset_retry_configuration_must_be_non_negative(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _worker(**kwargs)
