import asyncio

from agent_system.environments.env_package.awm.runtime.actions import (
    AWMAction,
    build_native_chat,
    normalize_tools,
)
from agent_system.environments.env_package.awm.runtime.envs import AWMWorker


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Oracle:
    def __init__(self, *, samples=None, sample_error=None, matcher_error=None):
        def sample(**kwargs):
            if sample_error is not None:
                raise sample_error
            return samples

        def match(teacher_messages, candidate_messages):
            if matcher_error is not None:
                raise matcher_error
            return {
                "counts": [0] * len(candidate_messages),
                "matrix": [[False] * len(teacher_messages) for _ in candidate_messages],
            }

        self.sample_multiset = _RemoteMethod(sample)
        self.match_message_pairs = _RemoteMethod(match)


def _worker(oracle):
    worker_class = AWMWorker.__ray_metadata__.modified_class
    worker = worker_class(
        base_url="unused",
        max_steps=20,
        history_window=3,
        verifier_mode="sql",
        reward_mode="semantic",
        oracle_actor=oracle,
    )
    worker._scenario = "scenario"
    worker._task_idx = 0
    worker._task = "Finish the task"
    worker._tools = normalize_tools(
        [
            {
                "name": "lookup",
                "description": "Lookup one record.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}},
                    "required": ["item_id"],
                },
            }
        ]
    )
    worker._chat = build_native_chat(worker._task)
    return worker


def test_teacher_failure_does_not_generate_or_advance_state():
    worker = _worker(_Oracle(sample_error=RuntimeError("API unavailable")))
    original_chat = list(worker._chat)

    ready, info = asyncio.run(worker.prepare_state_group())

    assert ready is False
    assert info["teacher_failure"] is True
    assert info["state_group_advanced"] is False
    assert worker._step == 0
    assert worker._chat == original_chat
    assert worker._prepared_supervision is None


def test_matcher_failure_happens_before_environment_advancement():
    message = AWMAction(kind="message", content="The task is complete.")
    samples = [{"sample_index": index, "action": message.to_dict()} for index in range(3)]
    worker = _worker(
        _Oracle(
            samples=samples,
            matcher_error=RuntimeError("matcher unavailable"),
        )
    )
    original_chat = list(worker._chat)
    ready, _ = asyncio.run(worker.prepare_state_group())
    assert ready is True

    result = asyncio.run(worker.step_candidate_group(["candidate"] * 4))
    candidate_results, selected_index, _, reward, rollout_done, failure_info = result

    assert selected_index == -1
    assert reward == 0.0
    assert rollout_done is True
    assert failure_info["matcher_failure"] is True
    assert failure_info["state_group_advanced"] is False
    assert all(item[3]["semantic_train_mask"] is False for item in candidate_results)
    assert all(item[3]["state_group_advanced"] is False for item in candidate_results)
    assert worker._step == 0
    assert worker._chat == original_chat


def test_only_selected_candidate_carries_terminal_judge_metadata():
    action = AWMAction(kind="tool", name="lookup", arguments={"item_id": 1})
    samples = [{"sample_index": index, "action": action.to_dict()} for index in range(3)]
    worker = _worker(_Oracle(samples=samples))
    ready, _ = asyncio.run(worker.prepare_state_group())
    assert ready is True

    async def terminate(_raw_action, _action):
        worker._last_info = {
            "protocol_reward": 1.0,
            "terminal_success": True,
            "terminal_reason": "final_response",
            "awm_reward_type": "complete",
            "awm_verify_result": {"result": "ok"},
            "terminal_label": "complete",
            "terminal_reward": 1.0,
            "terminal_outcome_valid": True,
            "outcome_train_mask": True,
            "terminal_judge_result": {"classification": "complete"},
            "terminal_judge_error": None,
            "runtime_train_mask": True,
            "runtime_failure": False,
            "runtime_policy_error": False,
        }
        worker._done = True
        return 1.0, True

    worker._execute = terminate
    raw = '<tool_call>{"name":"lookup","arguments":{"item_id":1}}</tool_call>'
    candidate_results, selected_index, *_ = asyncio.run(worker.step_candidate_group([raw] * 4))

    for index, (_, _, done, info) in enumerate(candidate_results):
        if index == selected_index:
            assert done is True
            assert info["terminal_label"] == "complete"
            assert info["terminal_judge_result"] == {"classification": "complete"}
        else:
            assert done is False
            assert info["terminal_label"] is None
            assert info["terminal_reward"] is None
            assert info["terminal_outcome_valid"] is False
            assert info["terminal_judge_result"] is None
