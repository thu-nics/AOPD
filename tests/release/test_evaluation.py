from pathlib import Path

from aopd.evaluate import build_eval_plan


def test_tau_eval_local_user_can_use_two_noncontiguous_gpus(tmp_path):
    runtime = {
        "model": "/models/qwen",
        "student": {"gpus": [1, 2, 4, 5, 6, 7], "tp": 1},
        "sources": {"tau": "/deps/tau"},
        "services": {"user": {"mode": "local", "model": "user", "model_path": "/models/user", "gpus": [0, 3], "tp": 2, "port": 8201}},
        "roles": {"user": {"service": "user", "generation": {"enable_thinking": True}}},
    }
    plan = build_eval_plan(runtime, tmp_path, smoke=False)
    assert len(plan["commands"]) == 3
    assert plan["agent_service"]["dp"] == 6
    assert plan["agent_service"]["gpus"] == [1, 2, 4, 5, 6, 7]
    for command in plan["commands"]:
        assert command[command.index("--task-split") + 1] == "base"
        assert command[command.index("--num-trials") + 1] == "4"
        assert command[command.index("--agent-temperature") + 1] == "0.0"
        assert "--user-api-key" not in command
        assert Path(command[1]).name == "native_eval.py"


def test_eval_remote_user_has_no_local_gpu_reservation(tmp_path):
    runtime = {"model": "/model", "student": {"gpus": [0, 1], "tp": 1}, "services": {"user": {"mode": "api", "base_url": "https://example.org/v1", "model": "x"}}, "roles": {"user": {"service": "user"}}}
    plan = build_eval_plan(runtime, tmp_path, smoke=True)
    assert plan["agent_service"]["gpus"] == [0, 1]
    assert "--num-tasks" in plan["commands"][0]
