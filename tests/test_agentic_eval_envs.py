from types import SimpleNamespace

import pytest


class RemoteCall:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return len(self.calls) - 1


def test_alfworld_deterministic_eval_visits_each_game_once(monkeypatch):
    from agent_system.environments.env_package.alfworld import envs as module

    workers = [
        SimpleNamespace(reset=RemoteCall()),
        SimpleNamespace(reset=RemoteCall()),
    ]
    env = module.AlfworldEnvs.__new__(module.AlfworldEnvs)
    env.num_processes = 2
    env.deterministic_eval = True
    env.eval_game_files = ["game-0", "game-1", "game-2", "game-3"]
    env.eval_cursor = 0
    env.workers = workers
    env.multi_modal = False
    env.prev_admissible_commands = [None, None]

    def fake_get(_futures):
        return [
            (
                [f"observation-{index}"],
                {"admissible_commands": [["look"]]},
            )
            for index in range(2)
        ]

    monkeypatch.setattr(module.ray, "get", fake_get)
    env.reset()
    env.reset()

    assert workers[0].reset.calls == [("game-0",), ("game-2",)]
    assert workers[1].reset.calls == [("game-1",), ("game-3",)]
    assert env.eval_cursor == 4
    with pytest.raises(RuntimeError, match="exhausted its game list"):
        env.reset()


def test_webshop_deterministic_eval_visits_test_goals_in_order(monkeypatch):
    from agent_system.environments.env_package.webshop import envs as module

    workers = [
        SimpleNamespace(reset=RemoteCall()),
        SimpleNamespace(reset=RemoteCall()),
    ]
    env = module.WebshopMultiProcessEnv.__new__(module.WebshopMultiProcessEnv)
    env.env_num = 2
    env.group_n = 1
    env.deterministic_eval = True
    env.shared_server = False
    env.eval_cursor = 0
    env.goal_idxs = range(4)
    env._workers = workers

    monkeypatch.setattr(
        module.ray,
        "get",
        lambda _futures: [("observation-0", {}), ("observation-1", {})],
    )
    env.reset()
    env.reset()

    assert workers[0].reset.calls == [(0,), (2,)]
    assert workers[1].reset.calls == [(1,), (3,)]
    assert env.eval_cursor == 4
    with pytest.raises(RuntimeError, match="exhausted its test goals"):
        env.reset()
    env._closed = True


def test_webshop_shared_server_loads_products_once(monkeypatch):
    from agent_system.environments.env_package.webshop import envs as module

    created = []

    class FakeEnv:
        def __init__(self, server=None, **kwargs):
            self.server = server or SimpleNamespace(goals=list(range(500)))
            self.kwargs = kwargs
            self.sessions = []
            created.append(self)

        def reset(self, session=None):
            self.sessions.append(session)
            return f"observation-{session}", None

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": []}

        def close(self):
            pass

    fake_package = SimpleNamespace(WebAgentTextEnv=FakeEnv)
    monkeypatch.setitem(__import__("sys").modules, "web_agent_site.envs", fake_package)

    env = module.WebshopMultiProcessEnv(
        seed=1000,
        env_num=2,
        group_n=1,
        resources_per_worker={},
        is_train=False,
        env_kwargs={
            "deterministic_eval": True,
            "shared_server": True,
        },
    )

    assert len(created) == 2
    assert created[0].server is created[1].server
    obs, infos = env.reset()
    assert obs == ["observation-0", "observation-1"]
    assert all(info["available_actions"]["has_search_bar"] for info in infos)
    env.close()


@pytest.mark.parametrize(
    "output",
    [
        "Reasoning first.\nAction: go to microwave 1",
        "Reasoning first.\n<action>go to microwave 1</action>",
        r"Reasoning first.\n\boxed{go to microwave 1}",
        "go to microwave 1",
    ],
)
def test_alfworld_native_projection_is_wrapper_relaxed(output):
    from agent_system.environments.env_package.alfworld import alfworld_projection

    actions, valids = alfworld_projection(
        [output],
        [["go to microwave 1", "look"]],
        native_action_protocol=True,
    )

    assert actions == ["go to microwave 1"]
    assert valids == [1]


def test_alfworld_native_projection_is_action_strict():
    from agent_system.environments.env_package.alfworld import alfworld_projection

    _, valids = alfworld_projection(
        ["Action: open microwave 1"],
        [["go to microwave 1", "look"]],
        native_action_protocol=True,
    )

    assert valids == [0]


def test_alfworld_legacy_projection_still_requires_think_tags():
    from agent_system.environments.env_package.alfworld import alfworld_projection

    _, valids = alfworld_projection(
        ["<action>look</action>"],
        [["look"]],
    )

    assert valids == [0]


@pytest.mark.parametrize(
    ("output", "pool", "expected"),
    [
        ("Action: search[red shoes]", ["search[<your query>]"], "search[red shoes]"),
        (r"\boxed{click[Buy Now]}", ["click[Buy Now]"], "click[Buy Now]"),
        ("<action>click[buy now]</action>", ["click[Buy Now]"], "click[Buy Now]"),
    ],
)
def test_webshop_native_projection_accepts_only_available_actions(
    output, pool, expected
):
    from agent_system.environments.env_package.webshop import webshop_projection

    actions, valids = webshop_projection(
        [output],
        [pool],
        native_action_protocol=True,
    )

    assert actions == [expected]
    assert valids == [1]
