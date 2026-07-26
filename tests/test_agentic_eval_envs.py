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
        r"Reasoning first.\n\boxed{ACTION: go to microwave 1}",
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
        [r"\boxed{ACTION: open microwave 1}"],
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
        (
            r"\boxed{ACTION: click[Buy Now]}",
            ["click[Buy Now]"],
            "click[Buy Now]",
        ),
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


class _FakeChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append((chat, kwargs))
        return "rendered chat"

    def encode(self, prompt, add_special_tokens=False):
        assert add_special_tokens is False
        return list(prompt)


def test_agentic_prompt_rendering_supports_raw_and_chatml():
    from agent_system.multi_turn_rollout.rollout_loop import _render_agentic_prompt

    tokenizer = _FakeChatTokenizer()
    chat = [{"role": "user", "content": "raw prompt"}]

    assert _render_agentic_prompt(tokenizer, chat, "raw", {}) == "raw prompt"
    assert tokenizer.calls == []
    assert (
        _render_agentic_prompt(tokenizer, chat, "chatml", {"flag": True})
        == "rendered chat"
    )
    assert tokenizer.calls[0][1] == {
        "add_generation_prompt": True,
        "tokenize": False,
        "flag": True,
    }

    import numpy as np

    tool_schema = {"type": "function", "function": {"name": "lookup"}}
    ray_style_tools = np.empty(1, dtype=object)
    ray_style_tools[0] = tool_schema
    _render_agentic_prompt(tokenizer, chat, "chatml", {}, tools=ray_style_tools)
    assert tokenizer.calls[-1][1]["tools"] == [tool_schema]

    tau_tool = SimpleNamespace(openai_schema=tool_schema)
    _render_agentic_prompt(tokenizer, chat, "chatml", {}, tools=[tau_tool])
    assert tokenizer.calls[-1][1]["tools"] == [tool_schema]
    with pytest.raises(ValueError, match="Unsupported agentic prompt rendering"):
        _render_agentic_prompt(tokenizer, chat, "invalid", {})


def test_tau_prompt_budget_drops_old_complete_chunks_but_keeps_contract():
    import json

    from agent_system.multi_turn_rollout.rollout_loop import (
        _render_tau_prompt_with_budget,
    )

    class BudgetTokenizer(_FakeChatTokenizer):
        def apply_chat_template(self, chat, **kwargs):
            self.calls.append((chat, kwargs))
            return json.dumps(chat, sort_keys=True) + json.dumps(kwargs, sort_keys=True)

    tokenizer = BudgetTokenizer()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "greeting"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "old action"},
        {"role": "tool", "content": "old result", "tool_call_id": "old"},
        {"role": "assistant", "content": "latest action"},
        {"role": "tool", "content": "latest result", "tool_call_id": "latest"},
    ]
    minimal = [chat[0], chat[2], chat[5], chat[6]]
    max_tokens = len(
        tokenizer.apply_chat_template(
            minimal,
            add_generation_prompt=True,
            tokenize=False,
            tools=tools,
        )
    )
    tokenizer.calls.clear()

    prompt = _render_tau_prompt_with_budget(
        tokenizer,
        chat,
        {},
        tools=tools,
        max_prompt_tokens=max_tokens,
    )

    rendered_chat, kwargs = tokenizer.calls[-1]
    assert rendered_chat == minimal
    assert "old action" not in prompt
    assert "latest result" in prompt
    assert kwargs["tools"] == tools


def test_tau_prompt_budget_fails_instead_of_truncating_required_context():
    from agent_system.multi_turn_rollout.rollout_loop import (
        _render_tau_prompt_with_budget,
    )

    tokenizer = _FakeChatTokenizer()
    chat = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
    ]
    with pytest.raises(ValueError, match="do not fit"):
        _render_tau_prompt_with_budget(
            tokenizer,
            chat,
            {},
            tools=[],
            max_prompt_tokens=1,
        )


def test_native_action_prompts_are_stock_without_think_requirement():
    from agent_system.environments.prompts.alfworld import (
        ALFWORLD_TEMPLATE,
        ALFWORLD_TEMPLATE_NO_HIS,
        ALFWORLD_NATIVE_ACTION_TEMPLATE,
        ALFWORLD_NATIVE_ACTION_TEMPLATE_NO_HIS,
    )
    from agent_system.environments.prompts.webshop import (
        WEBSHOP_TEMPLATE,
        WEBSHOP_TEMPLATE_NO_HIS,
        WEBSHOP_NATIVE_ACTION_TEMPLATE,
        WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS,
    )

    requirement = " This reasoning process MUST be enclosed within <think> </think> tags."
    pairs = (
        (ALFWORLD_NATIVE_ACTION_TEMPLATE, ALFWORLD_TEMPLATE),
        (ALFWORLD_NATIVE_ACTION_TEMPLATE_NO_HIS, ALFWORLD_TEMPLATE_NO_HIS),
        (WEBSHOP_NATIVE_ACTION_TEMPLATE, WEBSHOP_TEMPLATE),
        (WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE_NO_HIS),
    )
    for action_template, stock_template in pairs:
        assert action_template == stock_template.replace(requirement, "")
        assert "<action> </action>" in action_template
        assert requirement.strip() not in action_template
        assert r"\boxed" not in action_template
        assert "format example" not in action_template.lower()


def test_agentic_prompt_selector_supports_both_native_formats():
    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import select_agentic_prompt_template

    config = OmegaConf.create(
        {
            "env": {
                "agentic_eval": {
                    "native_action_protocol": True,
                    "action_format": "action_tag",
                }
            }
        }
    )

    assert select_agentic_prompt_template(config, "tag", "box", "legacy") == "tag"
    config.env.agentic_eval.action_format = "boxed"
    assert select_agentic_prompt_template(config, "tag", "box", "legacy") == "box"
    config.env.agentic_eval.native_action_protocol = False
    assert select_agentic_prompt_template(config, "tag", "box", "legacy") == "legacy"
    config.env.agentic_eval.native_action_protocol = True
    config.env.agentic_eval.action_format = "invalid"
    with pytest.raises(ValueError, match="Unsupported agentic action format"):
        select_agentic_prompt_template(config, "tag", "box", "legacy")


def test_agentic_summary_keeps_action_formats_separate():
    from examples.vpr_games.eval.summarize_agentic_ood import aggregate

    rows = [
        {
            "model_id": "model",
            "action_format": "action_tag",
            "benchmark": "alfworld",
            "seed": "0",
            "success_rate": 0.25,
            "task_score": None,
            "task_rates": {"pick_and_place": 0.25},
        },
        {
            "model_id": "model",
            "action_format": "boxed",
            "benchmark": "alfworld",
            "seed": "0",
            "success_rate": 0.75,
            "task_score": None,
            "task_rates": {"pick_and_place": 0.75},
        },
    ]

    summaries, task_summaries = aggregate(rows)

    assert len(summaries) == 2
    by_format = {item["action_format"]: item for item in summaries}
    assert by_format["action_tag"]["success_rate_mean"] == 0.25
    assert by_format["boxed"]["success_rate_mean"] == 0.75
    assert len(task_summaries) == 2
    assert {item["action_format"] for item in task_summaries} == {
        "action_tag",
        "boxed",
    }


def test_agentic_summary_collects_non_persistent_dual_format_layout(tmp_path):
    import json

    from examples.vpr_games.eval.summarize_agentic_ood import collect_rows

    seed_dir = (
        tmp_path
        / "results"
        / "model"
        / "boxed"
        / "alfworld"
        / "seed_0"
    )
    raw_dir = seed_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "validation.metrics.json").write_text(
        json.dumps({"val/success_rate": 0.5}),
        encoding="utf-8",
    )
    (seed_dir / ".done").write_text("complete\n", encoding="utf-8")

    rows = collect_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["model_id"] == "model"
    assert rows[0]["action_format"] == "boxed"
    assert rows[0]["benchmark"] == "alfworld"
    assert rows[0]["success_rate"] == 0.5


def test_boxed_prompts_only_replace_the_action_wrapper():
    from agent_system.environments.prompts.alfworld import (
        ALFWORLD_NATIVE_ACTION_TEMPLATE,
        ALFWORLD_NATIVE_ACTION_TEMPLATE_NO_HIS,
        ALFWORLD_NATIVE_BOXED_TEMPLATE,
        ALFWORLD_NATIVE_BOXED_TEMPLATE_NO_HIS,
    )
    from agent_system.environments.prompts.webshop import (
        WEBSHOP_NATIVE_ACTION_TEMPLATE,
        WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS,
        WEBSHOP_NATIVE_BOXED_TEMPLATE,
        WEBSHOP_NATIVE_BOXED_TEMPLATE_NO_HIS,
    )
    pairs = (
        (ALFWORLD_NATIVE_ACTION_TEMPLATE, ALFWORLD_NATIVE_BOXED_TEMPLATE),
        (
            ALFWORLD_NATIVE_ACTION_TEMPLATE_NO_HIS,
            ALFWORLD_NATIVE_BOXED_TEMPLATE_NO_HIS,
        ),
        (WEBSHOP_NATIVE_ACTION_TEMPLATE, WEBSHOP_NATIVE_BOXED_TEMPLATE),
        (
            WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS,
            WEBSHOP_NATIVE_BOXED_TEMPLATE_NO_HIS,
        ),
    )
    for action_template, boxed_template in pairs:
        assert boxed_template == action_template.replace(
            "<action> </action>", r"\boxed{{ACTION}}"
        )
        assert r"\boxed{{ACTION}}" in boxed_template
        assert "<action> </action>" not in boxed_template
