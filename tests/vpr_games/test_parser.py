"""Unit tests for the shared VPR action tag parser."""

import pytest
import sys
import importlib.util

def load_parser():
    spec = importlib.util.spec_from_file_location(
        "vpr_parser",
        "agent_system/environments/env_package/vpr_games/common/parser.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vpr_parser"] = mod
    spec.loader.exec_module(mod)
    return (
        mod.parse_action_tag,
        mod.parse_boxed_action,
        mod.parse_action,
        mod.normalize_action_format,
        mod.ParseResult,
    )


parse_action_tag, parse_boxed_action, parse_action, normalize_action_format, ParseResult = load_parser()


def test_basic_parse():
    r = parse_action_tag("<action>3</action>")
    assert r.parse_ok and r.action_text == "3"


def test_whitespace_tolerated():
    r = parse_action_tag("<think>ok</think><action> 3 </action>")
    assert r.parse_ok and r.action_text == "3"


def test_repeated_tags_last_wins():
    r = parse_action_tag("<action>5</action>...<action>7</action>")
    assert r.parse_ok and r.action_text == "7"


def test_empty_input():
    r = parse_action_tag("")
    assert not r.parse_ok and r.error == "no_action_tag"


def test_no_action_tag():
    r = parse_action_tag("some text without tags")
    assert not r.parse_ok and r.error == "no_action_tag"


def test_empty_action_tag():
    r = parse_action_tag("<action></action>")
    assert not r.parse_ok and r.error == "empty_action_tag"


def test_alias_open_to_reveal():
    r = parse_action_tag("<action>open 1 2</action>")
    assert r.parse_ok and r.action_text == "reveal 1 2"


def test_alias_click_to_reveal():
    r = parse_action_tag("<action>click 2 3</action>")
    assert r.parse_ok and r.action_text == "reveal 2 3"


def test_alias_mark_to_flag():
    r = parse_action_tag("<action>mark 3 4</action>")
    assert r.parse_ok and r.action_text == "flag 3 4"


def test_reveal_passthrough():
    r = parse_action_tag("<action>reveal 1 1</action>")
    assert r.parse_ok and r.action_text == "reveal 1 1"


def test_case_insensitive_tag():
    r = parse_action_tag("<ACTION>5</ACTION>")
    assert r.parse_ok and r.action_text == "5"


def test_none_input():
    r = parse_action_tag(None)
    assert not r.parse_ok and r.error == "no_action_tag"


def test_multiline_action():
    r = parse_action_tag("<action>\n5\n</action>")
    assert r.parse_ok and r.action_text == "5"


def test_boxed_action_parses_last_complete_block():
    r = parse_boxed_action(r"reasoning \boxed{left} correction \boxed{up}")
    assert r.parse_ok and r.action_text == "up"


def test_boxed_action_supports_balanced_inner_braces():
    r = parse_boxed_action(r"\boxed{reveal {2} 3}")
    assert r.parse_ok and r.action_text == "reveal {2} 3"


def test_boxed_action_normalizes_aliases():
    r = parse_boxed_action(r"\boxed{click 2 3}")
    assert r.parse_ok and r.action_text == "reveal 2 3"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", "no_boxed_action"),
        (r"\boxed{}", "empty_boxed_action"),
        (r"\boxed{up", "no_boxed_action"),
        ("<action>up</action>", "no_boxed_action"),
    ],
)
def test_boxed_action_rejects_invalid_or_other_protocol(text, error):
    r = parse_boxed_action(text)
    assert not r.parse_ok and r.error == error


def test_parse_action_dispatch_is_strict():
    assert parse_action(r"\boxed{up}", "boxed").action_text == "up"
    assert not parse_action("<action>up</action>", "boxed").parse_ok
    assert parse_action("<action>up</action>", "action_tag").action_text == "up"
    assert not parse_action(r"\boxed{up}", "action_tag").parse_ok


@pytest.mark.parametrize("value", ["", "xml", None, True])
def test_action_format_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="action_format"):
        normalize_action_format(value)


def test_boxed_game_prompts_match_parser_protocol():
    spec = importlib.util.spec_from_file_location(
        "vpr_prompt_templates",
        "agent_system/environments/prompts/vpr_games.py",
    )
    prompts = importlib.util.module_from_spec(spec)
    sys.modules["vpr_prompt_templates"] = prompts
    spec.loader.exec_module(prompts)

    for game in ("sokoban", "sudoku", "minesweeper"):
        boxed = prompts.get_vpr_game_template(game, "boxed")
        assert r"\boxed{{" in boxed
        assert "<action>" not in boxed
        assert "<action>" in prompts.get_vpr_game_template(game, "action_tag")
