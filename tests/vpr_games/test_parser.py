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
    return mod.parse_action_tag, mod.ParseResult


parse_action_tag, ParseResult = load_parser()


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
