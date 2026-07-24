"""Shared action parsers for VPR environments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_BOXED_START_RE = re.compile(r"\\boxed\s*\{", re.IGNORECASE)
SUPPORTED_ACTION_FORMATS = ("action_tag", "boxed")

_ALIASES: dict[str, str] = {
    "open": "reveal",
    "click": "reveal",
    "mark": "flag",
}


@dataclass
class ParseResult:
    raw_action: str
    action_text: Optional[str]
    parse_ok: bool
    error: Optional[str]


def normalize_action_format(action_format: str) -> str:
    normalized = str(action_format).strip().lower()
    if normalized not in SUPPORTED_ACTION_FORMATS:
        raise ValueError(
            f"action_format must be one of {SUPPORTED_ACTION_FORMATS}, "
            f"got {action_format!r}"
        )
    return normalized


def _normalize_alias(action: str) -> str:
    lower = action.lower()
    for alias, canonical in _ALIASES.items():
        if lower.startswith(alias + " ") or lower == alias:
            return canonical + action[len(alias):]
    return action


def parse_action_tag(text: str) -> ParseResult:
    """Extract the last <action>...</action> block from model output.

    Never raises. Returns ParseResult with parse_ok=False on any failure.
    Expands known aliases before returning (e.g. 'open 1 2' -> 'reveal 1 2').
    """
    raw = str(text) if text is not None else ""
    matches = _ACTION_RE.findall(raw)

    if not matches:
        return ParseResult(raw_action=raw, action_text=None, parse_ok=False, error="no_action_tag")

    last = matches[-1].strip()
    if not last:
        return ParseResult(raw_action=raw, action_text=None, parse_ok=False, error="empty_action_tag")

    return ParseResult(
        raw_action=raw,
        action_text=_normalize_alias(last),
        parse_ok=True,
        error=None,
    )


def parse_boxed_action(text: str) -> ParseResult:
    """Extract the last complete ``\\boxed{...}`` block from model output."""
    raw = str(text) if text is not None else ""
    matches = []
    for match in _BOXED_START_RE.finditer(raw):
        content_start = match.end()
        depth = 1
        for index in range(content_start, len(raw)):
            char = raw[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    matches.append(raw[content_start:index])
                    break

    if not matches:
        return ParseResult(
            raw_action=raw,
            action_text=None,
            parse_ok=False,
            error="no_boxed_action",
        )

    last = matches[-1].strip()
    if not last:
        return ParseResult(
            raw_action=raw,
            action_text=None,
            parse_ok=False,
            error="empty_boxed_action",
        )

    return ParseResult(
        raw_action=raw,
        action_text=_normalize_alias(last),
        parse_ok=True,
        error=None,
    )


def parse_action(text: str, action_format: str = "action_tag") -> ParseResult:
    """Parse a model action using the configured strict output protocol."""
    normalized = normalize_action_format(action_format)
    if normalized == "action_tag":
        return parse_action_tag(text)
    return parse_boxed_action(text)
