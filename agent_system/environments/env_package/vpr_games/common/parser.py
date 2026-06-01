"""Shared action tag parser for VPR environments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

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

    lower = last.lower()
    for alias, canonical in _ALIASES.items():
        if lower.startswith(alias + " ") or lower == alias:
            rest = last[len(alias):]
            last = canonical + rest
            break

    return ParseResult(raw_action=raw, action_text=last, parse_ok=True, error=None)
