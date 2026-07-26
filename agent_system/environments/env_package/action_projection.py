"""Action extraction helpers for benchmark-native agent evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable


_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_ACTION_LINE_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_PREFIX_RE = re.compile(r"^\s*Action\s*:\s*", re.IGNORECASE)


def _boxed_candidates(text: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return candidates

        content_start = start + len(marker)
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append((start, text[content_start:index].strip()))
                    cursor = index + 1
                    break
        else:
            return candidates


def extract_native_action(text: str) -> str | None:
    """Extract the last explicitly wrapped action, or the final non-empty line."""
    raw = str(text or "")
    candidates: list[tuple[int, str]] = []
    candidates.extend(
        (match.start(), match.group(1).strip())
        for match in _ACTION_TAG_RE.finditer(raw)
    )
    candidates.extend(
        (match.start(), match.group(1).strip())
        for match in _ACTION_LINE_RE.finditer(raw)
    )
    candidates.extend(_boxed_candidates(raw))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1] or None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else None


def canonicalize_admissible_action(
    action: str | None,
    admissible_actions: Iterable[str],
    *,
    allow_search_query: bool = False,
) -> str | None:
    """Return the environment's canonical spelling for an admissible action."""
    if action is None:
        return None

    candidate = _ACTION_PREFIX_RE.sub("", action.strip(), count=1)
    candidate_folded = candidate.casefold()
    pool = [str(item) for item in admissible_actions]
    for admissible in pool:
        if candidate_folded == admissible.strip().casefold():
            return admissible

    if allow_search_query and any(
        admissible.strip().casefold() == "search[<your query>]"
        for admissible in pool
    ):
        if candidate_folded.startswith("search[") and candidate.endswith("]"):
            query = candidate[len("search[") : -1].strip()
            if query and query.casefold() != "<your query>":
                return f"search[{query}]"

    return None


def project_native_actions(
    outputs: list[str],
    action_pools: list[Iterable[str]],
    *,
    allow_search_query: bool = False,
    fallback_chars: int = 30,
) -> tuple[list[str], list[int]]:
    """Project wrapper-relaxed model outputs onto strict environment actions."""
    if len(outputs) != len(action_pools):
        raise ValueError(
            f"Expected {len(outputs)} action pools, got {len(action_pools)}"
        )

    actions: list[str] = []
    valids: list[int] = []
    for output, action_pool in zip(outputs, action_pools):
        extracted = extract_native_action(output)
        canonical = canonicalize_admissible_action(
            extracted,
            action_pool,
            allow_search_query=allow_search_query,
        )
        if canonical is None:
            actions.append(str(extracted or output)[-fallback_chars:].lower())
            valids.append(0)
        else:
            actions.append(canonical)
            valids.append(1)
    return actions, valids
