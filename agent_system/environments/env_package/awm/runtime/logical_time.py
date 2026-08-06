"""Deterministic logical clock for the pinned AgentWorldModel dataset."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import re
import tokenize
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

LOGICAL_TIME_PROTOCOL_VERSION = 2
DATASET_REVISION = "dde80a0283fe781bdc51656bce57063dc5650213"
FALLBACK_LOGICAL_TIME = datetime(2025, 1, 15, 12, 0, 0)
LOGICAL_TIME_ENDPOINT = "/awm-logical-time"

_INSERT_RE = re.compile(
    r"\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+[^()]+"
    r"\((?P<columns>.*?)\)\s*VALUES\s*\((?P<values>.*)\)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SQLITE_NOW_RE = re.compile(
    r"(?P<prefix>\b(?:datetime|date|time|julianday|unixepoch)\s*\(\s*)"
    r"(?P<quote>['\"])now(?P=quote)",
    re.IGNORECASE,
)
_SQLITE_STRFTIME_NOW_RE = re.compile(
    r"(?P<prefix>\bstrftime\s*\(\s*(?:'[^']*'|\"[^\"]*\")\s*,\s*)"
    r"(?P<quote>['\"])now(?P=quote)",
    re.IGNORECASE,
)
_DATETIME_CALL_RE = re.compile(
    r"(?<![\w.])(?P<prefix>datetime\.datetime|dt\.datetime|datetime)\."
    r"(?P<method>utcnow|now|today)\(\s*(?P<argument>[^()]*)\s*\)"
)
_DATE_CALL_RE = re.compile(r"(?<![\w.])(?P<prefix>datetime\.date|dt\.date|date)\.today\(\s*\)")
_ANCHOR_COLUMNS = frozenset(
    {
        "created_at",
        "updated_at",
        "created_on",
        "updated_on",
        "inserted_at",
        "modified_at",
    }
)


def normalize_scenario_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]", "_", str(value).lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _split_sql_csv(value: str) -> list[str]:
    output: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    index = 0
    while index < len(value):
        char = value[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 2
                    continue
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            output.append(value[start:index].strip())
            start = index + 1
        index += 1
    output.append(value[start:].strip())
    return output


def _iter_insert_statements(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "insert_statements" and isinstance(nested, list):
                for statement in nested:
                    if isinstance(statement, str):
                        yield statement
            else:
                yield from _iter_insert_statements(nested)
    elif isinstance(value, list):
        for nested in value:
            if isinstance(nested, str):
                yield nested
            else:
                yield from _iter_insert_statements(nested)


def _parse_sql_datetime(value: str) -> datetime | None:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in {"'", '"'}:
        return None
    if stripped[-1] != stripped[0]:
        return None
    literal = stripped[1:-1].replace(stripped[0] * 2, stripped[0]).strip()
    if not re.match(r"^20\d{2}-\d{2}-\d{2}(?:[ T].*)?$", literal):
        return None
    try:
        if len(literal) == 10:
            parsed = datetime.combine(date.fromisoformat(literal), datetime.min.time())
        else:
            parsed = datetime.fromisoformat(literal.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def infer_logical_time(sample_data: Any) -> tuple[datetime, str]:
    """Infer a snapshot clock from explicit record creation/update timestamps."""
    anchors: list[datetime] = []
    for statement in _iter_insert_statements(sample_data):
        match = _INSERT_RE.search(statement.strip())
        if match is None:
            continue
        columns = [item.strip().strip('`"[]').casefold() for item in _split_sql_csv(match.group("columns"))]
        values = _split_sql_csv(match.group("values"))
        if len(columns) != len(values):
            continue
        for column, value in zip(columns, values, strict=True):
            if column not in _ANCHOR_COLUMNS:
                continue
            parsed = _parse_sql_datetime(value)
            if parsed is not None:
                anchors.append(parsed)
    if anchors:
        return max(anchors) + timedelta(seconds=1), "sample_created_or_updated_at"
    return FALLBACK_LOGICAL_TIME, "dataset_fallback"


def _datetime_arguments(value: datetime) -> str:
    components = [value.year, value.month, value.day, value.hour, value.minute, value.second]
    if value.microsecond:
        components.append(value.microsecond)
    return ", ".join(str(component) for component in components)


def _replace_datetime_call(match: re.Match[str], logical_time: datetime) -> str:
    prefix = match.group("prefix")
    method = match.group("method")
    argument = match.group("argument").strip()
    suffix = ""
    if method == "now" and argument:
        if argument.startswith("tz="):
            argument = argument[3:].strip()
        suffix = f", tzinfo={argument}"
    return f"{prefix}({_datetime_arguments(logical_time)}{suffix})"


def _freeze_sql_clock(value: str, logical_time: datetime) -> str:
    timestamp = logical_time.strftime("%Y-%m-%d %H:%M:%S")
    result = _SQLITE_NOW_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{timestamp}{match.group('quote')}",
        str(value),
    )
    result = _SQLITE_STRFTIME_NOW_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{timestamp}{match.group('quote')}",
        result,
    )
    result = re.sub(r"\bCURRENT_TIMESTAMP\b", f"'{timestamp}'", result)
    result = re.sub(
        r"\bCURRENT_DATE\b",
        f"'{logical_time.date().isoformat()}'",
        result,
    )
    result = re.sub(
        r"\bCURRENT_TIME\b",
        f"'{logical_time.time().replace(microsecond=0).isoformat()}'",
        result,
    )
    return result


def _freeze_python_clock(value: str, logical_time: datetime) -> str:
    result = str(value)
    result = _DATETIME_CALL_RE.sub(
        lambda match: _replace_datetime_call(match, logical_time),
        result,
    )
    result = _DATE_CALL_RE.sub(
        lambda match: (f"{match.group('prefix')}({logical_time.year}, {logical_time.month}, {logical_time.day})"),
        result,
    )
    return result


def freeze_temporal_text(value: str, logical_time: datetime) -> str:
    """Replace wall-clock expressions in a raw SQL/DDL text value."""
    frozen = _freeze_sql_clock(value, logical_time)
    return _freeze_python_clock(frozen, logical_time)


def freeze_temporal_source(value: str, logical_time: datetime) -> str:
    """Freeze source code while safely rewriting SQL inside string tokens."""
    source = _freeze_python_clock(str(value), logical_time)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        return source
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    replacements: list[tuple[int, int, str]] = []

    def absolute(position: tuple[int, int]) -> int:
        return line_offsets[position[0] - 1] + position[1]

    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        try:
            literal = ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(literal, str):
            continue
        frozen = _freeze_sql_clock(literal, logical_time)
        if frozen == literal:
            continue
        start = absolute(token.start)
        end = absolute(token.end)
        replacements.append((start, end, repr(frozen)))
    timestamp = logical_time.strftime("%Y-%m-%d %H:%M:%S")
    for index in range(len(tokens) - 4):
        window = tokens[index : index + 5]
        if [token.string for token in window[:2] + window[3:]] != [
            "func",
            ".",
            "(",
            ")",
        ]:
            continue
        if window[2].string not in {"current_timestamp", "now"}:
            continue
        start = absolute(window[0].start)
        end = absolute(window[4].end)
        replacements.append(
            (
                start,
                end,
                f"func.datetime({timestamp!r})",
            )
        )
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def freeze_temporal_structure(value: Any, logical_time: datetime) -> Any:
    if isinstance(value, str):
        return freeze_temporal_text(value, logical_time)
    if isinstance(value, list):
        return [freeze_temporal_structure(item, logical_time) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_temporal_structure(item, logical_time) for item in value)
    if isinstance(value, dict):
        return {key: freeze_temporal_structure(item, logical_time) for key, item in value.items()}
    return value


def _jsonl_by_scenario(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            scenario = normalize_scenario_name(record["scenario"])
            if scenario in output:
                raise RuntimeError(f"duplicate AWM scenario {scenario!r} in {path}")
            output[scenario] = record
    return output


@dataclass(frozen=True)
class LogicalTimePolicy:
    times: dict[str, datetime]
    sources: dict[str, str]

    def for_scenario(self, scenario: str) -> datetime:
        key = normalize_scenario_name(scenario)
        try:
            return self.times[key]
        except KeyError as exc:
            raise ValueError(f"AWM logical time is missing scenario {key!r}") from exc

    def scenario_record(self, scenario: str) -> dict[str, str]:
        key = normalize_scenario_name(scenario)
        when = self.for_scenario(key)
        return {
            "scenario": key,
            "logical_time_utc": when.isoformat(timespec="seconds") + "Z",
            "source": self.sources[key],
        }

    def protocol(self) -> dict[str, Any]:
        encoded = json.dumps(
            {scenario: when.isoformat(timespec="microseconds") for scenario, when in sorted(self.times.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
        counts: dict[str, int] = {}
        for source in self.sources.values():
            counts[source] = counts.get(source, 0) + 1
        return {
            "protocol_version": LOGICAL_TIME_PROTOCOL_VERSION,
            "dataset_revision": DATASET_REVISION,
            "fallback_logical_time_utc": (FALLBACK_LOGICAL_TIME.isoformat(timespec="seconds") + "Z"),
            "inference_rule": ("max literal created_at/updated_at timestamp plus one second; otherwise the pinned dataset fallback"),
            "scenario_count": len(self.times),
            "source_counts": dict(sorted(counts.items())),
            "scenario_times_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }


def load_logical_time_policy(data_dir: Path) -> LogicalTimePolicy:
    samples = _jsonl_by_scenario(Path(data_dir) / "gen_sample.jsonl")
    if len(samples) != 1000:
        raise RuntimeError(f"AWM logical-time protocol expected 1000 scenarios, got {len(samples)}")
    times: dict[str, datetime] = {}
    sources: dict[str, str] = {}
    for scenario, record in samples.items():
        times[scenario], sources[scenario] = infer_logical_time(record.get("sample_data"))
    return LogicalTimePolicy(times=times, sources=sources)


_INSTALLED_POLICY: LogicalTimePolicy | None = None
_INSTALLED_DATA_DIR: Path | None = None


def install_logical_time(data_dir: Path) -> LogicalTimePolicy:
    """Patch the pinned AWM loader without modifying its clean checkout."""
    global _INSTALLED_DATA_DIR, _INSTALLED_POLICY
    resolved = Path(data_dir).expanduser().resolve()
    if _INSTALLED_POLICY is not None:
        if _INSTALLED_DATA_DIR != resolved:
            raise RuntimeError("AWM logical time was already installed for another data directory")
        return _INSTALLED_POLICY

    from agent_world_model_env.server.data_loader import AWMDataLoader

    policy = load_logical_time_policy(resolved)
    original_get_db_schema = AWMDataLoader.get_db_schema
    original_get_env_code = AWMDataLoader.get_env_code
    original_get_sample_data = AWMDataLoader.get_sample_data
    original_get_verifier = AWMDataLoader.get_verifier

    def transform_structure(method: Callable, instance, scenario: str, *args):
        value = copy.deepcopy(method(instance, scenario, *args))
        return freeze_temporal_structure(value, policy.for_scenario(scenario))

    def get_db_schema(instance, scenario: str):
        return transform_structure(original_get_db_schema, instance, scenario)

    def get_env_code(instance, scenario: str):
        value = original_get_env_code(instance, scenario)
        return freeze_temporal_source(value, policy.for_scenario(scenario))

    def get_sample_data(instance, scenario: str):
        return transform_structure(original_get_sample_data, instance, scenario)

    def get_verifier(instance, scenario: str, task_idx: int, verifier_mode="sql"):
        entry = copy.deepcopy(original_get_verifier(instance, scenario, task_idx, verifier_mode))
        if entry is not None:
            verification = entry.get("verification") or {}
            code = verification.get("code")
            if isinstance(code, str):
                verification["code"] = freeze_temporal_source(code, policy.for_scenario(scenario))
                entry["verification"] = verification
        return entry

    AWMDataLoader.get_db_schema = get_db_schema
    AWMDataLoader.get_env_code = get_env_code
    AWMDataLoader.get_sample_data = get_sample_data
    AWMDataLoader.get_verifier = get_verifier
    _INSTALLED_DATA_DIR = resolved
    _INSTALLED_POLICY = policy
    return policy


def fetch_server_protocol(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    url = str(base_url).rstrip("/") + LOGICAL_TIME_ENDPOINT
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("protocol_version") != LOGICAL_TIME_PROTOCOL_VERSION:
        raise RuntimeError("AWM server logical-time protocol version mismatch")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError("AWM server logical-time dataset revision mismatch")
    if int(payload.get("scenario_count", -1)) != 1000:
        raise RuntimeError("AWM server logical-time scenario count mismatch")
    return payload


def require_server_protocol(base_url: str, data_dir: Path, timeout: float = 5.0) -> dict[str, Any]:
    actual = fetch_server_protocol(base_url, timeout=timeout)
    expected = load_logical_time_policy(data_dir).protocol()
    if actual != expected:
        raise RuntimeError("AWM server logical-time identity differs from the pinned local dataset")
    return actual
