"""Fresh-reset, executable evidence packets for AWM semantic review."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...runtime.actions import AWMAction, tool_schema_audit, tool_schema_hash
from ...runtime.failures import replay_observation_signature
from ...runtime.rollout import observation_dict, sha256_file

EVIDENCE_PROTOCOL_VERSION = 2
EVIDENCE_SEED = 300
SOURCE_ARTIFACT_NAMES = frozenset(
    {
        "environment",
        "sample",
        "schema",
        "tasks",
        "code_verifier",
        "sql_verifier",
    }
)


class ReplayDriftError(RuntimeError):
    """The fresh executable replay no longer matches the screened trajectory."""

    def __init__(self, component: str, *, expected: str, actual: str):
        super().__init__(f"fresh replay {component} drift: expected={expected!r}, actual={actual!r}")
        self.component = component
        self.expected = expected
        self.actual = actual


def require_replay_match(component: str, *, expected: str, actual: str) -> None:
    if actual != expected:
        raise ReplayDriftError(component, expected=expected, actual=actual)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_scenario(value: str) -> str:
    return "_".join(part for part in str(value).lower().replace("-", "_").split("_") if part)


class SourceCatalog:
    """Lazy byte-offset indexes over immutable, pinned AWM source JSONL files."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self._indexes: dict[
            tuple[str, str],
            dict[str | tuple[str, int], list[int]],
        ] = {}
        self._cohort_hash_cache: dict[tuple[str, int], dict[str, str]] = {}

    def _index(
        self,
        filename: str,
        key_kind: str,
    ) -> dict[str | tuple[str, int], list[int]]:
        cache_key = (filename, key_kind)
        if cache_key in self._indexes:
            return self._indexes[cache_key]
        index: dict[str | tuple[str, int], list[int]] = defaultdict(list)
        path = self.data_dir / filename
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                scenario = _normalize_scenario(record.get("scenario") or "")
                key: str | tuple[str, int]
                if key_kind == "scenario":
                    key = scenario
                elif key_kind == "task":
                    key = (scenario, int(record.get("task_idx", -1)))
                else:
                    raise ValueError(f"unknown source index kind: {key_kind!r}")
                index[key].append(offset)
        output = dict(index)
        self._indexes[cache_key] = output
        return output

    def _records(
        self,
        filename: str,
        key_kind: str,
        key: str | tuple[str, int],
    ) -> list[dict[str, Any]]:
        offsets = self._index(filename, key_kind).get(key) or []
        records = []
        with (self.data_dir / filename).open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                records.append(json.loads(handle.readline()))
        return records

    @staticmethod
    def _unique(records: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
        unique = {canonical_json(record): dict(record) for record in records}
        if len(unique) != 1:
            raise RuntimeError(f"semantic evidence requires one canonical {name}; found {len(records)} records with {len(unique)} distinct values")
        return next(iter(unique.values()))

    def task_sources(self, scenario: str, task_idx: int) -> dict[str, Any]:
        normalized = _normalize_scenario(scenario)
        key = (normalized, int(task_idx))
        sql_verifiers = self._records("gen_verifier.jsonl", "task", key)
        return {
            "environment": self._unique(
                self._records("gen_envs.jsonl", "scenario", normalized),
                "environment source",
            ),
            "sample": self._unique(
                self._records("gen_sample.jsonl", "scenario", normalized),
                "sample source",
            ),
            "schema": self._unique(
                self._records("gen_db.jsonl", "scenario", normalized),
                "database schema source",
            ),
            "tasks": self._unique(
                self._records("gen_tasks.jsonl", "scenario", normalized),
                "task source",
            ),
            "code_verifier": self._unique(
                self._records("gen_verifier.pure_code.jsonl", "task", key),
                "code verifier",
            ),
            "sql_verifier": (self._unique(sql_verifiers, "SQL verifier") if len(sql_verifiers) == 1 else {"records": sql_verifiers}),
        }

    def task_cohort_hashes(self, scenario: str, task_idx: int) -> dict[str, str]:
        normalized = _normalize_scenario(scenario)
        key = (normalized, int(task_idx))
        if key not in self._cohort_hash_cache:
            environment = self._unique(
                self._records("gen_envs.jsonl", "scenario", normalized),
                "environment source",
            )
            code_verifier = self._unique(
                self._records("gen_verifier.pure_code.jsonl", "task", key),
                "code verifier",
            )
            self._cohort_hash_cache[key] = {
                "environment": sha256_json(environment),
                "code_verifier": sha256_json(code_verifier),
            }
        return dict(self._cohort_hash_cache[key])


def write_source_bundle(
    *,
    sources: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for name, value in sources.items():
        content_sha256 = sha256_json(value)
        path = output_dir / "source_bundles" / name / f"{content_sha256}.json"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts[name] = {
            "path": str(path.relative_to(output_dir)),
            "sha256": sha256_file(path),
        }
    environment_hash = sha256_json(sources["environment"])
    verifier_hash = sha256_json(sources["code_verifier"])
    cohort_keys = {
        f"environment_source:{environment_hash}": "all tasks backed by the same environment implementation",
        f"code_verifier:{verifier_hash}": "all tasks backed by the same code-verifier record",
    }
    return artifacts, cohort_keys


def validate_evidence_packet(
    packet: Mapping[str, Any],
    *,
    output_dir: Path,
    task_id: str | None = None,
    screening_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate packet identity and every content-addressed source artifact."""
    required = {
        "protocol_version",
        "task_id",
        "scenario",
        "task_idx",
        "task",
        "screening_manifest_sha256",
        "screening_trial",
        "fresh_replay",
        "source_artifacts",
        "cohort_keys",
    }
    missing = sorted(required - set(packet))
    if missing:
        raise ValueError(f"semantic evidence packet missing fields: {missing!r}")
    if packet["protocol_version"] != EVIDENCE_PROTOCOL_VERSION:
        raise ValueError("semantic evidence protocol mismatch")
    actual_task_id = str(packet["task_id"])
    if task_id is not None and actual_task_id != task_id:
        raise ValueError("semantic evidence task ID mismatch")
    if screening_manifest_sha256 is not None and packet["screening_manifest_sha256"] != screening_manifest_sha256:
        raise ValueError("semantic evidence screening-manifest hash mismatch")
    replay = packet["fresh_replay"]
    if not isinstance(replay, Mapping) or replay.get("validation") != "matched":
        raise ValueError("semantic evidence fresh replay is not drift-validated")
    artifacts = packet["source_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != SOURCE_ARTIFACT_NAMES:
        raise ValueError("semantic evidence source-artifact set mismatch")
    source_root = (output_dir / "source_bundles").resolve()
    source_values = {}
    for name, metadata in artifacts.items():
        if not isinstance(metadata, Mapping) or set(metadata) != {"path", "sha256"}:
            raise ValueError(f"semantic evidence source artifact metadata mismatch: {name}")
        relative = Path(str(metadata["path"]))
        if relative.is_absolute():
            raise ValueError(f"semantic evidence source artifact path must be relative: {name}")
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"semantic evidence source artifact escapes bundle root: {name}")
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"semantic evidence source artifact hash mismatch: {name}")
        source_values[name] = json.loads(path.read_text(encoding="utf-8"))
    task_source = source_values["tasks"]
    task_idx = int(packet["task_idx"])
    source_tasks = task_source.get("tasks") or []
    if str(task_source.get("scenario") or "") != str(packet["scenario"]) or task_idx < 0 or task_idx >= len(source_tasks) or str(source_tasks[task_idx]) != str(packet["task"]):
        raise ValueError("semantic evidence task identity differs from source bundle")
    code_verifier = source_values["code_verifier"]
    if str(code_verifier.get("scenario") or "") != str(packet["scenario"]) or int(code_verifier.get("task_idx", -1)) != task_idx:
        raise ValueError("semantic evidence verifier identity differs from packet")
    cohort_keys = packet["cohort_keys"]
    if not isinstance(cohort_keys, Mapping) or not cohort_keys:
        raise ValueError("semantic evidence requires cohort keys")
    return dict(packet)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _database_snapshot(path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        tables = [str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})")]
            rows = [[_json_value(item) for item in row] for row in connection.execute(f"SELECT * FROM {quoted}")]
            rows.sort(key=canonical_json)
            output[table] = {"columns": columns, "rows": rows}
    return output


def sqlite_diff(initial_path: Path, final_path: Path) -> dict[str, Any]:
    initial = _database_snapshot(initial_path)
    final = _database_snapshot(final_path)
    tables: dict[str, Any] = {}
    for table in sorted(set(initial) | set(final)):
        before = initial.get(table, {"columns": [], "rows": []})
        after = final.get(table, {"columns": [], "rows": []})
        before_rows = {canonical_json(row): row for row in before["rows"]}
        after_rows = {canonical_json(row): row for row in after["rows"]}
        before_counts = Counter(canonical_json(row) for row in before["rows"])
        after_counts = Counter(canonical_json(row) for row in after["rows"])
        added = [after_rows[key] for key in sorted(after_counts) for _ in range(max(0, after_counts[key] - before_counts[key]))]
        removed = [before_rows[key] for key in sorted(before_counts) for _ in range(max(0, before_counts[key] - after_counts[key]))]
        if added or removed or before["columns"] != after["columns"]:
            tables[table] = {
                "columns_before": before["columns"],
                "columns_after": after["columns"],
                "rows_before": len(before["rows"]),
                "rows_after": len(after["rows"]),
                "added": added,
                "removed": removed,
            }
    return {
        "initial_db_sha256": sha256_file(initial_path),
        "final_db_sha256": sha256_file(final_path),
        "changed_tables": tables,
    }


def _parsed_action(entry: Mapping[str, Any]) -> AWMAction:
    payload = json.loads(str(entry.get("parsed_action") or "{}"))
    return AWMAction(
        kind=str(payload.get("kind") or "invalid"),
        name=payload.get("name"),
        arguments=dict(payload.get("arguments") or {}),
        content=payload.get("content"),
        error=payload.get("error"),
    )


def compact_screening_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    result = trial.get("result") or trial.get("last_result") or {}
    trajectory = []
    for entry in result.get("trajectory") or []:
        trajectory.append(
            {
                key: entry.get(key)
                for key in (
                    "decision",
                    "action_kind",
                    "parsed_action",
                    "raw_action",
                    "parse_error",
                    "native_tool_calls",
                    "skipped_tool_calls",
                    "tool_response",
                    "tool_response_is_error",
                    "tool_reward_type",
                    "runtime_infrastructure_error",
                    "runtime_error_signature",
                    "tool_observation_signature",
                )
            }
        )
    return {
        "status": trial.get("status"),
        "seed": trial.get("seed"),
        "infrastructure_attempts": trial.get("infrastructure_attempts"),
        "infrastructure_errors": trial.get("infrastructure_errors") or trial.get("errors") or [],
        "runtime_replay": trial.get("runtime_replay") or trial.get("last_runtime_replay"),
        "result": {
            key: result.get(key)
            for key in (
                "success",
                "reward",
                "reward_type",
                "decisions",
                "final_answer",
                "verify_result",
                "verify_infrastructure_error",
                "verify_error_signature",
                "verify_observation_signature",
                "tool_schema_hash",
                "raw_tool_schema_hash",
            )
        },
        "trajectory": trajectory,
    }


def _validated_session_dir(value: str, scenario: str) -> Path:
    path = Path(value).resolve()
    tmp_root = Path("/tmp").resolve()
    expected_prefix = f"openenv_awm_{_normalize_scenario(scenario)}_"
    if not path.is_relative_to(tmp_root) or not path.name.startswith(expected_prefix):
        raise RuntimeError(f"refusing unsafe AWM session path: {path}")
    return path


@asynccontextmanager
async def _replay_session(env, call_tool_action):
    async with env as connected:
        try:
            yield connected
        except Exception:
            try:
                await connected.step(call_tool_action(tool_name="done", arguments={}))
            except Exception:
                pass
            raise


async def replay_trial(
    row: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    awm_base_url: str,
) -> dict[str, Any]:
    """Replay recorded structured actions serially on a fresh seed-300 reset."""
    from agent_world_model_env import AWMEnv
    from openenv.core.env_server.mcp_types import CallToolAction

    result = trial.get("result") or trial.get("last_result") or {}
    final_answer = result.get("final_answer")
    session_dir: Path | None = None
    replay_steps = []
    try:
        async with _replay_session(
            AWMEnv(base_url=awm_base_url),
            CallToolAction,
        ) as env:
            reset = observation_dict(
                await env.reset(
                    scenario=str(row["scenario"]),
                    task_idx=int(row["task_idx"]),
                    seed=EVIDENCE_SEED,
                )
            )
            if reset.get("reward_type") not in {"reset_ok", "reset_warning"}:
                raise RuntimeError(f"evidence reset failed: {reset}")
            if str(reset.get("task") or "") != str(row["task"]):
                raise RuntimeError("evidence reset task text drifted")
            schema = tool_schema_audit(await env.list_tools(use_cache=False))
            actual_schema_hash = tool_schema_hash(schema["canonical_tools"])
            expected_schema_hash = str(result.get("tool_schema_hash") or "")
            require_replay_match(
                "canonical tool schema",
                expected=expected_schema_hash,
                actual=actual_schema_hash,
            )
            actual_raw_schema_hash = str(schema["raw_tool_schema_hash"])
            expected_raw_schema_hash = str(result.get("raw_tool_schema_hash") or "")
            require_replay_match(
                "raw tool schema",
                expected=expected_raw_schema_hash,
                actual=actual_raw_schema_hash,
            )
            for entry in result.get("trajectory") or []:
                action = _parsed_action(entry)
                if action.kind == "message":
                    final_answer = action.content or ""
                    replay_steps.append({"action": json.loads(str(entry["parsed_action"])), "observation": None})
                    break
                if action.kind != "tool":
                    replay_steps.append({"action": json.loads(str(entry.get("parsed_action") or "{}")), "observation": None})
                    continue
                observation = observation_dict(
                    await env.step(
                        CallToolAction(
                            tool_name=action.name or "",
                            arguments=action.arguments or {},
                        )
                    )
                )
                actual_signature = replay_observation_signature(observation)
                expected_signature = str(entry.get("tool_observation_signature") or "")
                require_replay_match(
                    "tool observation",
                    expected=expected_signature,
                    actual=actual_signature,
                )
                replay_steps.append(
                    {
                        "action": json.loads(str(entry["parsed_action"])),
                        "observation": observation,
                        "observation_signature": actual_signature,
                    }
                )
            verify = observation_dict(
                await env.step(
                    CallToolAction(
                        tool_name="verify",
                        arguments={"verifier_mode": "code", "final_answer": final_answer},
                    )
                )
            )
            actual_verify_signature = replay_observation_signature(verify)
            expected_verify_signature = str(result.get("verify_observation_signature") or "")
            require_replay_match(
                "verifier observation",
                expected=expected_verify_signature,
                actual=actual_verify_signature,
            )
            done = observation_dict(await env.step(CallToolAction(tool_name="done", arguments={"keep_session": True})))
            session_dir = _validated_session_dir(str(done.get("session_dir") or ""), str(row["scenario"]))
            initial_path = session_dir / f"{_normalize_scenario(str(row['scenario']))}_initial.db"
            final_path = session_dir / f"{_normalize_scenario(str(row['scenario']))}.db"
            if not initial_path.is_file() or not final_path.is_file():
                raise RuntimeError("AWM retained session is missing initial/final SQLite databases")
            database_diff = sqlite_diff(initial_path, final_path)
            server_log = session_dir / "server.log"
            log_tail = server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:] if server_log.is_file() else []
            return {
                "validation": "matched",
                "seed": EVIDENCE_SEED,
                "reset": reset,
                "tool_schema_hash": actual_schema_hash,
                "raw_tool_schema_hash": actual_raw_schema_hash,
                "steps": replay_steps,
                "final_answer": final_answer,
                "verify": verify,
                "verify_observation_signature": actual_verify_signature,
                "database_diff": database_diff,
                "server_log_tail": log_tail,
            }
    finally:
        if session_dir is not None and session_dir.is_dir():
            shutil.rmtree(session_dir)


async def build_evidence_packet(
    *,
    row: Mapping[str, Any],
    trial: Mapping[str, Any],
    catalog: SourceCatalog,
    output_dir: Path,
    awm_base_url: str,
    screening_manifest_sha256: str,
) -> dict[str, Any]:
    sources = catalog.task_sources(str(row["scenario"]), int(row["task_idx"]))
    artifacts, cohort_keys = write_source_bundle(
        sources=sources,
        output_dir=output_dir,
    )
    compact = compact_screening_trial(trial)
    for entry in compact["trajectory"]:
        signature = entry.get("runtime_error_signature")
        if signature:
            cohort_keys[f"runtime_error:{signature}"] = "all packets with the same deterministic runtime-error signature"
    return {
        "protocol_version": EVIDENCE_PROTOCOL_VERSION,
        "task_id": str(row["task_id"]),
        "scenario": str(row["scenario"]),
        "task_idx": int(row["task_idx"]),
        "task": str(row["task"]),
        "screening_manifest_sha256": screening_manifest_sha256,
        "screening_trial": compact,
        "fresh_replay": await replay_trial(row, trial, awm_base_url=awm_base_url),
        "source_artifacts": artifacts,
        "cohort_keys": cohort_keys,
    }
