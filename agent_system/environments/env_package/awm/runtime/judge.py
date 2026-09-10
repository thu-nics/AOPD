"""Code-augmented DeepSeek adjudication for AWM tool execution failures."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

RUNTIME_JUDGE_PROTOCOL_VERSION = 2
RUNTIME_JUDGE_INSTRUCTION = """You classify one failed AWM agent tool execution using the supplied task, action, successful fresh-reset reference, endpoint source, route registry, and database DDL.

The failed action has already passed the public tool JSON schema.

error_class:
- policy_execution_error: the chosen action violates the current tool/state contract even if its JSON matches the public schema.
  Examples: guessed/wrong IDs, duplicate creation prohibited by state constraints, invalid enum/date/value, malformed content,
  or missing prerequisites. Generated APIs may expose these as HTTP 500.
- infrastructure_error: the action is a reasonable state-valid use of a public tool, but the generated API/environment is defective or routes it incorrectly.
- uncertain: evidence is insufficient.

Critical rules:
- Do not classify from HTTP status alone.
- A different successful expert path does NOT prove that the failed action is wrong. Public tools with equivalent task semantics are valid alternatives unless source/schema/state evidence proves otherwise.
- An irrelevant or unnecessary but schema-valid tool call is not automatically a policy_execution_error. Task irrelevance is already handled by semantic reward 0. Classify it as policy_execution_error only when the action violates the tool/state contract.
- Compare argument values carefully. Truncated placeholders such as literal "..." inside encoded payloads are malformed data, not equivalent to a complete reference payload.

post_error_state:
- unchanged: strong source-level evidence that the handler was never entered, the operation is read-only, the exception happened before any write/commit, or an atomic commit failed and no prior commit occurred.
- possibly_mutated: a write/commit may have succeeded before the observed failure, including response construction/serialization failures after commit.
- unknown: source/evidence cannot establish either condition.

Return exactly this JSON shape and no other keys:
{"error_class":"policy_execution_error|infrastructure_error|uncertain","classification_confidence":95,"post_error_state":"unchanged|possibly_mutated|unknown","rationale":"short explanation"}

Evidence calibration: the supplied evidence is incomplete, not a complete execution trace.
An empty successful_fresh_reset_reference_actions list means NO REFERENCE WAS PROVIDED;
it proves nothing about which actions happened or which rows exist. Numeric IDs and example
values in Path/Body metadata are not database observations. Do not infer an ID was guessed,
a UNIQUE/FK constraint actually failed, or a prior lookup was omitted without affirmative
evidence. A possible explanation is not an established cause. Return uncertain when policy
error and environment defect cannot be distinguished. For post_error_state, if commit is
followed by response construction, serialization, or other fallible code and the failure
location is unknown, unchanged is not established; use possibly_mutated. Read-only handlers
and demonstrated failures before writes can be unchanged. Give a short evidence-based rationale.
"""
RUNTIME_JUDGE_PROMPT_HASH = hashlib.sha256(RUNTIME_JUDGE_INSTRUCTION.encode()).hexdigest()

_ERROR_CLASSES = {
    "policy_execution_error",
    "infrastructure_error",
    "uncertain",
}
_POST_ERROR_STATES = {"unchanged", "possibly_mutated", "unknown"}


def runtime_judge_decoding_config(
    *,
    provider: str = "deepseek",
    reasoning_effort: str = "auto",
    max_tokens: int = 8192,
) -> dict[str, Any]:
    provider = str(provider).lower()
    if reasoning_effort == "auto":
        reasoning_effort = "low" if provider == "deepseek" else "max"
    if int(max_tokens) < 8192:
        raise ValueError("AWM runtime judge requires max_tokens >= 8192")
    common = {
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if provider == "deepseek":
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("DeepSeek runtime judge reasoning_effort must be low, high, or max")
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": reasoning_effort,
            **common,
        }
    if provider == "dashscope":
        return {
            "enable_thinking": True,
            "thinking_budget": 4096,
            "temperature": 0.6,
            "top_p": 0.95,
            **common,
        }
    if provider == "zai":
        if reasoning_effort != "max":
            raise ValueError("ZAI GLM-5.3-Flash runtime judge requires reasoning_effort='max'")
        return {
            "thinking": {"type": "enabled", "clear_thinking": False},
            "reasoning_effort": reasoning_effort,
            "temperature": 1.0,
            "top_p": 0.95,
            **common,
        }
    raise ValueError(f"unsupported runtime judge provider: {provider!r}")


def validate_runtime_judge_verdict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime judge response must be a JSON object")
    expected_keys = {
        "error_class",
        "classification_confidence",
        "post_error_state",
        "rationale",
    }
    if set(value) != expected_keys:
        raise ValueError("runtime judge response has unexpected fields")
    error_class = value.get("error_class")
    if error_class not in _ERROR_CLASSES:
        raise ValueError("runtime judge returned an invalid error_class")
    confidence = value.get("classification_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        raise ValueError("runtime judge confidence must be an integer in [0, 100]")
    post_error_state = value.get("post_error_state")
    if post_error_state not in _POST_ERROR_STATES:
        raise ValueError("runtime judge returned an invalid post_error_state")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("runtime judge rationale must be a non-empty string")
    return {
        "error_class": str(error_class),
        "classification_confidence": int(confidence),
        "post_error_state": str(post_error_state),
        "rationale": rationale.strip(),
    }


def runtime_judge_fingerprint(
    *,
    model: str,
    decoding_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    payload = {
        "protocol_version": RUNTIME_JUDGE_PROTOCOL_VERSION,
        "prompt_hash": RUNTIME_JUDGE_PROMPT_HASH,
        "model": str(model),
        "decoding_config": dict(decoding_config),
        "evidence": dict(evidence),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _decorator_info(decorator: ast.expr) -> dict[str, Any] | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute) or not isinstance(decorator.func.value, ast.Name) or decorator.func.value.id != "app":
        return None
    path = decorator.args[0].value if decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str) else None
    operation_id = None
    for keyword in decorator.keywords:
        if keyword.arg == "operation_id" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            operation_id = keyword.value.value
    return {
        "method": decorator.func.attr.upper(),
        "path": path,
        "operation_id": operation_id,
    }


def endpoint_code_evidence(
    full_code: str,
    tool_name: str,
    ddl_by_table: Mapping[str, str],
) -> dict[str, Any]:
    """Extract only the failed handler, neighboring routes, and referenced DDL."""
    try:
        tree = ast.parse(full_code)
    except (SyntaxError, ValueError):
        return {
            "failed_endpoint_source": "",
            "related_route_registry": [],
            "referenced_database_ddl": [],
        }
    class_tables: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__tablename__" for target in statement.targets):
                continue
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                class_tables[node.name] = statement.value.value

    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    target_route: dict[str, Any] | None = None
    routes: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        infos = [info for decorator in node.decorator_list if (info := _decorator_info(decorator)) is not None]
        routes.extend(infos)
        for info in infos:
            if info.get("operation_id") == tool_name:
                target = node
                target_route = info

    source = ""
    referenced_ddls: list[str] = []
    if target is not None:
        lines = full_code.splitlines()
        start = min([target.lineno] + [decorator.lineno for decorator in target.decorator_list]) - 1
        source = "\n".join(lines[start : target.end_lineno])
        names = {item.id for item in ast.walk(target) if isinstance(item, ast.Name)}
        referenced_ddls = [str(ddl_by_table[table_name]) for class_name, table_name in class_tables.items() if class_name in names and table_name in ddl_by_table]

    related_routes: list[dict[str, Any]] = []
    if target_route and target_route.get("path"):
        parts = str(target_route["path"]).strip("/").split("/")
        prefix = "/" + "/".join(parts[:2])
        related_routes = [route for route in routes if str(route.get("path") or "").startswith(prefix)]
    return {
        "failed_endpoint_source": source[:14000],
        "related_route_registry": related_routes[:80],
        "referenced_database_ddl": referenced_ddls[:12],
    }


class RuntimeJudgeEvidenceStore:
    """Lazy offset-backed index for public AWM source, DDL, and expert trials."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        reference_trials_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.envs_path = self.data_dir / "gen_envs.jsonl"
        self.db_path = self.data_dir / "gen_db.jsonl"
        for path in (self.envs_path, self.db_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing AWM runtime-judge evidence: {path}")
        self.reference_trials_path = Path(reference_trials_path).expanduser().resolve() if reference_trials_path else None
        if self.reference_trials_path is not None and not self.reference_trials_path.is_file():
            raise FileNotFoundError(f"missing AWM runtime-judge expert references: {self.reference_trials_path}")
        self._env_offsets = self._scenario_offsets(self.envs_path)
        self._db_offsets = self._scenario_offsets(self.db_path)
        self._trial_offsets = self._trial_task_offsets(self.reference_trials_path)
        self._code_cache: dict[str, str] = {}
        self._ddl_cache: dict[str, dict[str, str]] = {}
        self._reference_cache: dict[str, list[dict[str, Any]]] = {}
        self._endpoint_cache: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _scenario_offsets(path: Path) -> dict[str, int]:
        offsets: dict[str, int] = {}
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                item = json.loads(line)
                scenario = str(item.get("scenario") or "")
                if scenario:
                    offsets[scenario] = offset
        return offsets

    @staticmethod
    def _trial_task_offsets(path: Path | None) -> dict[str, int]:
        if path is None:
            return {}
        offsets: dict[str, int] = {}
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                item = json.loads(line)
                item_task_id = str(item.get("task_id") or "")
                if item_task_id:
                    offsets[item_task_id] = offset
        return offsets

    @staticmethod
    def _read_at(path: Path, offset: int) -> dict[str, Any]:
        with path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def _full_code(self, scenario: str) -> str:
        if scenario not in self._code_cache:
            offset = self._env_offsets.get(scenario)
            if offset is None:
                self._code_cache[scenario] = ""
            else:
                item = self._read_at(self.envs_path, offset)
                self._code_cache[scenario] = str(item.get("full_code") or "")
        return self._code_cache[scenario]

    def _ddls(self, scenario: str) -> dict[str, str]:
        if scenario not in self._ddl_cache:
            offset = self._db_offsets.get(scenario)
            if offset is None:
                self._ddl_cache[scenario] = {}
            else:
                item = self._read_at(self.db_path, offset)
                tables = (item.get("db_schema") or {}).get("tables") or []
                self._ddl_cache[scenario] = {str(table["name"]): str(table["ddl"]) for table in tables if isinstance(table, Mapping) and table.get("name") and table.get("ddl")}
        return self._ddl_cache[scenario]

    def _reference_actions(self, item_task_id: str) -> list[dict[str, Any]]:
        if item_task_id in self._reference_cache:
            return list(self._reference_cache[item_task_id])
        actions: list[dict[str, Any]] = []
        offset = self._trial_offsets.get(item_task_id)
        if offset is not None and self.reference_trials_path is not None:
            trial = self._read_at(self.reference_trials_path, offset)
            for step in (trial.get("result") or {}).get("trajectory") or []:
                if step.get("action_kind") != "tool":
                    continue
                try:
                    action = json.loads(step.get("parsed_action") or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if action.get("name"):
                    actions.append(
                        {
                            "name": action.get("name"),
                            "arguments": action.get("arguments") or {},
                        }
                    )
        self._reference_cache[item_task_id] = actions
        return list(actions)

    def build(
        self,
        *,
        scenario: str,
        task_idx: int,
        task: str,
        failed_action: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        tool_name = str(failed_action.get("name") or "")
        endpoint_key = (str(scenario), tool_name)
        if endpoint_key not in self._endpoint_cache:
            self._endpoint_cache[endpoint_key] = endpoint_code_evidence(
                self._full_code(str(scenario)),
                tool_name,
                self._ddls(str(scenario)),
            )
        return {
            "task_id": f"{scenario}:{int(task_idx)}",
            "task": str(task),
            "failed_action": dict(failed_action),
            "http_error": payload.get("error"),
            "successful_fresh_reset_reference_actions": self._reference_actions(f"{scenario}:{int(task_idx)}"),
            **self._endpoint_cache[endpoint_key],
        }
