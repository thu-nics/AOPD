"""Pinned EnvScaler source loading and deterministic task execution primitives."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import types
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from agent_system.environments.env_package.awm.runtime.actions import (
    normalize_tools,
    openai_tools,
)

ENVSCALER_COMMIT = "87e667397abacf274858c0964796beb8f984aafe"
ENV_METADATA_SHA256 = "d2c0010f16ff77d6d55868ee386353b1d0aadace58beed1eed678e8f7c84c33d"
RL_METADATA_SHA256 = "5977bda0b941a9111b290cbf5ffd6d70678a36ddc499b8f153826fd22999337e"
SFT_METADATA_SHA256 = "4389861728c68de9c6b37c08dba923ccff4c939165c2538456892a1316f435e8"
EXPECTED_ENV_COUNT = 191
EXPECTED_RL_ENV_COUNT = 51
EXPECTED_RL_TASK_COUNT = 2550
EXPECTED_TASKS_PER_RL_ENV = 50
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[4].parent / "EnvScaler"

_DATA_RELATIVE = Path("interact_with_env/envscaler_env/data")
_SOURCE_FILES = {
    "191_env_metadata.json": ENV_METADATA_SHA256,
    "rl_scenario_metadata.json": RL_METADATA_SHA256,
    "sft_scenario_metadata.json": SFT_METADATA_SHA256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_envscaler_source(root: str | Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EnvScaler source root does not exist: {root}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        tracked_changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve EnvScaler commit under {root}") from exc
    if commit != ENVSCALER_COMMIT:
        raise RuntimeError(f"EnvScaler source must be pinned to {ENVSCALER_COMMIT}, got {commit}")
    if tracked_changes:
        raise RuntimeError(f"EnvScaler source has tracked modifications under {root}:\n{tracked_changes}")
    data_root = root / _DATA_RELATIVE
    hashes = {}
    for filename, expected in _SOURCE_FILES.items():
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned EnvScaler data: {path}")
        actual = sha256_file(path)
        hashes[filename] = actual
        if actual != expected:
            raise RuntimeError(f"EnvScaler data hash mismatch for {filename}: expected {expected}, got {actual}")
    return {
        "source_root": str(root),
        "commit": commit,
        "source_sha256": hashes,
    }


@dataclass(frozen=True)
class EnvScalerSource:
    root: Path
    environments: dict[str, dict[str, Any]]
    tasks: tuple[dict[str, Any], ...]
    identity: dict[str, Any]
    task_index_by_id: dict[str, int]


@lru_cache(maxsize=4)
def load_envscaler_source(
    root: str | Path = DEFAULT_SOURCE_ROOT,
) -> EnvScalerSource:
    resolved = Path(root).expanduser().resolve()
    identity = validate_envscaler_source(resolved)
    data_root = resolved / _DATA_RELATIVE
    environments = json.loads((data_root / "191_env_metadata.json").read_text(encoding="utf-8"))
    tasks = json.loads((data_root / "rl_scenario_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(environments, dict) or len(environments) != EXPECTED_ENV_COUNT:
        raise RuntimeError(f"EnvScaler must contain {EXPECTED_ENV_COUNT} environments")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_RL_TASK_COUNT:
        raise RuntimeError(f"EnvScaler must contain {EXPECTED_RL_TASK_COUNT} RL tasks")
    task_ids = [str(item.get("task_id") or "") for item in tasks]
    if any(not value for value in task_ids) or len(set(task_ids)) != len(task_ids):
        raise RuntimeError("EnvScaler task IDs must be non-empty and unique")
    counts: dict[str, int] = {}
    for task in tasks:
        env_id = str(task.get("env_id") or "")
        if env_id not in environments:
            raise RuntimeError(f"EnvScaler task {task.get('task_id')} references missing {env_id}")
        counts[env_id] = counts.get(env_id, 0) + 1
    if len(counts) != EXPECTED_RL_ENV_COUNT or set(counts.values()) != {EXPECTED_TASKS_PER_RL_ENV}:
        raise RuntimeError("EnvScaler RL split must contain 51 environments with 50 tasks each")
    return EnvScalerSource(
        root=resolved,
        environments=environments,
        tasks=tuple(tasks),
        identity=identity,
        task_index_by_id={task_id: index for index, task_id in enumerate(task_ids)},
    )


def state_dict(instance: Any) -> dict[str, Any]:
    return deepcopy({key: value for key, value in vars(instance).items() if not (key.startswith("__") and key.endswith("__"))})


def restore_state(instance: Any, snapshot: Mapping[str, Any]) -> None:
    instance.__dict__.clear()
    instance.__dict__.update(deepcopy(dict(snapshot)))


def build_environment_instance(
    environment: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Any:
    module = types.ModuleType(f"envscaler_{str(task['env_id']).replace('-', '_')}")
    code = str(environment["env_class_code"])
    compiled = compile(code, f"<EnvScaler:{task['env_id']}>", "exec")
    exec(compiled, module.__dict__)
    class_name = str(task["env_class_name"])
    if not hasattr(module, class_name):
        raise ValueError(f"EnvScaler class {class_name!r} is missing")
    cls = getattr(module, class_name)
    init_config = deepcopy(task.get("init_config") or {})
    try:
        instance = cls(init_config) if init_config else cls({})
    except TypeError:
        instance = cls()
    for key, value in init_config.items():
        setattr(instance, key, deepcopy(value))
    return instance


def evaluate_checkers(
    task: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for index, checker in enumerate(task.get("checklist_with_func") or []):
        code = str(checker.get("check_func") or "")
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "initial_state": deepcopy(dict(initial_state)),
        }
        success = False
        result = None
        error = None
        try:
            exec(compile(code, f"<EnvScaler-check:{task['task_id']}:{index}>", "exec"), namespace)
            function = namespace.get("check_func")
            if not callable(function):
                raise ValueError("check_func entrypoint is missing")
            value = function(deepcopy(dict(final_state)))
            if not isinstance(value, bool):
                raise TypeError(f"check_func returned {type(value).__name__}, expected bool")
            success = True
            result = value
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        output.append(
            {
                "index": index,
                "check_item": str(checker.get("check_item") or ""),
                "check_func": code,
                "success": success,
                "result": result,
                "error": error,
            }
        )
    return output


def checker_summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [dict(item) for item in results if not item.get("success")]
    passed = sum(item.get("result") is True for item in results)
    total = len(results)
    return {
        "checker_count": total,
        "checker_passed": passed,
        "checker_fraction": (passed / total) if total else 0.0,
        "checker_errors": errors,
        "state_complete": bool(total and not errors and passed == total),
    }


def validate_tool_contract(
    environment: Mapping[str, Any],
    instance: Any,
) -> list[dict[str, Any]]:
    tools = normalize_tools(environment.get("tools") or [])
    for tool in tools:
        name = tool["name"]
        if not hasattr(instance, name) or not callable(getattr(instance, name)):
            raise ValueError(f"tool {name!r} has no callable environment method")
        Draft202012Validator.check_schema(tool["inputSchema"])
        signature = inspect.signature(getattr(instance, name))
        parameters = signature.parameters
        schema_properties = tool["inputSchema"].get("properties") or {}
        required = set(tool["inputSchema"].get("required") or [])
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        missing_parameters = [field for field in schema_properties if field not in parameters and not accepts_kwargs]
        if missing_parameters:
            raise ValueError(f"tool {name!r} schema fields absent from method signature: {missing_parameters}")
        missing_required_schema = [
            field
            for field, parameter in parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and parameter.default is inspect.Parameter.empty
            and field not in required
        ]
        if missing_required_schema:
            raise ValueError(f"tool {name!r} required method fields absent from schema.required: {missing_required_schema}")
    # Conversion validates provider-compatible names and duplicate names.
    openai_tools(tools)
    return tools
