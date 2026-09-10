"""Source-only evidence for reward comparison, never for action execution.

No task, verifier, initial database or live state is accepted by this interface.
Deterministic exceptions are scoped to reviewed native function hashes. Unknown
differences go to the matcher; missing evidence raises there, not during reset.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import textwrap
from functools import lru_cache
from pathlib import Path

RULES_VERSION = 1
MAX_SOURCE_CHARS = 48000


def function_hash(node):
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()


@lru_cache(maxsize=1)
def _rules():
    return json.loads(Path(__file__).with_name("tool_matching_rules.json").read_text())


def _packet(tree, root, owner=None):
    """Reachable local helpers/types, not the environment initializer/checker.

    Module imports are declarations only. Never include executable module-level
    assignments (which can contain fixtures/credentials or seeded databases).
    Referenced literal constants are included only when scalar.
    """
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    methods = {n.name: n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} if owner else {}
    selected, pending = {}, [root]
    while pending:
        node = pending.pop()
        if id(node) in selected:
            continue
        if node.name == "__init__" or node.name.startswith(("check_task", "verify_task", "evaluate_task")):
            raise ValueError("tool depends on initializer or task-checking code")
        selected[id(node)] = node
        receivers = {node.args.args[0].arg} if node in methods.values() and node.args.args and node.args.args[0].arg in {"self", "cls"} else set()
        if receivers:
            self_calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in receivers}
            missing = self_calls - methods.keys()
            if missing:
                raise ValueError(f"unresolved native method dependencies: {', '.join(sorted(missing))}")
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        names.update(n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in receivers)
        for name in names:
            dep = methods.get(name) or definitions.get(name)
            if dep is not None and dep is not owner and id(dep) not in selected:
                pending.append(dep)
    names = {n.id for node in selected.values() for n in ast.walk(node) if isinstance(n, ast.Name)}
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets) and isinstance(node.value, ast.Constant):
            constants.append(ast.unparse(node))
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return "\n\n".join([*imports, *constants, *(ast.unparse(n) for n in sorted(selected.values(), key=lambda n: n.lineno))])


def source_tool_matching_metadata(source, tools, *, family, environment, class_name=None, defaults=None):
    """Build a serializable per-tool contract from native source and schema."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        return {t.get("function", t)["name"]: {"error": f"native source unavailable: {exc}"} for t in tools}
    owner = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None) if class_name else None
    functions, operation_ids = {}, {}
    for node in owner.body if owner else tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        functions.setdefault(node.name, []).append(node)
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "operation_id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        operation_ids.setdefault(kw.value.value, []).append(node)
    output = {}
    for raw in tools:
        tool = raw.get("function", raw)
        name = tool["name"]
        entry = {"family": family, "environment": environment, "tool_name": name, "rules_version": RULES_VERSION, "defaults": (defaults or {}).get(name, {}), "rules": []}
        output[name] = entry
        # FastAPI's explicit public operation ID wins over a same-named private
        # helper. Ambiguous public IDs themselves still fail closed.
        candidates = operation_ids.get(name, functions.get(name, []))
        if len(candidates) != 1 or (class_name and owner is None):
            entry["error"] = "native tool function cannot be uniquely resolved"
            continue
        node = candidates[0]
        entry["function_hash"] = function_hash(node)
        try:
            packet = _packet(tree, node, owner)
            if len(packet) > MAX_SOURCE_CHARS:
                raise ValueError("source evidence exceeds budget; refusing to truncate dependencies")
            entry["source"] = packet
            entry["source_hash"] = hashlib.sha256(packet.encode()).hexdigest()
            # Full module identity also invalidates verdicts if an external
            # local helper/constant was changed outside the extracted packet.
            entry["module_hash"] = hashlib.sha256(source.encode()).hexdigest()
            entry["rules"] = [r for r in _rules() if (r["family"], r["environment"], r["tool"], r["function_hash"], r["module_hash"]) == (family, environment, name, entry["function_hash"], entry["module_hash"])]
        except ValueError as exc:
            entry["error"] = str(exc)
    return output


def callable_tool_matching_metadata(tools, *, family, environment):
    """Read native bound functions, not objects' state/initialization values."""
    from agent_system.environments.action_matching import callable_defaults

    output = {}
    for tool in tools:
        function = getattr(tool, "_func", None)
        try:
            module = inspect.getmodule(function)
            source = inspect.getsource(module)
            owner = function.__qualname__.split(".")[-2] if "." in function.__qualname__ else None
            entry = source_tool_matching_metadata(source, [tool.openai_schema], family=family, environment=environment, class_name=owner, defaults={tool.name: callable_defaults(function)})[tool.name]
            # Imported native data types contain schema/validation rules, not DB
            # instances. Include their definitions without reading any values.
            packet = entry.get("source", "")
            types, seen = [], set()
            pending = [(packet, module)]
            while pending:
                code, namespace = pending.pop()
                names = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
                for name in sorted(names):
                    obj = vars(namespace).get(name)
                    if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, "__module__", "").startswith("tau2.domains.") and obj.__module__ != module.__name__ and obj not in seen:
                        seen.add(obj)
                        definition = textwrap.dedent(inspect.getsource(obj))
                        types.append(definition)
                        pending.append((definition, inspect.getmodule(obj)))
            if types:
                entry["source"] = packet + "\n\n" + "\n\n".join(types)
                entry["source_hash"] = hashlib.sha256(entry["source"].encode()).hexdigest()
            if len(entry.get("source", "")) > MAX_SOURCE_CHARS:
                entry["error"] = "native source evidence exceeds budget"
            output[tool.name] = entry
        except (OSError, TypeError, ValueError, AttributeError, SyntaxError) as exc:
            output[tool.name] = {"error": f"native callable evidence unavailable: {exc}"}
    return output


def comparison_arguments(arguments, metadata):
    """A reward-only copy; list rules retain roles and multiplicity by default."""
    result = copy.deepcopy({**metadata.get("defaults", {}), **arguments})
    for rule in metadata.get("rules", []):
        operation, fields = rule["operation"], rule["fields"]
        if operation == "ignore":
            for field in fields:
                result.pop(field, None)
        elif operation == "lower":
            for field in fields:
                if isinstance(result.get(field), str):
                    result[field] = result[field].lower()
        elif operation in {"multiset", "set"}:
            for field in fields:
                value = result.get(field)
                if value is None and rule.get("none_is_empty"):
                    value = []
                if isinstance(value, list):
                    key = lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
                    ordered = sorted(value, key=key)
                    result[field] = list({key(item): item for item in ordered}.values()) if operation == "set" else ordered
        elif operation == "paired_multiset" and all(isinstance(result.get(f), list) for f in fields):
            if len({len(result[f]) for f in fields}) == 1:
                pairs = sorted(zip(*(result[f] for f in fields), strict=True), key=lambda item: json.dumps(item, sort_keys=True))
                for i, field in enumerate(fields):
                    result[field] = [p[i] for p in pairs]
    return result
