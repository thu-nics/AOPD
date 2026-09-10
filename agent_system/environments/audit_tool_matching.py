"""Read-only native tool inventory; writes a report, never rewrites task pools.

Usage: python -m agent_system.environments.audit_tool_matching --output runs/.../audit.json
Optional --awm-pool/--envscaler-pool scope the inventory to training environments.
Pattern counts are audit leads, NOT proven equivalence bugs or failure counts.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from agent_system.environments.tool_matching_metadata import source_tool_matching_metadata


def _pool_envs(path):
    if not path:
        return None
    import pandas as pd

    return {str(info.get("scenario") or info.get("env_id")) for info in pd.read_parquet(path)["extra_info"]}


def _inventory(source, tools, family, environment, class_name=None):
    metadata = source_tool_matching_metadata(source, tools, family=family, environment=environment, class_name=class_name)
    rows = []
    for name, entry in metadata.items():
        code = entry.get("source", "")
        rows.append(
            {
                "environment": environment,
                "tool": name,
                "source_hash": entry.get("source_hash"),
                "function_hash": entry.get("function_hash"),
                "source_chars": len(code),
                "source_error": entry.get("error"),
                "program_rules": entry.get("rules", []),
                "audit_leads": [
                    label
                    for label, markers in {
                        "unordered_operations": ("set(", "sorted(", ".sort("),
                        "string_normalization": (".lower(", ".strip(", ".casefold("),
                        "paired_inputs": ("zip(",),
                        "presence_sensitive": ("model_fields_set", "__fields_set__", "exclude_unset"),
                    }.items()
                    if any(marker in code for marker in markers)
                ],
            }
        )
    return rows


def main():
    siblings = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awm-data-dir", type=Path, default=siblings / "openenv-awm-cache")
    parser.add_argument("--envscaler-root", type=Path, default=siblings / "EnvScaler")
    parser.add_argument("--awm-pool", type=Path)
    parser.add_argument("--envscaler-pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    awm_envs, envscaler_envs = _pool_envs(args.awm_pool), _pool_envs(args.envscaler_pool)
    report = {"note": "Static source inventory, not task validation or proof that every audit lead is a bug. AWM names are read from endpoint declarations, not live MCP schema validation.", "families": {}}
    rows = []
    with (args.awm_data_dir / "gen_envs.jsonl").open() as stream:
        for raw in stream:
            env = json.loads(raw)
            if awm_envs is not None and env["scenario"] not in awm_envs:
                continue
            source = env["full_code"]
            tools = []
            for node in ast.parse(source).body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in {"get", "post", "patch", "put", "delete"}:
                        name = next((kw.value.value for kw in dec.keywords if kw.arg == "operation_id" and isinstance(kw.value, ast.Constant)), node.name)
                        tools.append({"name": name})
                        break
            rows.extend(_inventory(source, tools, "awm", env["scenario"]))
    report["families"]["awm"] = rows
    metadata_path = args.envscaler_root / "interact_with_env/envscaler_env/data/191_env_metadata.json"
    rows = []
    for name, env in json.loads(metadata_path.read_text()).items():
        if envscaler_envs is None or name in envscaler_envs:
            rows.extend(_inventory(env["env_class_code"], env["tools"], "envscaler", name, env["env_class_name"]))
    report["families"]["envscaler"] = rows
    report["summary"] = {}
    for family, rows in report["families"].items():
        report["summary"][family] = {
            "environments": len({r["environment"] for r in rows}),
            "tools": len(rows),
            "source_errors": dict(Counter(r["source_error"] for r in rows if r["source_error"])),
            "reviewed_rule_tools": sum(bool(r["program_rules"]) for r in rows),
            "audit_leads_not_bugs": dict(Counter(label for r in rows for label in r["audit_leads"])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
