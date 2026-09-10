"""Repair MCP's contradictory schemas without rewriting argument payloads."""

from __future__ import annotations

ACTION_SCHEMA_PROTOCOL_VERSION = 1


def repair_mcp_tool_schemas(mcp) -> None:
    from .actions import normalize_tools

    for tool in mcp.tools:
        canonical = normalize_tools([{"name": tool.name, "inputSchema": tool.inputSchema}])[0]
        tool.inputSchema = canonical["inputSchema"]


def install_action_schema_patch() -> None:
    """Patch only this server's generated subprocess code, not upstream files."""
    from agent_world_model_env.server import scenario_manager

    if getattr(scenario_manager._patch_env_code, "_awm_action_schema_patch", False):
        return
    original = scenario_manager._patch_env_code

    def patched(*args, **kwargs):
        code = original(*args, **kwargs)
        marker = "mcp.mount_http()"
        if code.count(marker) != 1:
            raise RuntimeError("AWM MCP injection changed: cannot install action schema repair")
        return code.replace(marker, "from agent_system.environments.env_package.awm.runtime.tool_schema import repair_mcp_tool_schemas\n    repair_mcp_tool_schemas(mcp)\n    mcp.mount_http()")

    patched._awm_action_schema_patch = True
    scenario_manager._patch_env_code = patched
