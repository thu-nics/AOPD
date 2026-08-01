#!/usr/bin/env python3
"""Serve pinned AWM with a deterministic, repository-owned logical clock."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .logical_time import install_logical_time

DATA_DIR = Path(os.environ["AWM_DATA_DIR"])
POLICY = install_logical_time(DATA_DIR)

# Import only after patching AWMDataLoader; app.py constructs its shared loader
# at import time.
from agent_world_model_env.server.app import app  # noqa: E402


@app.get("/awm-logical-time", tags=["protocol"])
async def logical_time_protocol():
    return POLICY.protocol()


@app.get("/awm-logical-time/{scenario}", tags=["protocol"])
async def scenario_logical_time(scenario: str):
    return POLICY.scenario_record(scenario)


def main() -> None:
    protocol = POLICY.protocol()
    print(
        f"AWM logical-time protocol v{protocol['protocol_version']} sha256={protocol['scenario_times_sha256']}",
        flush=True,
    )
    uvicorn.run(
        app,
        host=os.environ.get("AWM_HOST", "127.0.0.1"),
        port=int(os.environ.get("AWM_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
