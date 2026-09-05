#!/usr/bin/env python3
"""Serve pinned AWM with a deterministic, repository-owned logical clock."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .logical_time import install_logical_time
from .terminal_judge import (
    install_terminal_judge_transport,
    terminal_judge_protocol,
)

DATA_DIR = Path(os.environ["AWM_DATA_DIR"])
POLICY = install_logical_time(DATA_DIR)
TERMINAL_JUDGE = install_terminal_judge_transport()
RUN_ID = os.environ.get("AWM_SERVER_RUN_ID", "standalone")

# Import only after patching AWMDataLoader; app.py constructs its shared loader
# at import time.
from agent_world_model_env.server.app import app  # noqa: E402


@app.get("/awm-logical-time", tags=["protocol"])
async def logical_time_protocol():
    return POLICY.protocol()


@app.get("/awm-logical-time/{scenario}", tags=["protocol"])
async def scenario_logical_time(scenario: str):
    return POLICY.scenario_record(scenario)


@app.get("/awm-run-identity", tags=["protocol"])
async def run_identity_protocol():
    return {"run_id": RUN_ID}


@app.get("/awm-terminal-judge", tags=["protocol"])
async def terminal_judge_identity_protocol():
    return terminal_judge_protocol()


def main() -> None:
    protocol = POLICY.protocol()
    print(
        f"AWM logical-time protocol v{protocol['protocol_version']} sha256={protocol['scenario_times_sha256']}",
        flush=True,
    )
    print(
        f"AWM terminal-judge protocol v{TERMINAL_JUDGE['protocol_version']} model={TERMINAL_JUDGE['model']}",
        flush=True,
    )
    uvicorn.run(
        app,
        host=os.environ.get("AWM_HOST", "127.0.0.1"),
        port=int(os.environ.get("AWM_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
