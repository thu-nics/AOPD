#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/opt/venvs/verl-agent-sokoban/bin/python}"
OPENENV_ROOT="${OPENENV_ROOT:-/opt/src/openenv-awm}"
AWM_DATA_DIR="${AWM_DATA_DIR:-$HOME/.cache/openenv/awm}"
AWM_HOST="${AWM_HOST:-127.0.0.1}"
AWM_PORT="${AWM_PORT:-8000}"
OPENENV_COMMIT="5298e0d91c6cd55d5f3a81259d5b2a9a1e05eff0"

if [[ ! -d "$OPENENV_ROOT/.git" ]]; then
    echo "ERROR: OpenEnv checkout not found at $OPENENV_ROOT; run install_awm.sh" >&2
    exit 1
fi
actual_commit="$(git -C "$OPENENV_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$OPENENV_COMMIT" ]]; then
    echo "ERROR: $OPENENV_ROOT is at $actual_commit, expected $OPENENV_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git -C "$OPENENV_ROOT" status --porcelain)" ]]; then
    echo "ERROR: $OPENENV_ROOT has local modifications; the pinned AWM runtime must be clean" >&2
    git -C "$OPENENV_ROOT" status --short >&2
    exit 1
fi
for filename in gen_scenario.jsonl gen_tasks.jsonl gen_db.jsonl gen_sample.jsonl gen_envs.jsonl gen_verifier.jsonl gen_verifier.pure_code.jsonl dataset_identity.json; do
    if [[ ! -f "$AWM_DATA_DIR/$filename" ]]; then
        echo "ERROR: missing $AWM_DATA_DIR/$filename" >&2
        echo "Run examples/awm/prepare_data.py for the pinned dataset revision." >&2
        exit 1
    fi
done
"$PYTHON" - "$AWM_DATA_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

expected_identity = {
    "dataset": "Snowflake/AgentWorldModel-1K",
    "dataset_revision": "dde80a0283fe781bdc51656bce57063dc5650213",
    "source_sha256": {
        "gen_db.jsonl": "ae8acb3c23765ca4866b35799ffb980fbb15831240fdc35c046e8a7d27a2c0e8",
        "gen_envs.jsonl": "2c7749c1710303f0f663bbe14aea689ade77c19282e2dd0ef0c54e2a95f5e7d8",
        "gen_sample.jsonl": "39c40969ad76d52a3ea51384752a639cf55dce15bc1a8f0f022cfb6bbd25db3c",
        "gen_scenario.jsonl": "6362e31af6e39bc914c6606b32f62b3d5f571f9653e86a6a9c29e507ae0e647e",
        "gen_tasks.jsonl": "0537871c8824cd56d23cc51118294ae3c3070d990cb9a3f766f9dc690f91bf45",
        "gen_verifier.jsonl": "269a97a085dce103afed6e5c48d001b8d47a9e7e2b142f2edc0ddaf6da08a348",
        "gen_verifier.pure_code.jsonl": "2de0b668bd0c6b37a033dda7d697bcd8ddc3bf5b0c9bf83fbd691fbd2c3827f7",
    },
}
root = Path(sys.argv[1])
identity = json.loads((root / "dataset_identity.json").read_text())
if identity != expected_identity:
    raise SystemExit("AWM dataset_identity.json does not match the pinned protocol")
for filename, expected in expected_identity["source_sha256"].items():
    digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"AWM dataset hash mismatch: {filename}")
PY

export AWM_DATA_DIR
export PYTHONPATH="$OPENENV_ROOT/src:$OPENENV_ROOT/envs${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m uvicorn agent_world_model_env.server.app:app \
    --host "$AWM_HOST" --port "$AWM_PORT"
