#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

OPENENV_COMMIT="5298e0d91c6cd55d5f3a81259d5b2a9a1e05eff0"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter is not executable: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$OPENENV_ROOT/.git" ]]; then
    git clone --no-checkout https://github.com/meta-pytorch/OpenEnv.git "$OPENENV_ROOT"
    git -C "$OPENENV_ROOT" checkout --detach "$OPENENV_COMMIT"
fi
actual_commit="$(git -C "$OPENENV_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$OPENENV_COMMIT" ]]; then
    echo "ERROR: $OPENENV_ROOT is at $actual_commit, expected $OPENENV_COMMIT" >&2
    echo "Use a fresh OPENENV_ROOT; this script will not overwrite an existing checkout." >&2
    exit 1
fi
if [[ -n "$(git -C "$OPENENV_ROOT" status --porcelain)" ]]; then
    echo "ERROR: $OPENENV_ROOT has local modifications; refusing an unpinned editable install" >&2
    git -C "$OPENENV_ROOT" status --short >&2
    exit 1
fi

"$PYTHON" -m pip install -e "$OPENENV_ROOT"
"$PYTHON" -m pip install -e "$OPENENV_ROOT/envs/agent_world_model_env"
# OpenEnv's unconstrained Gradio dependency may otherwise upgrade this past
# Transformers' supported range in an existing training environment.
"$PYTHON" -m pip install "gradio==6.15.0" "huggingface-hub>=0.34,<1.0"
"$PYTHON" - <<'PY'
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction

print("AWM client import OK", AWMEnv, CallToolAction)
PY
