#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export ENABLE_ENVSCALER=1
export VARIANT=agentic_opd
if [[ "${SHUFFLE:-false}" != "false" ]]; then
    echo "ERROR: mixed training requires SHUFFLE=false" >&2
    exit 1
fi
exec bash "$REPO_ROOT/examples/awm/train/run_training.sh" "$@"
