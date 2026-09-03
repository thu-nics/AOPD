#!/usr/bin/env bash
# Backward-compatible path. New workflows use ../data/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../data/run_static_feasibility_judge.sh" "$@"
