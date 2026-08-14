#!/usr/bin/env bash
# Backward-compatible entry point. New workflows should use the explicitly
# named static-feasibility script below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_static_feasibility_judge.sh" "$@"
