#!/usr/bin/env bash
# Backward-compatible path. New workflows use ../diagnostics/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../diagnostics/run_expert_screening.sh" "$@"
