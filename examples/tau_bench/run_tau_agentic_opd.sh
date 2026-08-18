#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT=agentic_opd exec bash "$SCRIPT_DIR/run_tau_training.sh" "$@"
