#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT=vpr exec bash "$SCRIPT_DIR/run_tau_training.sh" "$@"
