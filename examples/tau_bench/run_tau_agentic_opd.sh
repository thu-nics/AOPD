#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=agentic_opd exec bash "$SCRIPT_DIR/train/run.sh" "$@"
