#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=outcome exec bash "$SCRIPT_DIR/train/run.sh" "$@"
