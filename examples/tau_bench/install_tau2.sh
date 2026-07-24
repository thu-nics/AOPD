#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/.cache/tau2-bench-17e07b1}"
TAU2_COMMIT="17e07b1da2bbc0cadfddeea36412686e0604127b"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Tau Bench requires Python >=3.12"'
if [[ ! -d "$TAU2_ROOT/.git" ]]; then
    git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_ROOT"
fi
if ! git -C "$TAU2_ROOT" rev-parse --verify "$TAU2_COMMIT^{commit}" >/dev/null 2>&1; then
    git -C "$TAU2_ROOT" fetch origin
fi
git -C "$TAU2_ROOT" checkout --detach "$TAU2_COMMIT"

PATCH_FILE="$SCRIPT_DIR/tau2_v1_optional_voice.patch"
if git -C "$TAU2_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    : # Patch already applied.
elif git -C "$TAU2_ROOT" apply --unidiff-zero --check "$PATCH_FILE"; then
    git -C "$TAU2_ROOT" apply --unidiff-zero "$PATCH_FILE"
else
    echo "ERROR: Tau compatibility patch cannot be applied cleanly" >&2
    exit 1
fi

"$PYTHON" -m pip install -e "$TAU2_ROOT[gym,knowledge]" "scipy>=1.10.0"
printf 'Tau source: %s\nSet TAU2_DATA_DIR=%s/data when running qualification or training.\n' \
    "$TAU2_ROOT" "$TAU2_ROOT"
