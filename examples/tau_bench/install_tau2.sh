#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TAU2_ROOT="${TAU2_ROOT:-/mnt/public2/yuanhuining/repos/tau2-bench}"
TAU2_COMMIT="17e07b1da2bbc0cadfddeea36412686e0604127b"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Tau Bench requires Python >=3.12"'
if [[ ! -d "$TAU2_ROOT/.git" ]]; then
    if [[ -e "$TAU2_ROOT" ]] && [[ -n "$(find "$TAU2_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: TAU2_ROOT exists and is not an empty Git checkout: $TAU2_ROOT" >&2
        exit 1
    fi
    mkdir -p "$TAU2_ROOT"
    git -C "$TAU2_ROOT" init
    git -C "$TAU2_ROOT" remote add origin https://github.com/sierra-research/tau2-bench.git
fi
if ! git -C "$TAU2_ROOT" rev-parse --verify "$TAU2_COMMIT^{commit}" >/dev/null 2>&1; then
    git -c http.version=HTTP/1.1 -C "$TAU2_ROOT" fetch \
        --depth 1 origin "$TAU2_COMMIT"
fi
git -C "$TAU2_ROOT" checkout --detach "$TAU2_COMMIT"
if [[ "$(git -C "$TAU2_ROOT" config --bool core.sparseCheckout || true)" == "true" ]]; then
    git -C "$TAU2_ROOT" sparse-checkout add data/tau2/user_simulator
fi
for guideline in simulation_guidelines.md simulation_guidelines_tools.md; do
    if [[ ! -f "$TAU2_ROOT/data/tau2/user_simulator/$guideline" ]]; then
        echo "ERROR: pinned Tau checkout is missing data/tau2/user_simulator/$guideline" >&2
        exit 1
    fi
done

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
printf 'Tau source: %s\nSet TAU2_DATA_DIR=%s/data when running training or evaluation.\n' \
    "$TAU2_ROOT" "$TAU2_ROOT"
