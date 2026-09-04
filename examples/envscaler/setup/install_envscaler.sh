#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
ENVSCALER_ROOT="${ENVSCALER_ROOT:-$REPO_ROOT/../EnvScaler}"
ENVSCALER_COMMIT="87e667397abacf274858c0964796beb8f984aafe"

if [[ "$PYTHON" != */* ]]; then
    PYTHON="$(command -v "$PYTHON" || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON:-<unset>}" >&2
    exit 1
fi

if [[ ! -d "$ENVSCALER_ROOT/.git" ]]; then
    if [[ -e "$ENVSCALER_ROOT" ]] && [[ -n "$(find "$ENVSCALER_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: ENVSCALER_ROOT exists and is not an empty Git checkout: $ENVSCALER_ROOT" >&2
        exit 1
    fi
    git clone --no-checkout https://github.com/RUC-NLPIR/EnvScaler.git "$ENVSCALER_ROOT"
    git -C "$ENVSCALER_ROOT" checkout --detach "$ENVSCALER_COMMIT"
fi

actual_commit="$(git -C "$ENVSCALER_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$ENVSCALER_COMMIT" ]]; then
    echo "ERROR: $ENVSCALER_ROOT is at $actual_commit, expected $ENVSCALER_COMMIT" >&2
    echo "Use a fresh ENVSCALER_ROOT; this script will not overwrite an existing checkout." >&2
    exit 1
fi
if [[ -n "$(git -C "$ENVSCALER_ROOT" status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: $ENVSCALER_ROOT has tracked modifications; refusing an unpinned source checkout" >&2
    git -C "$ENVSCALER_ROOT" status --short --untracked-files=no >&2
    exit 1
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'from agent_system.environments.env_package.envscaler.source import validate_envscaler_source; import sys; validate_envscaler_source(sys.argv[1])' "$ENVSCALER_ROOT"

printf 'EnvScaler source: %s\nCommit: %s\n' "$ENVSCALER_ROOT" "$ENVSCALER_COMMIT"
