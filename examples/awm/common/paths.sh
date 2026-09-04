#!/usr/bin/env bash
# Shared, overridable defaults for an AWM development installation.

AWM_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWM_REPO_ROOT="$(cd "$AWM_COMMON_DIR/../../.." && pwd)"
VENV_PATH="${VENV_PATH:-}"
AWM_SOURCE_DIR="${AWM_SOURCE_DIR:-$AWM_REPO_ROOT/../openenv-awm}"
AWM_CACHE_DIR="${AWM_CACHE_DIR:-$AWM_REPO_ROOT/../openenv-awm-cache}"

if [[ -n "$VENV_PATH" ]]; then
    PYTHON="${PYTHON:-$VENV_PATH/bin/python}"
else
    PYTHON="${PYTHON:-$(command -v python || true)}"
fi
if [[ -n "$PYTHON" && "$PYTHON" != */* ]]; then
    PYTHON="$(command -v "$PYTHON" || true)"
fi
OPENENV_ROOT="${OPENENV_ROOT:-$AWM_SOURCE_DIR}"
AWM_DATA_DIR="${AWM_DATA_DIR:-$AWM_CACHE_DIR}"

unset AWM_COMMON_DIR AWM_REPO_ROOT
