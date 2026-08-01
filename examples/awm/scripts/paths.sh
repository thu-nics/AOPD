#!/usr/bin/env bash
# Shared, overridable defaults for the local AWM development installation.

VENV_PATH="${VENV_PATH:-/opt/venvs/verl-agent}"
AWM_SOURCE_DIR="${AWM_SOURCE_DIR:-/mnt/public2/yuanhuining/repos/openenv-awm}"
AWM_CACHE_DIR="${AWM_CACHE_DIR:-/mnt/public2/yuanhuining/repos/openenv-awm-cache}"

PYTHON="${PYTHON:-$VENV_PATH/bin/python}"
OPENENV_ROOT="${OPENENV_ROOT:-$AWM_SOURCE_DIR}"
AWM_DATA_DIR="${AWM_DATA_DIR:-$AWM_CACHE_DIR}"
