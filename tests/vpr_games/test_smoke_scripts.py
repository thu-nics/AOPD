"""Executable preflight tests for the GRPO smoke scripts.

These do NOT launch training (no GPU / model needed). They verify that a
nonexistent MODEL_PATH must fail fast with a clear file-not-found error rather
than hanging or silently proceeding. The scripts
read MODEL_PATH from the environment (overridable default), so we point it at a
path that does not exist and assert a quick, clear failure.
"""
import os
import subprocess

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SCRIPT_DIR = os.path.join(_REPO_ROOT, "examples", "vpr_games", "smoke")

_SCRIPTS = [
    "grpo_tictactoe_smoke.sh",
    "grpo_sudoku_smoke.sh",
    "grpo_minesweeper_smoke.sh",
    "grpo_sokoban_smoke.sh",
]


@pytest.mark.parametrize("script", _SCRIPTS)
def test_nonexistent_model_path_fails_fast(script):
    script_path = os.path.join(_SCRIPT_DIR, script)
    assert os.path.exists(script_path), f"Missing smoke script: {script_path}"

    env = dict(os.environ)
    env["MODEL_PATH"] = "/definitely/not/a/real/model/path"
    # Keep PYTHON pointed at a real interpreter so the only failure is the model
    # check; the model check runs before the python check anyway.

    try:
        r = subprocess.run(
            ["bash", script_path],
            cwd=_SCRIPT_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,  # must fail fast; a hang would blow this budget
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script} hung on a nonexistent MODEL_PATH instead of failing fast")

    assert r.returncode != 0, f"{script} should exit non-zero for a missing model"
    combined = (r.stdout + r.stderr).lower()
    assert "model not found" in combined, (
        f"{script} did not report a clear model-not-found error.\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")
