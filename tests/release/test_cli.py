import json
import subprocess
import sys

import yaml
from test_recipes import runtime


def test_cli_check_emits_portable_plan_without_creating_run(tmp_path):
    source = tmp_path / "runtime.yaml"
    source.write_text(yaml.safe_dump(runtime()))
    run = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-m", "aopd", "train", "tau-full", "--runtime", str(source), "--run-dir", str(run), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["env"]["CUDA_VISIBLE_DEVICES"] == "4,7"
    assert not run.exists()
