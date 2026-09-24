import json
from pathlib import Path

import pytest

from aopd.export import checkpoint_identity, export_checkpoint


def checkpoint(tmp_path):
    root = tmp_path / "global_step_1"
    actor = root / "actor"
    actor.mkdir(parents=True)
    (actor / "model_world_size_2_rank_0.pt").write_bytes(b"one")
    (actor / "model_world_size_2_rank_1.pt").write_bytes(b"two")
    return root


def test_incomplete_shards_fail(tmp_path):
    root = checkpoint(tmp_path)
    (root / "actor/model_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_identity(root)


def test_atomic_export_and_identity_checked_reuse(tmp_path, monkeypatch):
    root, target = checkpoint(tmp_path), tmp_path / "export"

    def merge(command, **kwargs):
        output = Path(command[command.index("--target_dir") + 1])
        (output / "config.json").write_text("{}")
        (output / "model.safetensors").write_bytes(b"model")
        (output / "tokenizer.json").write_text("{}")

    monkeypatch.setattr("aopd.export.subprocess.run", merge)
    export_checkpoint(root, target)
    assert json.loads((target / "export_manifest.json").read_text())["checkpoint"]["files"]
    export_checkpoint(root, target)
    (root / "actor/model_world_size_2_rank_0.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        export_checkpoint(root, target)


def test_failed_export_never_publishes_target(tmp_path, monkeypatch):
    root, target = checkpoint(tmp_path), tmp_path / "export"

    def fail(*args, **kwargs):
        raise RuntimeError("merge failed")

    monkeypatch.setattr("aopd.export.subprocess.run", fail)
    with pytest.raises(RuntimeError):
        export_checkpoint(root, target)
    assert not target.exists()
