"""Frozen release data identity is independent of an archive's checksum manifest."""

import json

import pytest

from aopd import data


@pytest.fixture
def miniature_bundle(tmp_path, monkeypatch):
    # Small real files exercise the integrity boundary without downloading data.
    contents = {
        "awm/pool.parquet": b"fixed AWM pool",
        "envscaler/task_audit.jsonl": b"fixed feasibility evidence",
        "tau/customer_briefs.json": b"reviewed training-only briefs",
    }
    for relative, content in contents.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    expected = {relative: data.sha256(tmp_path / relative) for relative in contents}
    monkeypatch.setattr(data, "RELEASE_FILE_SHA256", expected)
    return tmp_path, {"protocol": "aopd-fixed-pools-v1", "files": dict(expected)}


def test_exact_frozen_files_pass_without_external_sources(miniature_bundle):
    root, manifest = miniature_bundle
    data.verify_release_files(root, manifest)


def test_frozen_manifest_cannot_omit_evidence(miniature_bundle):
    root, manifest = miniature_bundle
    del manifest["files"]["envscaler/task_audit.jsonl"]
    with pytest.raises(ValueError, match="file set"):
        data.verify_release_files(root, manifest)


@pytest.mark.parametrize("relative", ["awm/pool.parquet", "envscaler/task_audit.jsonl", "tau/customer_briefs.json"])
def test_recomputed_manifest_cannot_replace_released_data(miniature_bundle, relative):
    root, manifest = miniature_bundle
    (root / relative).write_bytes(b"different pool, altered evidence, or held-out evaluation briefs")
    manifest["files"][relative] = data.sha256(root / relative)
    with pytest.raises(ValueError, match="released identity"):
        data.verify_release_files(root, manifest)


def test_frozen_file_bytes_are_checked(miniature_bundle):
    root, manifest = miniature_bundle
    (root / "envscaler/task_audit.jsonl").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        data.verify_release_files(root, manifest)


def test_frozen_symlink_cannot_escape_bundle(miniature_bundle, tmp_path):
    root, manifest = miniature_bundle
    path = root / "tau/customer_briefs.json"
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="integrity"):
        data.verify_release_files(root, manifest)


def test_public_verify_requires_all_frozen_evidence_before_native_load(tmp_path):
    (tmp_path / "bundle.json").write_text(json.dumps({"protocol": "aopd-fixed-pools-v1", "files": {}}))
    with pytest.raises(ValueError, match="file set"):
        data.verify(tmp_path)
