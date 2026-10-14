"""Frozen release data identity is independent of an archive's checksum manifest."""

import json

import pytest

from actionopd import data


@pytest.fixture(params=["actionopd-fixed-pools-v2", "aopd-fixed-pools-v2"])
def miniature_bundle(tmp_path, monkeypatch, request):
    # Small real files exercise the integrity boundary without downloading data.
    contents = {
        "awm/pool.parquet": b"fixed AWM pool",
        "envscaler/task_audit.jsonl": b"fixed feasibility evidence",
        "awm/deterministic_manifest.json": b"fixed deterministic evidence",
    }
    for relative, content in contents.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    expected = {relative: data.sha256(tmp_path / relative) for relative in contents}
    monkeypatch.setattr(data, "RELEASE_FILE_SHA256", expected)
    return tmp_path, {"protocol": request.param, "files": dict(expected)}


def test_exact_frozen_files_pass_without_external_sources(miniature_bundle):
    root, manifest = miniature_bundle
    data.verify_release_files(root, manifest)


def test_frozen_manifest_cannot_omit_evidence(miniature_bundle):
    root, manifest = miniature_bundle
    del manifest["files"]["envscaler/task_audit.jsonl"]
    with pytest.raises(ValueError, match="file set"):
        data.verify_release_files(root, manifest)


@pytest.mark.parametrize("protocol", ["aopd-fixed-pools-v1", "actionopd-fixed-pools-v1", "unknown"])
def test_previous_bundle_protocol_is_rejected(miniature_bundle, protocol):
    root, manifest = miniature_bundle
    manifest["protocol"] = protocol
    with pytest.raises(ValueError, match="protocol"):
        data.verify_release_files(root, manifest)


def test_frozen_manifest_cannot_add_unreleased_data(miniature_bundle):
    root, manifest = miniature_bundle
    manifest["files"]["unexpected.json"] = "0" * 64
    with pytest.raises(ValueError, match="file set"):
        data.verify_release_files(root, manifest)


@pytest.mark.parametrize("relative", ["awm/pool.parquet", "envscaler/task_audit.jsonl", "awm/deterministic_manifest.json"])
def test_recomputed_manifest_cannot_replace_released_data(miniature_bundle, relative):
    root, manifest = miniature_bundle
    (root / relative).write_bytes(b"different pool, altered evidence, or changed task membership")
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
    path = root / "awm/deterministic_manifest.json"
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="integrity"):
        data.verify_release_files(root, manifest)


def test_public_verify_requires_all_frozen_evidence_before_native_load(tmp_path):
    (tmp_path / "bundle.json").write_text(json.dumps({"protocol": "actionopd-fixed-pools-v2", "files": {}}))
    with pytest.raises(ValueError, match="file set"):
        data.verify(tmp_path)


def test_new_bundle_uses_actionopd_protocol_without_editing_sources(tmp_path, monkeypatch):
    runs = tmp_path / "research-runs"
    for domain, stages in data.FILES.items():
        for stage, names in stages.items():
            for name in names:
                path = runs / f"{domain}_data_processing" / stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"source": {"source_root": "/original/source"}, "artifacts": {"config": {"sha256": "original"}}}))
    before = {str(p.relative_to(runs)): p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    monkeypatch.setattr(data, "verify", lambda output: json.loads((output / "bundle.json").read_text()))
    manifest = data.bundle(runs, tmp_path / "bundle")
    assert manifest["protocol"] == "actionopd-fixed-pools-v2"
    assert {str(p.relative_to(runs)): p.read_bytes() for p in runs.rglob("*") if p.is_file()} == before
