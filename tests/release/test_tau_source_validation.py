"""Validate actual Git trees without touching the shared checkout or index."""

import subprocess

import pytest

from agent_system.environments.env_package.tau_bench import envs


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT)


@pytest.fixture
def tau_checkout(tmp_path, monkeypatch):
    root = tmp_path / "tau"
    root.mkdir()
    contents = {
        "src/tau2/__init__.py": "VOICE_ENABLED = True\n",
        "src/tau2/environment.py": "REWARD = 1\n",
        "data/tau2/domains/airline/tasks.json": '[{"id": "a", "goal": "original"}]',
        "data/tau2/domains/retail/tasks.json": '[{"id": "r", "goal": "original"}]',
        "data/tau2/domains/telecom/tasks.json": '[{"id": "t", "goal": "original"}]',
        "data/tau2/user_simulator/simulation_guidelines.md": "original instructions\n",
        "data/tau2/user_simulator/simulation_guidelines_tools.md": "original tools\n",
        "data/tau2/domains/banking/tasks.json": "[]",
        "web/leaderboard.html": "unneeded for scientific protocol",
    }
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(root, "init", "--quiet")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "pinned fixture")
    commit = _git(root, "rev-parse", "HEAD").strip()
    patched = root / "src/tau2/__init__.py"
    patched.write_text("VOICE_ENABLED = False\n")
    patch = tmp_path / "approved.patch"
    patch.write_text(_git(root, "diff", "--binary"))
    _git(root, "sparse-checkout", "init", "--cone")
    _git(root, "sparse-checkout", "set", "src", "data/tau2/domains/airline", "data/tau2/domains/retail", "data/tau2/domains/telecom", "data/tau2/user_simulator")
    monkeypatch.setattr(envs, "tau_source_root", lambda: root)
    monkeypatch.setattr(envs, "TAU2_COMMIT", commit)
    monkeypatch.setattr(envs, "compatibility_patch_path", lambda: patch)
    return root, commit


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}


def test_exact_approved_patch_accepts_sparse_checkout_without_mutation(tau_checkout):
    root, commit = tau_checkout
    assert not (root / "web").exists()
    assert not (root / "data/tau2/domains/banking").exists()
    before, index = _snapshot(root), (root / ".git/index").read_bytes()
    git_before = {p.relative_to(root): p.read_bytes() for p in (root / ".git").rglob("*") if p.is_file()}
    assert envs.validate_tau_source(root)["tau2_commit"] == commit
    assert _snapshot(root) == before
    assert (root / ".git/index").read_bytes() == index
    assert {p.relative_to(root): p.read_bytes() for p in (root / ".git").rglob("*") if p.is_file()} == git_before


@pytest.mark.parametrize(
    "relative",
    [
        "src/tau2/__init__.py",
        "src/tau2/environment.py",
        "data/tau2/domains/airline/tasks.json",
        "data/tau2/domains/retail/tasks.json",
        "data/tau2/domains/telecom/tasks.json",
        "data/tau2/user_simulator/simulation_guidelines.md",
    ],
)
@pytest.mark.parametrize("staged", [False, True])
def test_extra_scientific_edits_are_rejected_without_mutation(tau_checkout, relative, staged):
    root, _ = tau_checkout
    target = root / relative
    target.write_text(target.read_text() + "\nunauthorized change\n")
    if staged:
        _git(root, "add", relative)
    before, index = _snapshot(root), (root / ".git/index").read_bytes()
    with pytest.raises(RuntimeError, match="scientific|pinned source|tracked"):
        envs.validate_tau_source(root)
    assert _snapshot(root) == before
    assert (root / ".git/index").read_bytes() == index


def test_missing_tracked_scientific_file_is_rejected(tau_checkout):
    root, _ = tau_checkout
    (root / "src/tau2/environment.py").unlink()
    with pytest.raises(RuntimeError, match="scientific|pinned source|tracked"):
        envs.validate_tau_source(root)
