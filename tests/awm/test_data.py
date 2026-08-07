import importlib
import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    path = Path("examples/awm/data/prepare_data.py")
    spec = importlib.util.spec_from_file_location("awm_prepare_data_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records():
    tasks = [
        {
            "scenario": f"scenario_{scenario}",
            "tasks": [f"task {scenario} {task}" for task in range(10)],
        }
        for scenario in range(1000)
    ]
    verifiers = [
        {
            "scenario": f"scenario_{scenario}",
            "task_idx": task,
            "verification": {"code": "def verify(): return True"},
        }
        for scenario in range(1000)
        for task in range(10)
    ]
    return tasks, verifiers


def test_public_revision_expands_to_10000_and_splits_by_id_only():
    module = _load_prepare_module()
    tasks, verifiers = _records()
    rows = module._validate_and_expand(tasks, verifiers)
    splits = module._select_splits(rows)
    assert len(rows) == 10000
    assert (len({row["scenario"] for row in splits["dev"]}), len(splits["dev"])) == (
        32,
        256,
    )
    assert (
        len({row["scenario"] for row in splits["smoke"]}),
        len(splits["smoke"]),
    ) == (4, 8)
    assert {row["task_id"] for row in splits["smoke"]} <= {row["task_id"] for row in splits["dev"]}


def test_missing_code_verifier_fails_loudly():
    module = _load_prepare_module()
    tasks, verifiers = _records()
    with pytest.raises(RuntimeError, match="missing pure-code verifier"):
        module._validate_and_expand(tasks, verifiers[:-1])


def test_source_hash_mismatch_fails_loudly():
    module = _load_prepare_module()
    with pytest.raises(RuntimeError, match="pinned dataset revision"):
        module._validate_source_hashes({"gen_tasks.jsonl": "not-pinned"})


def test_reorganized_awm_namespace_subpackages_are_importable():
    modules = {
        "agent_system.environments.env_package.awm.data",
        "agent_system.environments.env_package.awm.evaluation",
        "agent_system.environments.env_package.awm.runtime",
        "agent_system.environments.env_package.awm.screening",
    }
    for module in modules:
        imported = importlib.import_module(module)
        assert imported.__file__ is not None
