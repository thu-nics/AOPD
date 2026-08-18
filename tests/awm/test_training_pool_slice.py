import json

import pandas as pd
import pytest

import agent_system.environments.env_package.awm.data.pools as verification


def _source_pool(tmp_path, tasks=10):
    task_ids = [f"scenario:{index}" for index in range(tasks)]
    data = tmp_path / "awm_training_pool.parquet"
    manifest = tmp_path / "integrity_manifest.json"
    pd.DataFrame([{"extra_info": {"task_id": task_id}} for task_id in task_ids]).to_parquet(data, index=False)
    manifest.write_text(
        json.dumps(
            {
                "selection_counts": {"tasks": tasks},
                "training_pool_task_ids": task_ids,
            }
        )
    )
    return data, manifest, task_ids


def test_fraction_materializes_hash_bound_ordered_prefix(tmp_path, monkeypatch):
    data, manifest, task_ids = _source_pool(tmp_path)
    monkeypatch.setattr(
        verification,
        "verify_training_pool",
        lambda *_: {
            "kind": "healthy_training_pool",
            "tasks": len(task_ids),
        },
    )
    output_data = tmp_path / "run" / "awm_training_pool_slice.parquet"
    output_manifest = tmp_path / "run" / "training_slice_manifest.json"

    result = verification.materialize_training_slice(
        source_data=data,
        source_manifest_path=manifest,
        output_data=output_data,
        output_manifest_path=output_manifest,
        fraction="0.3",
    )

    assert result["tasks"] == 3
    frame = pd.read_parquet(output_data)
    assert [item["task_id"] for item in frame["extra_info"]] == task_ids[:3]
    recorded = json.loads(output_manifest.read_text())
    assert recorded["requested_fraction"] == "0.3"
    assert recorded["fraction_base_tasks"] == 10
    assert recorded["task_ids"] == task_ids[:3]
    assert (
        verification.materialize_training_slice(
            source_data=data,
            source_manifest_path=manifest,
            output_data=output_data,
            output_manifest_path=output_manifest,
            fraction="0.30",
        )
        == result
    )


def test_explicit_task_count_and_invalid_selection_are_checked(tmp_path, monkeypatch):
    data, manifest, task_ids = _source_pool(tmp_path)
    monkeypatch.setattr(
        verification,
        "verify_training_pool",
        lambda *_: {
            "kind": "healthy_training_pool",
            "tasks": len(task_ids),
        },
    )
    result = verification.materialize_training_slice(
        source_data=data,
        source_manifest_path=manifest,
        output_data=tmp_path / "count.parquet",
        output_manifest_path=tmp_path / "count.json",
        task_count=4,
    )
    assert result["tasks"] == 4
    with pytest.raises(ValueError, match="exceeds"):
        verification._slice_spec(
            json.loads(manifest.read_text()),
            available_tasks=10,
            task_count=11,
            fraction=None,
        )


def test_exact_schedule_cycles_all_source_tasks_without_drop_last_loss(tmp_path, monkeypatch):
    data, manifest, task_ids = _source_pool(tmp_path, tasks=5)

    def verified_source(*_):
        return pd.read_parquet(data), task_ids

    monkeypatch.setattr(
        verification,
        "_verified_schedule_source",
        verified_source,
    )
    output_data = tmp_path / "run" / "schedule.parquet"
    output_manifest = tmp_path / "run" / "schedule.json"

    result = verification.materialize_training_schedule(
        source_data=data,
        source_manifest_path=manifest,
        output_data=output_data,
        output_manifest_path=output_manifest,
        train_steps=2,
        train_batch_size=4,
    )

    assert result["rows"] == 8
    frame = pd.read_parquet(output_data)
    assert [item["task_id"] for item in frame["extra_info"]] == [
        *task_ids,
        *task_ids[:3],
    ]
    recorded = json.loads(output_manifest.read_text())
    assert recorded["complete_source_passes"] == 1
    assert recorded["partial_next_pass_tasks"] == 3
    assert recorded["minimum_task_occurrences"] == 1
    assert recorded["maximum_task_occurrences"] == 2
    assert recorded["protocol_version"] == 2
    assert recorded["schedule_coordinates"] == "zero_based_step_and_slot"
    assert [item["schedule_step"] for item in frame["env_kwargs"]] == [0] * 4 + [1] * 4
    assert [item["schedule_slot"] for item in frame["env_kwargs"]] == [0, 1, 2, 3] * 2
    assert [item["schedule_step"] for item in frame["extra_info"]] == [0] * 4 + [1] * 4
