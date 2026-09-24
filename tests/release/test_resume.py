import pytest

from aopd.resume import validate_training_checkpoint


def saved_checkpoint(tmp_path):
    root = tmp_path / "global_step_10"
    actor = root / "actor"
    actor.mkdir(parents=True)
    for kind in ("model", "optim", "extra_state"):
        for rank in range(2):
            (actor / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"saved")
    (root / "data.pt").write_bytes(b"dataloader")
    return root


def test_complete_checkpoint_same_world_size(tmp_path):
    validate_training_checkpoint(saved_checkpoint(tmp_path), 2)


def test_resume_cannot_change_fsdp_world_size(tmp_path):
    with pytest.raises(ValueError, match="world size"):
        validate_training_checkpoint(saved_checkpoint(tmp_path), 8)


@pytest.mark.parametrize("missing", ["optim_world_size_2_rank_1.pt", "extra_state_world_size_2_rank_0.pt"])
def test_resume_rejects_missing_auxiliary_shard(tmp_path, missing):
    root = saved_checkpoint(tmp_path)
    (root / "actor" / missing).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        validate_training_checkpoint(root, 2)
