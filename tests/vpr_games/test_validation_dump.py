import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_dump_generations_serializes_validation_metadata(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 25

    trainer._dump_generations(
        inputs=["prompt"],
        outputs=["<action>right</action>"],
        scores=[1.0],
        reward_extra_infos_dict={
            "traj_uid": ["trajectory-1"],
            "turn_index": [np.int32(2)],
            "terminal_success": [np.bool_(True)],
            "token_counts": [torch.tensor([3, 4])],
        },
        dump_path=str(tmp_path),
    )

    record = json.loads((tmp_path / "25.jsonl").read_text().strip())
    assert record["traj_uid"] == "trajectory-1"
    assert record["turn_index"] == 2
    assert record["terminal_success"] is True
    assert record["token_counts"] == [3, 4]


def test_val_only_checkpoint_load_skips_training_dataloader(tmp_path):
    checkpoint = tmp_path / "global_step_25"
    (checkpoint / "actor").mkdir(parents=True)
    torch.save({"incompatible": "training state"}, checkpoint / "data.pt")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "trainer": {
            "resume_mode": "resume_path",
            "resume_from_path": str(checkpoint),
            "default_hdfs_dir": None,
            "default_local_dir": str(tmp_path),
            "del_local_ckpt_after_load": False,
            "val_only": True,
        }
    })
    trainer.actor_rollout_wg = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.use_critic = False
    trainer.global_steps = 0

    trainer._load_checkpoint()

    trainer.actor_rollout_wg.load_checkpoint.assert_called_once()
    trainer.train_dataloader.load_state_dict.assert_not_called()
    assert trainer.global_steps == 25


def test_val_only_ignores_val_before_train_false():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "trainer": {
            "project_name": "test",
            "experiment_name": "val-only",
            "logger": ["console"],
            "val_before_train": False,
            "val_only": True,
        }
    })
    trainer.val_reward_fn = object()
    trainer._load_checkpoint = MagicMock()
    trainer._validate = MagicMock(return_value={"val/env/success_rate": 0.5})

    with patch("verl.utils.tracking.Tracking") as tracking:
        trainer.fit()

    trainer._validate.assert_called_once_with()
    tracking.return_value.log.assert_called_once_with(
        data={"val/env/success_rate": 0.5}, step=0
    )
