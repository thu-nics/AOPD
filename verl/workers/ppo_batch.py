"""Small FSDP actor/critic batch helpers; preserve complete optimizer minibatches."""

import torch

from verl import DataProto


def training_response_mask(data: DataProto, *, multi_turn=False):
    batch = data.batch
    length = batch["responses"].shape[-1]
    mask = batch["attention_mask"][:, -length:].bool().clone()
    if "response_mask" in batch:
        mask &= batch["response_mask"].bool()
    if multi_turn and "loss_mask" in batch:
        mask &= batch["loss_mask"][:, -length:].bool()
    # These are explicit estimator metadata, never inferred from zero advantages
    # or duplicate row contents. They also apply when multi_turn is disabled.
    for key in ("is_padding", "state_group_skip_loss", "outcome_skip_loss", "dapo_skip_loss", "vpr_skip_loss", "vineppo_skip_loss"):
        if key in data.non_tensor_batch:
            skip = torch.as_tensor(data.non_tensor_batch[key], device=mask.device, dtype=torch.bool)
            if skip.shape != mask.shape[:1]:
                raise ValueError(f"Invalid per-row {key} shape: {skip.shape}")
            mask &= ~skip[:, None]
    return mask


def split_ppo_batch(batch, size):
    if size is None or size <= 0:
        raise ValueError("PPO batch size must be positive")
    # DataProto.chunk(floor(n/size)) changes sizes or drops the small tail.
    # Slices preserve tensor/non-tensor linkage for multimodal inputs as well.
    return [batch[start : start + size] for start in range(0, len(batch), size)]
