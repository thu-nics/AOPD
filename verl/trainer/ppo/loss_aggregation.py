"""Additive loss contributions normalized over one optimizer minibatch.

FSDP averages gradients across its world. Ulysses replicates the post-gather
batch on SP ranks and already scales its gather backward. Count each logical
DP batch once and compensate only the DP gradient average, never SP twice.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

LOSS_NORMALIZATION_VERSION = "global-minibatch-v1"
LOSS_AGG_MODES = {"token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"}


def loss_normalization_metadata(config):
    return {
        "loss_normalization": LOSS_NORMALIZATION_VERSION,
        "loss_agg_mode": config.loss_agg_mode,
        "loss_normalizer_length": config.get("loss_normalizer_length") if config.loss_agg_mode == "seq-mean-token-sum-norm" else None,
        "policy_loss_mode": config.get("policy_loss", {}).get("loss_mode"),
    }


def validate_loss_config(config):
    mode = config.loss_agg_mode
    if mode not in LOSS_AGG_MODES:
        raise ValueError(f"Invalid loss_agg_mode: {mode}")
    version = config.get("loss_normalization", LOSS_NORMALIZATION_VERSION)
    if version != LOSS_NORMALIZATION_VERSION:
        raise ValueError(f"Unsupported loss_normalization: {version}")
    length = config.get("loss_normalizer_length")
    if mode == "seq-mean-token-sum-norm" and (isinstance(length, bool) or not isinstance(length, int) or length <= 0):
        raise ValueError("seq-mean-token-sum-norm requires a positive integer loss_normalizer_length")
    if config.get("policy_loss", {}).get("loss_mode") == "gspo" and mode != "seq-mean-token-mean":
        raise ValueError("GSPO requires loss_agg_mode=seq-mean-token-mean")


@dataclass(frozen=True)
class LossNormalization:
    token_count: float
    sequence_count: float
    gradient_scale: float = 1.0
    loss_normalizer_length: int | None = None

    @classmethod
    def from_mask(cls, mask, *, distributed=False, device=None, group=None, sequence_parallel_size=1, loss_normalizer_length=None):
        # Reduction dtype avoids allocating a full minibatch-sized float64 copy.
        lengths = mask.detach().sum(-1, dtype=torch.float64)
        counts = torch.stack((lengths.sum(), (lengths > 0).sum()))
        world_size = 1
        if distributed and dist.is_initialized():
            world_size = dist.get_world_size(group)
            if device is not None:
                counts = counts.to(device)
            dist.all_reduce(counts, group=group)
        if sequence_parallel_size < 1 or world_size % sequence_parallel_size:
            raise ValueError("sequence_parallel_size must divide the loss-reduction world size")
        tokens, sequences = (counts / sequence_parallel_size).tolist()
        return cls(tokens, sequences, world_size / sequence_parallel_size, loss_normalizer_length)

    @property
    def is_empty(self):
        return self.token_count == 0


def aggregate_loss(loss_mat, loss_mask, loss_agg_mode, *, normalization=None, loss_normalizer_length=None):
    if loss_agg_mode not in LOSS_AGG_MODES:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    if loss_mat.shape != loss_mask.shape or loss_mat.ndim != 2:
        raise ValueError("Loss and mask must have matching [rows, tokens] shapes")
    if normalization is None:
        normalization = LossNormalization.from_mask(loss_mask, loss_normalizer_length=loss_normalizer_length)
    # Accumulate bf16/fp16 inputs in fp32. Masked NaNs must not poison statistics.
    dtype = torch.float64 if loss_mat.dtype == torch.float64 else torch.float32
    mask = loss_mask.to(dtype=dtype)
    masked_loss = torch.where(mask != 0, loss_mat.to(dtype), 0.0) * mask
    sums = masked_loss.sum(-1)
    if loss_agg_mode == "seq-mean-token-mean":
        numerator = (sums / mask.sum(-1).clamp(min=1)).sum()
    else:
        numerator = sums.sum()
    denominator = normalization.token_count if loss_agg_mode == "token-mean" else normalization.sequence_count
    if loss_agg_mode == "seq-mean-token-sum-norm":
        length = normalization.loss_normalizer_length if loss_normalizer_length is None else loss_normalizer_length
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("seq-mean-token-sum-norm requires a positive integer loss_normalizer_length")
        denominator *= length
    return numerator * (normalization.gradient_scale / max(denominator, 1.0))
