"""Shared state-group filtering and advantage normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

ADVANTAGE_MODES = {"group_whiten", "mean_then_batch_whiten"}
NORMALIZATION_SCOPES = {"batch", "environment"}


def state_group_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and materialize the canonical state-group configuration."""
    values = dict(config or {})
    mode = str(values.get("advantage_mode", "group_whiten"))
    scope = str(values.get("normalization_scope", "batch"))
    eps = float(values.get("equal_reward_eps", 1e-8))
    min_candidates = int(values.get("min_candidates", 2))
    min_effective_groups = int(values.get("min_effective_groups", 1))
    compact_policy_rows = bool(values.get("compact_policy_rows", False))
    diagnostic_only = values.get("diagnostic_only", False)
    if mode not in ADVANTAGE_MODES:
        raise ValueError(f"unsupported state-group advantage mode: {mode!r}")
    if scope not in NORMALIZATION_SCOPES:
        raise ValueError(f"unsupported state-group normalization scope: {scope!r}")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("state-group equal_reward_eps must be finite and positive")
    if min_candidates < 2:
        raise ValueError("state-group min_candidates must be at least two")
    if min_effective_groups < 1:
        raise ValueError("state-group min_effective_groups must be positive")
    if not isinstance(diagnostic_only, (bool, np.bool_)):
        raise ValueError("state-group diagnostic_only must be a boolean")
    return {
        "advantage_mode": mode,
        "normalization_scope": scope,
        "equal_reward_eps": eps,
        "min_candidates": min_candidates,
        "min_effective_groups": min_effective_groups,
        "compact_policy_rows": compact_policy_rows,
        "diagnostic_only": bool(diagnostic_only),
    }


def compute_state_group_train_mask(
    data,
    *,
    filter_rewards: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return canonical trainable rows and state-group filtering metrics."""
    cfg = state_group_config(config)
    if "state_group_uid" not in data.non_tensor_batch:
        raise ValueError("state-group training requires state_group_uid")
    row_count = len(data.non_tensor_batch["state_group_uid"])
    if filter_rewards is None:
        if "rewards" not in data.non_tensor_batch:
            raise ValueError("state-group training requires raw environment rewards")
        filter_rewards = np.asarray(data.non_tensor_batch["rewards"], dtype=np.float32)
    else:
        filter_rewards = np.asarray(filter_rewards, dtype=np.float32)

    group_ids = np.asarray(data.non_tensor_batch["state_group_uid"], dtype=object)
    is_padding = np.asarray(
        data.non_tensor_batch.get("is_padding", np.zeros(row_count, dtype=bool)),
        dtype=bool,
    )
    semantic_mask = np.asarray(
        data.non_tensor_batch.get("semantic_train_mask", np.ones(row_count, dtype=bool)),
        dtype=bool,
    )
    runtime_mask = np.asarray(
        data.non_tensor_batch.get("runtime_train_mask", np.ones(row_count, dtype=bool)),
        dtype=bool,
    )
    for name, values in (
        ("state_group_uid", group_ids),
        ("filter_rewards", filter_rewards),
        ("is_padding", is_padding),
        ("semantic_train_mask", semantic_mask),
        ("runtime_train_mask", runtime_mask),
    ):
        if values.shape != (row_count,):
            raise ValueError(f"{name} must contain one value per response")

    real = ~is_padding
    eligible = real & semantic_mask & runtime_mask
    train_mask = np.zeros(row_count, dtype=bool)
    raw_groups = 0
    effective_groups = 0
    missing_supervision_groups = 0
    equal_reward_groups = 0
    for group_id in np.unique(group_ids[real]):
        raw_groups += 1
        group_mask = eligible & (group_ids == group_id)
        if int(group_mask.sum()) < cfg["min_candidates"]:
            missing_supervision_groups += 1
            continue
        if np.ptp(filter_rewards[group_mask]) <= cfg["equal_reward_eps"]:
            equal_reward_groups += 1
            continue
        train_mask[group_mask] = True
        effective_groups += 1

    metrics = {
        "state_group/raw_groups": float(raw_groups),
        "state_group/effective_groups": float(effective_groups),
        "state_group/skipped_equal_reward_rate": float(equal_reward_groups / max(raw_groups, 1)),
        "state_group/missing_supervision_group_rate": float(missing_supervision_groups / max(raw_groups, 1)),
        "state_group/supervision_row_rate": float(eligible.sum() / max(real.sum(), 1)),
        "state_group/train_row_rate": float(train_mask.sum() / max(real.sum(), 1)),
    }
    return train_mask, metrics


def compute_state_group_row_advantages(
    row_scores: np.ndarray,
    group_ids: np.ndarray,
    train_mask: np.ndarray,
    *,
    mode: str,
    eps: float,
    partitions: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one scalar advantage per response over canonical train rows."""
    scores = np.asarray(row_scores, dtype=np.float32)
    groups = np.asarray(group_ids, dtype=object)
    train = np.asarray(train_mask, dtype=bool)
    if not (scores.shape == groups.shape == train.shape):
        raise ValueError("state-group scores, IDs, and train mask must align")
    if mode not in ADVANTAGE_MODES:
        raise ValueError(f"unsupported state-group advantage mode: {mode!r}")

    advantages = np.zeros_like(scores, dtype=np.float32)
    for group_id in np.unique(groups[train]):
        mask = train & (groups == group_id)
        centered = scores[mask] - scores[mask].mean()
        if mode == "group_whiten":
            # Preserve the existing DAPO/GRPO sample-standard-deviation convention.
            std = centered.std(ddof=1) if int(mask.sum()) > 1 else 0.0
            if std > eps:
                centered = centered / (std + eps)
            else:
                centered = np.zeros_like(centered)
        advantages[mask] = centered

    if mode == "mean_then_batch_whiten":
        if partitions is None:
            partitions = np.full(len(scores), "batch", dtype=object)
        else:
            partitions = np.asarray(partitions, dtype=object)
            if partitions.shape != scores.shape:
                raise ValueError("state-group normalization partitions must align")
        for partition in np.unique(partitions[train]):
            mask = train & (partitions == partition)
            std = advantages[mask].std(ddof=0)
            if std > eps:
                advantages[mask] /= std + eps
            else:
                advantages[mask] = 0.0
    return advantages


def state_group_token_advantages(
    data,
    *,
    row_scores: np.ndarray,
    train_mask: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast canonical row advantages over generated response tokens."""
    cfg = state_group_config(config)
    partitions = None
    if cfg["normalization_scope"] == "environment":
        if "vpr_game" not in data.non_tensor_batch:
            raise ValueError("normalization_scope=environment requires vpr_game metadata")
        partitions = np.asarray(data.non_tensor_batch["vpr_game"], dtype=object)
    row_advantages = compute_state_group_row_advantages(
        row_scores,
        np.asarray(data.non_tensor_batch["state_group_uid"], dtype=object),
        train_mask,
        mode=cfg["advantage_mode"],
        eps=cfg["equal_reward_eps"],
        partitions=partitions,
    )
    response_mask = data.batch["response_mask"]
    skip = torch.as_tensor(
        ~np.asarray(train_mask, dtype=bool),
        dtype=torch.bool,
        device=response_mask.device,
    )
    if skip.any():
        response_mask = response_mask.clone()
        response_mask[skip] = 0
        data.batch["response_mask"] = response_mask
    rows = torch.as_tensor(
        row_advantages,
        dtype=torch.float32,
        device=response_mask.device,
    )
    advantages = rows.unsqueeze(-1) * response_mask.float()
    return advantages, advantages
