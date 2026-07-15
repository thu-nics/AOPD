# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def _extract_turn_level_row_values(values: torch.Tensor, response_mask: torch.Tensor, value_token: str) -> torch.Tensor:
    """Extract one scalar critic value per generated turn row."""
    if value_token not in {"first", "last"}:
        raise ValueError(f"unknown TurnLevelPPO value_token: {value_token!r}")

    lengths = response_mask.sum(dim=1).long()
    has_tokens = lengths > 0
    if value_token == "last":
        idx = torch.clamp(lengths - 1, min=0)
    else:
        idx = torch.zeros_like(lengths)
    row_values = values[torch.arange(values.size(0), device=values.device), idx]
    return row_values * has_tokens.float()


def compute_turn_level_ppo_advantage(
    data: DataProto,
    gamma: float = 1.0,
    lam: float = 1.0,
    eps: float = 1e-8,
    normalize_adv: bool = True,
    value_token: str = "first",
    reward_source: str = "non_tensor_rewards",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute turn-level TD-GAE over environment turns instead of response tokens.

    Each DataProto row is one completed environment action. Rows sharing a
    ``traj_uid`` form one trajectory and are ordered by ``turn_index``.
    The default value_token="first" uses the pre-action state value V(s_t);
    value_token="last" is still pre-final-token under the current critic path.
    """
    if "values" not in data.batch:
        raise KeyError("TurnLevelPPO requires critic values in data.batch['values']")
    if "response_mask" not in data.batch:
        raise KeyError("TurnLevelPPO requires data.batch['response_mask']")

    response_mask = data.batch["response_mask"].float()
    n = len(data)

    if reward_source == "non_tensor_rewards":
        rewards = np.asarray(data.non_tensor_batch["rewards"], dtype=np.float32)
    elif reward_source == "token_level_rewards":
        rewards = (data.batch["token_level_rewards"] * response_mask).sum(dim=-1).detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"unknown TurnLevelPPO reward_source: {reward_source!r}")

    turn_indices = np.asarray(data.non_tensor_batch["turn_index"], dtype=np.int32)
    traj_uids = np.asarray(data.non_tensor_batch["traj_uid"], dtype=object)
    is_terminal = np.asarray(
        data.non_tensor_batch.get("is_terminal", np.zeros(n, dtype=bool)), dtype=bool
    )
    is_padding = np.asarray(
        data.non_tensor_batch.get("is_padding", np.zeros(n, dtype=bool)), dtype=bool
    )
    keep = ~is_padding

    row_values_t = _extract_turn_level_row_values(data.batch["values"], response_mask, value_token)
    row_values = row_values_t.detach().cpu().numpy().astype(np.float32)

    row_advantages_raw = np.zeros(n, dtype=np.float32)
    row_returns = np.zeros(n, dtype=np.float32)

    for uid in np.unique(traj_uids[keep]):
        idxs = np.where((traj_uids == uid) & keep)[0]
        if idxs.size == 0:
            continue
        idxs = idxs[np.argsort(turn_indices[idxs], kind="stable")]

        last_gae = 0.0
        for pos in reversed(range(len(idxs))):
            i = idxs[pos]
            has_next = pos + 1 < len(idxs)
            nonterminal = 1.0 if has_next and not is_terminal[i] else 0.0
            next_value = row_values[idxs[pos + 1]] if nonterminal else 0.0
            delta = rewards[i] + gamma * next_value * nonterminal - row_values[i]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            row_advantages_raw[i] = last_gae
            row_returns[i] = last_gae + row_values[i]

    row_advantages = row_advantages_raw.copy()
    if normalize_adv and keep.any():
        raw = row_advantages[keep]
        adv_mean = raw.mean()
        adv_std = raw.std()
        if adv_std > eps:
            row_advantages[keep] = (raw - adv_mean) / (adv_std + eps)
        else:
            row_advantages[keep] = 0.0
    else:
        adv_mean = row_advantages_raw[keep].mean() if keep.any() else 0.0
        adv_std = row_advantages_raw[keep].std() if keep.any() else 0.0

    if is_padding.any():
        skip_t = torch.tensor(is_padding, dtype=torch.bool, device=response_mask.device)
        response_mask = response_mask.clone()
        response_mask[skip_t] = 0
        data.batch["response_mask"] = response_mask

    adv_tensor = torch.tensor(row_advantages, dtype=torch.float32, device=response_mask.device)
    ret_tensor = torch.tensor(row_returns, dtype=torch.float32, device=response_mask.device)
    token_advantages = adv_tensor.unsqueeze(-1) * response_mask
    token_returns = ret_tensor.unsqueeze(-1) * response_mask

    kept_rewards = rewards[keep]
    kept_values = row_values[keep]
    kept_returns = row_returns[keep]
    data.meta_info["turn_level_ppo/reward_mean"] = float(kept_rewards.mean()) if kept_rewards.size else 0.0
    data.meta_info["turn_level_ppo/value_mean"] = float(kept_values.mean()) if kept_values.size else 0.0
    data.meta_info["turn_level_ppo/value_std"] = float(kept_values.std()) if kept_values.size else 0.0
    data.meta_info["turn_level_ppo/adv_mean_raw"] = float(adv_mean)
    data.meta_info["turn_level_ppo/adv_std_raw"] = float(adv_std)
    data.meta_info["turn_level_ppo/return_mean"] = float(kept_returns.mean()) if kept_returns.size else 0.0
    data.meta_info["turn_level_ppo/num_trajs"] = float(len(np.unique(traj_uids[keep]))) if keep.any() else 0.0
    data.meta_info["turn_level_ppo/num_turns"] = float(keep.sum())
    data.meta_info["turn_level_ppo/padding_rate"] = float(is_padding.mean()) if n else 0.0

    return token_advantages, token_returns



def compute_vineppo_advantage(
    data: DataProto,
    gamma: float = 1.0,
    normalize_adv: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute VinePPO row advantages from MC state values."""
    if "response_mask" not in data.batch:
        raise KeyError("VinePPO requires data.batch['response_mask']")
    for key in ["rewards", "vine_v_curr", "vine_v_next"]:
        if key not in data.non_tensor_batch:
            raise KeyError(f"VinePPO requires data.non_tensor_batch[{key!r}]")

    n = len(data)
    response_mask = data.batch["response_mask"].float()
    rewards = np.asarray(data.non_tensor_batch["rewards"], dtype=np.float32)
    v_curr = np.asarray(data.non_tensor_batch["vine_v_curr"], dtype=np.float32)
    v_next = np.asarray(data.non_tensor_batch["vine_v_next"], dtype=np.float32)
    is_terminal = np.asarray(data.non_tensor_batch.get("is_terminal", np.zeros(n, dtype=bool)), dtype=bool)
    is_padding = np.asarray(data.non_tensor_batch.get("is_padding", np.zeros(n, dtype=bool)), dtype=bool)
    train_mask = np.asarray(data.non_tensor_batch.get("vine_train_mask", np.ones(n, dtype=bool)), dtype=bool)

    v_next = np.where(is_terminal, 0.0, v_next)
    raw_adv = rewards + float(gamma) * v_next - v_curr
    keep = (~is_padding) & train_mask
    response_lengths = response_mask.sum(dim=-1).detach().cpu().numpy().astype(np.float64)
    token_keep = keep & (response_lengths > 0)
    row_adv = np.zeros(n, dtype=np.float32)
    token_count = 0.0
    token_raw_mean = 0.0
    token_raw_std = 0.0
    if keep.any():
        vals = raw_adv[keep]
        raw_mean = float(vals.mean())
        raw_std = float(vals.std())
        raw_abs_max = float(np.abs(vals).max())
        zero_adv_row_rate = float(np.mean(np.abs(vals) <= eps))

        if normalize_adv and token_keep.any():
            token_count = float(response_lengths[token_keep].sum())
            token_raw_mean = float(
                np.sum(raw_adv[token_keep] * response_lengths[token_keep]) / token_count
            )
            centered = raw_adv[token_keep] - token_raw_mean
            variance_denom = token_count - 1.0
            token_raw_var = (
                float(np.sum(response_lengths[token_keep] * centered * centered) / variance_denom)
                if variance_denom > 0
                else 0.0
            )
            token_raw_std = float(np.sqrt(max(token_raw_var, 0.0)))
            if token_raw_std > eps:
                row_adv[token_keep] = centered / (token_raw_std + eps)
        elif not normalize_adv:
            row_adv[keep] = vals
    else:
        raw_mean = 0.0
        raw_std = 0.0
        raw_abs_max = 0.0
        zero_adv_row_rate = 1.0
        token_count = 0.0
        token_raw_mean = 0.0
        token_raw_std = 0.0

    if not normalize_adv:
        token_count = float(response_lengths[token_keep].sum())
        if token_keep.any():
            token_raw_mean = float(
                np.sum(raw_adv[token_keep] * response_lengths[token_keep]) / token_count
            )
            centered = raw_adv[token_keep] - token_raw_mean
            variance_denom = token_count - 1.0
            token_raw_var = (
                float(np.sum(response_lengths[token_keep] * centered * centered) / variance_denom)
                if variance_denom > 0
                else 0.0
            )
            token_raw_std = float(np.sqrt(max(token_raw_var, 0.0)))
        else:
            token_raw_mean = 0.0
            token_raw_std = 0.0

    adv_tensor = torch.tensor(row_adv, dtype=torch.float32, device=response_mask.device)
    token_advantages = adv_tensor.unsqueeze(-1) * response_mask
    token_returns = token_advantages.clone()

    train_adv = row_adv[token_keep]
    effective_adv_abs_max = float(np.abs(train_adv).max()) if train_adv.size else 0.0
    all_zero_advantage = effective_adv_abs_max <= eps
    if token_keep.any():
        token_adv_mean = float(
            np.sum(row_adv[token_keep] * response_lengths[token_keep]) / token_count
        )
        centered_adv = row_adv[token_keep] - token_adv_mean
        token_adv_var = (
            float(
                np.sum(response_lengths[token_keep] * centered_adv * centered_adv)
                / (token_count - 1.0)
            )
            if token_count > 1.0
            else 0.0
        )
        token_adv_std = float(np.sqrt(max(token_adv_var, 0.0)))
    else:
        token_adv_mean = 0.0
        token_adv_std = 0.0
    data.meta_info["vineppo/v_curr_mean"] = float(v_curr[keep].mean()) if keep.any() else 0.0
    data.meta_info["vineppo/v_curr_std"] = float(v_curr[keep].std()) if keep.any() else 0.0
    data.meta_info["vineppo/v_next_mean"] = float(v_next[keep].mean()) if keep.any() else 0.0
    data.meta_info["vineppo/v_next_std"] = float(v_next[keep].std()) if keep.any() else 0.0
    data.meta_info["vineppo/raw_adv_mean"] = raw_mean
    data.meta_info["vineppo/raw_adv_std"] = raw_std
    data.meta_info["vineppo/raw_adv_abs_max"] = raw_abs_max
    data.meta_info["vineppo/zero_adv_row_rate"] = zero_adv_row_rate
    data.meta_info["vineppo/token_raw_adv_mean"] = token_raw_mean
    data.meta_info["vineppo/token_raw_adv_std"] = token_raw_std
    data.meta_info["vineppo/num_train_tokens"] = token_count
    data.meta_info["vineppo/effective_adv_abs_max"] = effective_adv_abs_max
    data.meta_info["vineppo/all_zero_advantage"] = float(all_zero_advantage)
    data.meta_info["vineppo/adv_mean"] = token_adv_mean
    data.meta_info["vineppo/adv_std"] = token_adv_std
    data.meta_info["vineppo/num_rows"] = float(keep.sum())
    data.meta_info["vineppo/train_row_rate"] = float(keep.mean()) if n else 0.0
    data.meta_info["vineppo/padding_rate"] = float(is_padding.mean()) if n else 0.0
    if "vine_num_states" in data.meta_info:
        data.meta_info["vineppo/num_states"] = float(data.meta_info["vine_num_states"])
    if "vine_num_mc_rollouts" in data.meta_info:
        data.meta_info["vineppo/num_mc_rollouts"] = float(data.meta_info["vine_num_mc_rollouts"])
    return token_advantages, token_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return step_advantages


# ------------------------------------------------------------------ #
# --------------- VPR Turn-Level Advantage Estimation -------------- #
# ------------------------------------------------------------------ #
def compute_vpr_turn_level_advantage(
    data: DataProto,
    min_group_size: int = 4,
    eps: float = 1e-8,
    outcome_reward_scale: float = 1.0,
    state_group_advantage_mode: str = "group_whiten",
) -> tuple:
    """VPR per-turn normalized advantage estimation.

    For each turn position t, normalizes VPR oracle rewards r_t across all batch
    rows at turn t using (r_t - mean_t) / (std_t + eps). Falls back to batch-wide
    normalization when fewer than min_group_size rows share the same turn index.

    If outcome_reward_scale > 0, a terminal bonus (scale * terminal_success) is
    added to the final step's effective reward before per-turn normalization, and
    stored separately in data.non_tensor_batch['vpr_outcome_bonus'] for metric
    logging. The bonus is zero for all non-terminal steps.

    state_group_advantage_mode controls state_group rows:
      - "group_whiten": current behavior, (reward - group_mean) / group_std.
      - "mean_then_batch_whiten": subtract group mean only, then whiten all
        non-skipped rows in the batch.

    Returns (advantages, returns) as token-level tensors of shape (batch, response_len).
    """
    if state_group_advantage_mode not in {"group_whiten", "mean_then_batch_whiten"}:
        raise ValueError(f"unknown state_group_advantage_mode: {state_group_advantage_mode!r}")
    vpr_oracle_rewards = np.array(data.non_tensor_batch['rewards'], dtype=np.float32)
    turn_indices = np.array(data.non_tensor_batch['turn_index'], dtype=np.int32)

    # Divisibility padding (random duplicate rows appended by adjust_batch purely to make
    # the batch divisible across DP workers) must NOT participate in VPR normalization:
    # the population at each turn must reflect only real episodes that reached that turn.
    # `keep` marks real rows; padded rows get zero advantage, zero response mask, and are
    # excluded from statistics, logged metrics, and emitted evidence.
    is_padding = np.asarray(
        data.non_tensor_batch.get('is_padding', np.zeros(len(vpr_oracle_rewards), dtype=bool)),
        dtype=bool,
    )
    keep = ~is_padding

    # Compute outcome bonus separately for logging and then add to effective reward
    outcome_bonus = np.zeros_like(vpr_oracle_rewards)
    if outcome_reward_scale != 0.0:
        is_terminal = np.array(
            data.non_tensor_batch.get('is_terminal', np.zeros(len(vpr_oracle_rewards), dtype=bool)),
            dtype=bool,
        )
        terminal_success = np.array(
            data.non_tensor_batch.get('terminal_success', np.zeros(len(vpr_oracle_rewards), dtype=bool)),
            dtype=bool,
        )
        outcome_bonus = is_terminal.astype(np.float32) * (outcome_reward_scale * terminal_success.astype(np.float32))

    # Store for separate metric logging (oracle reward vs outcome bonus)
    data.non_tensor_batch['vpr_oracle_reward'] = vpr_oracle_rewards
    data.non_tensor_batch['vpr_outcome_bonus'] = outcome_bonus

    # Effective per-step reward for normalization: VPR oracle + outcome bonus at terminal
    per_step_rewards = vpr_oracle_rewards + outcome_bonus

    n = len(per_step_rewards)
    row_advantages = np.zeros(n, dtype=np.float32)
    kept_rewards = per_step_rewards[keep]
    global_mean = kept_rewards.mean() if kept_rewards.size else 0.0
    global_std = (kept_rewards.std() + eps) if kept_rewards.size else eps

    vpr_skip_loss = is_padding.copy()

    state_group_ids = data.non_tensor_batch.get('state_group_uid', None)
    if state_group_ids is not None:
        state_group_ids = np.asarray(state_group_ids, dtype=object)
        group_stds = []
        zero_std_groups = 0
        skipped_equal_reward_groups = 0
        state_group_count = 0
        for gid in np.unique(state_group_ids[keep]):
            mask = (state_group_ids == gid) & keep
            group = per_step_rewards[mask]
            state_group_count += 1
            if len(group) >= 2:
                std = group.std()
                group_stds.append(float(std))
                if std > eps:
                    centered = group - group.mean()
                    if state_group_advantage_mode == "mean_then_batch_whiten":
                        row_advantages[mask] = centered
                    else:
                        row_advantages[mask] = centered / (std + eps)
                else:
                    zero_std_groups += 1
                    skipped_equal_reward_groups += 1
                    vpr_skip_loss[mask] = True
            else:
                zero_std_groups += 1
                skipped_equal_reward_groups += 1
                vpr_skip_loss[mask] = True

        train_mask = keep & ~vpr_skip_loss
        if state_group_advantage_mode == "mean_then_batch_whiten" and train_mask.any():
            raw_advantages = row_advantages[train_mask]
            batch_adv_mean = raw_advantages.mean()
            batch_adv_std = raw_advantages.std()
            if batch_adv_std > eps:
                row_advantages[train_mask] = (raw_advantages - batch_adv_mean) / (batch_adv_std + eps)
            else:
                row_advantages[train_mask] = 0.0
            data.meta_info['state_group_batch_adv_mean'] = float(batch_adv_mean)
            data.meta_info['state_group_batch_adv_std'] = float(batch_adv_std)
        else:
            data.meta_info['state_group_batch_adv_mean'] = 0.0
            data.meta_info['state_group_batch_adv_std'] = 0.0

        data.non_tensor_batch['vpr_skip_loss'] = vpr_skip_loss

        selected = np.asarray(
            data.non_tensor_batch.get('state_group_selected', np.zeros(n, dtype=bool)),
            dtype=bool,
        ) & keep
        data.meta_info['state_group_best_reward_mean'] = (
            float(per_step_rewards[selected].mean()) if selected.any() else 0.0
        )
        data.meta_info['state_group_reward_std_mean'] = (
            float(np.mean(group_stds)) if group_stds else 0.0
        )
        data.meta_info['state_group_zero_std_rate'] = (
            float(zero_std_groups / max(state_group_count, 1)) if keep.any() else 0.0
        )
        data.meta_info['state_group_skipped_equal_reward_rate'] = (
            float(skipped_equal_reward_groups / max(state_group_count, 1)) if keep.any() else 0.0
        )
        data.meta_info['state_group_train_sample_rate'] = (
            float(train_mask.sum() / max(keep.sum(), 1)) if keep.any() else 0.0
        )
        skipped = keep & vpr_skip_loss
        skipped_denom = max(int(skipped.sum()), 1)
        data.meta_info['state_group_skipped_sample_rate'] = (
            float(skipped.sum() / max(keep.sum(), 1)) if keep.any() else 0.0
        )
        if 'state_group_unique_action_rate' in data.non_tensor_batch:
            _uniq = np.asarray(data.non_tensor_batch['state_group_unique_action_rate'], dtype=np.float32)
            data.meta_info['state_group_unique_action_rate'] = (
                float(_uniq[selected].mean()) if selected.any() else 0.0
            )
        _move = None
        if 'move_optimal' in data.non_tensor_batch:
            _move = np.asarray(data.non_tensor_batch['move_optimal'], dtype=bool)
            data.meta_info['state_group_selected_oracle_rate'] = (
                float(_move[selected].mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_candidate_oracle_rate'] = (
                float(_move[keep].mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_skipped_oracle_rate'] = (
                float(_move[skipped].sum() / skipped_denom) if skipped.any() else 0.0
            )
        _random_selected = None
        _best_selected = None
        if 'state_group_selection_type' in data.non_tensor_batch:
            _selection_type = np.asarray(data.non_tensor_batch['state_group_selection_type'], dtype=object).astype(str)
            _random_selected = selected & (_selection_type == 'random')
            _best_selected = selected & (_selection_type != 'random')
            data.meta_info['state_group_random_selected_rate'] = (
                float(_random_selected.sum() / max(selected.sum(), 1)) if selected.any() else 0.0
            )
            data.meta_info['state_group_best_selected_rate'] = (
                float(_best_selected.sum() / max(selected.sum(), 1)) if selected.any() else 0.0
            )
            if _move is not None:
                data.meta_info['state_group_random_selected_oracle_rate'] = (
                    float(_move[_random_selected].mean()) if _random_selected.any() else 0.0
                )
                data.meta_info['state_group_best_selected_oracle_rate'] = (
                    float(_move[_best_selected].mean()) if _best_selected.any() else 0.0
                )
        if 'is_action_valid' in data.non_tensor_batch:
            _valid = np.asarray(data.non_tensor_batch['is_action_valid'], dtype=bool)
            data.meta_info['state_group_candidate_valid_action_rate'] = (
                float(_valid[keep].mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_candidate_invalid_action_rate'] = (
                float((~_valid[keep]).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_skipped_valid_action_rate'] = (
                float(_valid[skipped].sum() / skipped_denom) if skipped.any() else 0.0
            )
            data.meta_info['state_group_skipped_invalid_action_rate'] = (
                float((~_valid[skipped]).sum() / skipped_denom) if skipped.any() else 0.0
            )
            if _random_selected is not None and _best_selected is not None:
                data.meta_info['state_group_random_selected_valid_action_rate'] = (
                    float(_valid[_random_selected].mean()) if _random_selected.any() else 0.0
                )
                data.meta_info['state_group_best_selected_valid_action_rate'] = (
                    float(_valid[_best_selected].mean()) if _best_selected.any() else 0.0
                )
        if 'legal_non_oracle' in data.non_tensor_batch and 'parsed_action' in data.non_tensor_batch:
            _legal_non_oracle = np.asarray(data.non_tensor_batch['legal_non_oracle'], dtype=bool)
            _parsed = np.asarray(data.non_tensor_batch['parsed_action'], dtype=object)
            _is_reveal = np.char.startswith(_parsed.astype(str), 'reveal')
            _is_flag = np.char.startswith(_parsed.astype(str), 'flag')
            data.meta_info['state_group_selected_non_oracle_reveal_rate'] = (
                float((_legal_non_oracle[selected] & _is_reveal[selected]).mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_selected_non_oracle_flag_rate'] = (
                float((_legal_non_oracle[selected] & _is_flag[selected]).mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_candidate_non_oracle_reveal_rate'] = (
                float((_legal_non_oracle[keep] & _is_reveal[keep]).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_candidate_non_oracle_flag_rate'] = (
                float((_legal_non_oracle[keep] & _is_flag[keep]).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_skipped_non_oracle_reveal_rate'] = (
                float((_legal_non_oracle[skipped] & _is_reveal[skipped]).sum() / skipped_denom)
                if skipped.any() else 0.0
            )
            data.meta_info['state_group_skipped_non_oracle_flag_rate'] = (
                float((_legal_non_oracle[skipped] & _is_flag[skipped]).sum() / skipped_denom)
                if skipped.any() else 0.0
            )
            if _random_selected is not None and _best_selected is not None:
                data.meta_info['state_group_random_selected_non_oracle_reveal_rate'] = (
                    float((_legal_non_oracle[_random_selected] & _is_reveal[_random_selected]).mean())
                    if _random_selected.any() else 0.0
                )
                data.meta_info['state_group_random_selected_non_oracle_flag_rate'] = (
                    float((_legal_non_oracle[_random_selected] & _is_flag[_random_selected]).mean())
                    if _random_selected.any() else 0.0
                )
                data.meta_info['state_group_best_selected_non_oracle_reveal_rate'] = (
                    float((_legal_non_oracle[_best_selected] & _is_reveal[_best_selected]).mean())
                    if _best_selected.any() else 0.0
                )
                data.meta_info['state_group_best_selected_non_oracle_flag_rate'] = (
                    float((_legal_non_oracle[_best_selected] & _is_flag[_best_selected]).mean())
                    if _best_selected.any() else 0.0
                )
            if _move is not None:
                data.meta_info['state_group_candidate_oracle_reveal_rate'] = (
                    float((_move[keep] & _is_reveal[keep]).mean()) if keep.any() else 0.0
                )
                data.meta_info['state_group_candidate_oracle_flag_rate'] = (
                    float((_move[keep] & _is_flag[keep]).mean()) if keep.any() else 0.0
                )
                data.meta_info['state_group_skipped_oracle_reveal_rate'] = (
                    float((_move[skipped] & _is_reveal[skipped]).sum() / skipped_denom)
                    if skipped.any() else 0.0
                )
                data.meta_info['state_group_skipped_oracle_flag_rate'] = (
                    float((_move[skipped] & _is_flag[skipped]).sum() / skipped_denom)
                    if skipped.any() else 0.0
                )
        if 'oracle_tier' in data.non_tensor_batch and _move is not None:
            _tier = np.asarray(data.non_tensor_batch['oracle_tier'], dtype=object).astype(str)
            data.meta_info['state_group_candidate_safe_reveal_rate'] = (
                float((_move[keep] & (_tier[keep] == 'safe_reveal')).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_candidate_certain_flag_rate'] = (
                float((_move[keep] & (_tier[keep] == 'certain_flag')).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_candidate_guess_rate'] = (
                float((_move[keep] & (_tier[keep] == 'guess')).mean()) if keep.any() else 0.0
            )
            data.meta_info['state_group_selected_safe_reveal_rate'] = (
                float((_move[selected] & (_tier[selected] == 'safe_reveal')).mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_selected_certain_flag_rate'] = (
                float((_move[selected] & (_tier[selected] == 'certain_flag')).mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_selected_guess_rate'] = (
                float((_move[selected] & (_tier[selected] == 'guess')).mean()) if selected.any() else 0.0
            )
            data.meta_info['state_group_skipped_safe_reveal_rate'] = (
                float((_move[skipped] & (_tier[skipped] == 'safe_reveal')).sum() / skipped_denom)
                if skipped.any() else 0.0
            )
            data.meta_info['state_group_skipped_certain_flag_rate'] = (
                float((_move[skipped] & (_tier[skipped] == 'certain_flag')).sum() / skipped_denom)
                if skipped.any() else 0.0
            )
            data.meta_info['state_group_skipped_guess_rate'] = (
                float((_move[skipped] & (_tier[skipped] == 'guess')).sum() / skipped_denom)
                if skipped.any() else 0.0
            )
    else:
        for t in np.unique(turn_indices[keep]):
            mask = (turn_indices == t) & keep
            group = per_step_rewards[mask]
            if len(group) >= min_group_size:
                mean_t = group.mean()
                std_t = group.std() + eps
            else:
                mean_t = global_mean
                std_t = global_std
            row_advantages[mask] = (group - mean_t) / std_t
    # Padded rows and equal-reward state groups keep advantage 0 (initialized above).

    response_mask = data.batch['response_mask']
    # Zero the response mask for rows that should not contribute training tokens: DP
    # padding rows, plus state-group rows whose candidate rewards are all identical.
    if vpr_skip_loss.any():
        skip_t = torch.tensor(vpr_skip_loss, dtype=torch.bool, device=response_mask.device)
        response_mask = response_mask.clone()
        response_mask[skip_t] = 0
        data.batch['response_mask'] = response_mask
    adv_tensor = torch.tensor(row_advantages, dtype=torch.float32).to(response_mask.device)
    # Broadcast per-row advantage across all response tokens (matching GRPO convention)
    token_advantages = adv_tensor.unsqueeze(-1) * response_mask.float()

    # Emit per-batch evidence when VPR_SMOKE_EVIDENCE is set.
    # Structure: {"batches": [{batch_id, min_group_size, eps, global_mean, global_std, rows}]}
    # Each batch boundary is preserved so smoke_verify.py can recompute per-turn advantages
    # and verify exact consistency rather than accepting fabricated aggregate values.
    _evidence_path = os.environ.get("VPR_SMOKE_EVIDENCE", "")
    if _evidence_path:
        import json as _json
        # Compute prompt lengths from token masks
        attn = data.batch.get("attention_mask", None)
        resp = data.batch.get("response_mask", None)
        prompt_lens = (
            (attn.sum(dim=1) - resp.sum(dim=1)).cpu().tolist()
            if attn is not None and resp is not None
            else [None] * n
        )

        traj_uids = data.non_tensor_batch.get("traj_uid", [None] * n)
        is_terminal_arr = data.non_tensor_batch.get("is_terminal", [False] * n)
        terminal_success_arr = data.non_tensor_batch.get("terminal_success", [False] * n)

        # Read prompt sidecar written by rollout_loop.py (keyed by traj_uid+turn_index)
        _sidecar_path = _evidence_path + ".prompts.jsonl"
        _prompt_map = {}
        try:
            with open(_sidecar_path) as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if _line:
                        _entry = _json.loads(_line)
                        _key = (_entry.get("traj_uid"), _entry.get("turn_index"))
                        _prompt_map[_key] = _entry
        except FileNotFoundError:
            pass

        rows = []
        for i in range(n):
            if is_padding[i]:
                continue  # divisibility padding is not a real observation
            _uid = str(traj_uids[i]) if traj_uids[i] is not None else None
            _ti = int(turn_indices[i])
            _pm = _prompt_map.get((_uid, _ti), {})
            rows.append({
                "traj_uid": _uid,
                "turn_index": _ti,
                "oracle_reward": float(vpr_oracle_rewards[i]),
                "outcome_bonus": float(outcome_bonus[i]),
                "effective_reward": float(per_step_rewards[i]),
                "advantage": float(row_advantages[i]),
                "is_terminal": bool(is_terminal_arr[i]),
                "terminal_success": bool(terminal_success_arr[i]),
                "prompt_len": int(prompt_lens[i]) if prompt_lens[i] is not None else None,
                "prompt_prefix": _pm.get("prompt_prefix", ""),
                "action_prefix": _pm.get("action_prefix", ""),
            })

        # Read existing evidence and append this batch
        evidence = {"batches": []}
        try:
            with open(_evidence_path) as _f:
                evidence = _json.load(_f)
            if "rows" in evidence and "batches" not in evidence:
                # Migrate old flat format
                evidence = {"batches": [{"batch_id": 0, "rows": evidence["rows"],
                                          "min_group_size": min_group_size, "eps": eps,
                                          "global_mean": float(global_mean),
                                          "global_std": float(global_std)}]}
        except (FileNotFoundError, ValueError):
            pass

        evidence["batches"].append({
            "batch_id": len(evidence["batches"]),
            "min_group_size": min_group_size,
            "eps": eps,
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "rows": rows,
        })
        with open(_evidence_path, "w") as _f:
            _json.dump(evidence, _f)

    returns = token_advantages.clone()
    return token_advantages, returns

