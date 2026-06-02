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
) -> tuple:
    """VPR per-turn normalized advantage estimation.

    For each turn position t, normalizes VPR oracle rewards r_t across all batch
    rows at turn t using (r_t - mean_t) / (std_t + eps). Falls back to batch-wide
    normalization when fewer than min_group_size rows share the same turn index.

    If outcome_reward_scale > 0, a terminal bonus (scale * terminal_success) is
    added to the final step's effective reward before per-turn normalization, and
    stored separately in data.non_tensor_batch['vpr_outcome_bonus'] for metric
    logging. The bonus is zero for all non-terminal steps.

    Returns (advantages, returns) as token-level tensors of shape (batch, response_len).
    """
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
    # Padded rows keep advantage 0 (initialized above).

    response_mask = data.batch['response_mask']
    # Zero the response mask for padded rows so they contribute no advantage tokens and no
    # masked loss/metric downstream.
    if is_padding.any():
        pad_t = torch.tensor(is_padding, dtype=torch.bool, device=response_mask.device)
        response_mask = response_mask.clone()
        response_mask[pad_t] = 0
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

