# VPR Training Pipeline Audit

This document records the exact tensor path from `env.step()` through advantage computation,
as required by task0 (AC-8).

## Reward Tensor Path

```
env.step(text_actions)
  → rewards: np.ndarray[float32, shape=(batch_size,)]        [envs.step() return]
  → VPRBaseEnvironmentManager.step() returns (observations, rewards, dones, infos)
```

### 1. Per-Step Reward Storage (rollout_loop.py)

**File**: `agent_system/multi_turn_rollout/rollout_loop.py`

In `vanilla_multi_turn_loop()` (line 285), each rollout step:

```python
# line 387
batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
# line 390
batch.non_tensor_batch['turn_index'] = np.full(batch_size, _step, dtype=np.int32)
# line 392-395
batch.non_tensor_batch['is_terminal'] = active_masks & _dones_np
batch.non_tensor_batch['terminal_success'] = np.array(
    [bool(info.get('terminal_success', False)) for info in infos], dtype=bool
)
```

- `rewards`: per-step VPR oracle rewards from `env.step()` (shape: batch_size)
- `turn_index`: step index within the episode (scalar, same for all active rows at step t)
- `is_terminal`: True for rows where the episode ended at this step
- `terminal_success`: True for rows where `info['terminal_success']` is True

These are stored per active row in the final collated `total_batch_list`.

### 2. Active Row Aggregation (gather_rollout_data, line 233)

After all rollout steps, `gather_rollout_data()` collects active rows across all
steps into a flat batch. Each row in the flat batch corresponds to one (episode, step)
pair. The `episode_rewards` accumulator holds per-episode total reward (used by
`EpisodeRewardManager`).

### 3. EpisodeRewardManager Collapse (episode.py)

**File**: `agent_system/reward_manager/episode.py`

`EpisodeRewardManager.__call__()` (line 20):

```python
# line 72-79 — collapse to episode level
episode_rewards = data_item.non_tensor_batch['episode_rewards']
...
score = episode_rewards  # (or /episode_lengths if normalize_by_episode_length)
reward_tensor[i, valid_response_length - 1] = score
```

- **EpisodeRewardManager collapses** per-step rewards to a single episode score
  and places it at the **last response token** of each row.
- This is the default behavior for episode-level credit assignment.
- The raw per-step rewards remain in `non_tensor_batch['rewards']`; they are NOT
  used by the default reward manager.

Output: `reward_tensor` shape `(batch_size, response_length)` placed in
`data.batch["token_level_scores"]`.

### 4. VPR Estimator Dispatch (ray_trainer.py)

**File**: `verl/trainer/ppo/ray_trainer.py`

At line 1209:
```python
batch.batch["token_level_scores"] = reward_tensor
```

At line 1227 (no KL):
```python
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

At lines 1233–1258, `compute_advantage()` is called with `adv_estimator='vpr'`:

```python
# line 361–380 (ray_trainer.py:compute_advantage)
elif adv_estimator == AdvantageEstimator.VPR:
    vpr_outcome_scale = kwargs.get('vpr_outcome_reward_scale', 1.0)
    advantages, returns = core_gigpo.compute_vpr_turn_level_advantage(
        data=data,
        min_group_size=4,
        outcome_reward_scale=vpr_outcome_scale,
    )
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
```

### 5. VPR Turn-Level Advantage Computation (core_gigpo.py)

**File**: `gigpo/core_gigpo.py`, function `compute_vpr_turn_level_advantage()` (line 390)

```python
vpr_oracle_rewards = data.non_tensor_batch['rewards']       # per-step VPR oracle rewards
turn_indices = data.non_tensor_batch['turn_index']          # step position in episode

# Optional: compute and separate outcome bonus
outcome_bonus = is_terminal * outcome_scale * terminal_success
data.non_tensor_batch['vpr_oracle_reward'] = vpr_oracle_rewards   # logged separately
data.non_tensor_batch['vpr_outcome_bonus'] = outcome_bonus        # logged separately

# Effective reward = VPR oracle + outcome bonus (combined only for normalization)
per_step_rewards = vpr_oracle_rewards + outcome_bonus

# Per-turn normalization with fallback
for t in unique(turn_indices):
    group = per_step_rewards[turn_index == t]
    if len(group) >= min_group_size:
        row_advantages[mask] = (group - group.mean()) / (group.std() + eps)
    else:
        row_advantages[mask] = (group - global_mean) / global_std

# Broadcast scalar advantage to all response tokens
token_advantages = row_advantages.unsqueeze(-1) * response_mask.float()
```

Output: `data.batch['advantages']` shape `(batch_size, response_length)`.

## Summary Table

| Step | File | Line | Key Tensor / Key |
|------|------|------|-----------------|
| env.step() returns rewards | VPRBaseEnvironmentManager | base_manager.py:38 | `rewards: np.ndarray` |
| Per-step reward stored | rollout_loop.py | 387 | `non_tensor_batch['rewards']` |
| Turn index stored | rollout_loop.py | 390 | `non_tensor_batch['turn_index']` |
| Terminal metadata stored | rollout_loop.py | 392-395 | `non_tensor_batch['is_terminal']`, `['terminal_success']` |
| EpisodeRewardManager collapses to episode | episode.py | 72-79 | `data.batch['token_level_scores']` (last token) |
| Token-level scores assigned | ray_trainer.py | 1209 | `batch.batch['token_level_scores']` |
| VPR dispatch | ray_trainer.py | 361 | `compute_vpr_turn_level_advantage()` |
| Per-turn normalization | core_gigpo.py | 390 | `data.batch['advantages']` |

## Key Findings

1. **EpisodeRewardManager** collapses multi-step rewards to episode level (places
   one score at the last response token). The per-step rewards in
   `non_tensor_batch['rewards']` are **preserved separately** and used directly
   by `compute_vpr_turn_level_advantage()`, bypassing the collapsed score.

2. **VPR advantage** uses raw per-step rewards and `turn_index` — not
   `token_level_scores` — so each step's reward is normalized independently
   within its turn-position group across the batch.

3. **Outcome bonus** is computed inside `compute_vpr_turn_level_advantage()`,
   added to the effective reward before normalization, and stored separately in
   `non_tensor_batch['vpr_outcome_bonus']` for logging. It does NOT overwrite
   `non_tensor_batch['rewards']`.
