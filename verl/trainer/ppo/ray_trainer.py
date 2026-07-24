# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    DAPO = "dapo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    VPR = 'vpr'
    TurnLevelPPO = 'turn_level_ppo'
    VinePPO = 'vineppo'


def _should_skip_vpr_state_group_update(meta_info, vpr_cfg):
    threshold = None
    if vpr_cfg is not None:
        threshold = vpr_cfg.get("skip_update_equal_reward_threshold", None)
    if threshold is None:
        return False, 0.0, 0.0

    skipped_equal_rate = float(meta_info.get('state_group_skipped_equal_reward_rate', 0.0) or 0.0)
    skipped_oracle_rate = float(meta_info.get('state_group_skipped_oracle_rate', 0.0) or 0.0)
    product = skipped_equal_rate * skipped_oracle_rate
    return skipped_equal_rate > float(threshold), skipped_equal_rate, product

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _compute_validation_diagnostics(data_sources, validation_extra_infos):
    """Aggregate response-level validation diagnostics by data source."""
    data_sources = np.asarray(data_sources)
    num_samples = len(data_sources)
    metrics = {}

    aligned_infos = {
        key: np.asarray(values)
        for key, values in validation_extra_infos.items()
        if len(values) == num_samples
    }

    for data_source in np.unique(data_sources):
        source_mask = data_sources == data_source
        prefix = f"val/{data_source}"
        source_count = int(source_mask.sum())
        metrics[f"{prefix}/num_samples"] = source_count

        if "acc" in aligned_infos:
            accuracy = aligned_infos["acc"][source_mask].astype(np.float64)
            metrics[f"{prefix}/accuracy"] = float(accuracy.mean())
            metrics[f"{prefix}/correct_count"] = int(accuracy.sum())

        if "score" in aligned_infos:
            raw_scores = aligned_infos["score"][source_mask].astype(np.float64)
            metrics[f"{prefix}/raw_score/mean"] = float(raw_scores.mean())

        if "pred" in aligned_infos:
            predictions = aligned_infos["pred"][source_mask]
            valid_answers = np.asarray(
                [str(prediction) != "[INVALID]" for prediction in predictions],
                dtype=np.float64,
            )
            valid_answer_rate = float(valid_answers.mean())
            metrics[f"{prefix}/valid_answer_rate"] = valid_answer_rate
            metrics[f"{prefix}/invalid_answer_rate"] = 1.0 - valid_answer_rate

        if "overlong" in aligned_infos:
            overlong = aligned_infos["overlong"][source_mask].astype(np.float64)
            metrics[f"{prefix}/overlong_rate"] = float(overlong.mean())

        if "overlong_reward" in aligned_infos:
            penalties = aligned_infos["overlong_reward"][source_mask].astype(
                np.float64
            )
            metrics[f"{prefix}/overlong_penalty/mean"] = float(penalties.mean())

        if "response_length" in aligned_infos:
            response_lengths = aligned_infos["response_length"][
                source_mask
            ].astype(np.float64)
            max_response_length = aligned_infos["max_response_length"][
                source_mask
            ].astype(np.float64)
            metrics[f"{prefix}/response_length/mean"] = float(
                response_lengths.mean()
            )
            metrics[f"{prefix}/response_length/p50"] = float(
                np.percentile(response_lengths, 50)
            )
            metrics[f"{prefix}/response_length/p95"] = float(
                np.percentile(response_lengths, 95)
            )
            metrics[f"{prefix}/response_length/max"] = float(
                response_lengths.max()
            )
            metrics[f"{prefix}/response_length/clip_ratio"] = float(
                np.mean(response_lengths >= max_response_length)
            )

    return metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.DAPO):
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn and "loss_mask" in data.batch:
            # If multi-turn AND a per-token loss_mask is available, use its response part
            # (it excludes interleaved observation tokens). The loss_mask is not populated
            # until just before the actor update, so at advantage time it is usually absent
            # for these single-turn-per-step VPR rollouts — fall back to response_mask.
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        group_index = data.non_tensor_batch["uid"]
        if (
            adv_estimator == AdvantageEstimator.DAPO
            and "state_group_uid" in data.non_tensor_batch
        ):
            group_index = data.non_tensor_batch["state_group_uid"]
        sample_mask = None
        if adv_estimator == AdvantageEstimator.DAPO:
            is_padding = np.asarray(
                data.non_tensor_batch.get(
                    "is_padding", np.zeros(len(data), dtype=bool)
                ),
                dtype=bool,
            )
            dapo_skip_loss = is_padding.copy()
            if "state_group_uid" in data.non_tensor_batch:
                if "rewards" not in data.non_tensor_batch:
                    raise ValueError(
                        "state-group DAPO requires raw environment rewards for dynamic filtering"
                    )
                state_group_ids = np.asarray(
                    data.non_tensor_batch["state_group_uid"], dtype=object
                )
                raw_rewards = np.asarray(
                    data.non_tensor_batch["rewards"], dtype=np.float32
                )
                keep = ~is_padding
                raw_group_count = 0
                effective_group_count = 0
                for state_group_id in np.unique(state_group_ids[keep]):
                    group_mask = keep & (state_group_ids == state_group_id)
                    raw_group_count += 1
                    if np.ptp(raw_rewards[group_mask]) <= 1e-8:
                        dapo_skip_loss[group_mask] = True
                    else:
                        effective_group_count += 1
                data.meta_info["dapo/raw_state_groups"] = float(raw_group_count)
                data.meta_info["dapo/effective_state_groups"] = float(
                    effective_group_count
                )
                data.meta_info["dapo/skipped_equal_reward_rate"] = float(
                    1.0 - effective_group_count / max(raw_group_count, 1)
                )
                data.meta_info["dapo/train_sample_rate"] = float(
                    (~dapo_skip_loss).sum() / max(keep.sum(), 1)
                )
                if "vpr_game" in data.non_tensor_batch:
                    tasks = np.asarray(
                        data.non_tensor_batch["vpr_game"], dtype=object
                    ).astype(str)
                    for task in sorted(set(tasks[keep])):
                        if not task:
                            continue
                        task_mask = keep & (tasks == task)
                        task_groups = np.unique(state_group_ids[task_mask])
                        effective_task_groups = sum(
                            bool(
                                np.any(
                                    task_mask
                                    & (state_group_ids == state_group_id)
                                    & ~dapo_skip_loss
                                )
                            )
                            for state_group_id in task_groups
                        )
                        prefix = f"dapo/{task}"
                        data.meta_info[f"{prefix}/raw_state_groups"] = float(
                            len(task_groups)
                        )
                        data.meta_info[f"{prefix}/effective_state_groups"] = float(
                            effective_task_groups
                        )
                        data.meta_info[f"{prefix}/skipped_equal_reward_rate"] = float(
                            1.0 - effective_task_groups / max(len(task_groups), 1)
                        )
                        data.meta_info[f"{prefix}/train_sample_rate"] = float(
                            (task_mask & ~dapo_skip_loss).sum()
                            / max(task_mask.sum(), 1)
                        )
            data.non_tensor_batch["dapo_skip_loss"] = dapo_skip_loss
            sample_mask = ~dapo_skip_loss
            if dapo_skip_loss.any():
                grpo_calculation_mask = grpo_calculation_mask.clone()
                grpo_calculation_mask[
                    torch.as_tensor(
                        dapo_skip_loss,
                        dtype=torch.bool,
                        device=grpo_calculation_mask.device,
                    )
                ] = 0
                data.batch["response_mask"] = grpo_calculation_mask
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=group_index,
            traj_index=data.non_tensor_batch.get(
                "traj_uid", data.non_tensor_batch["uid"]
            ),
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            compute_mean_std_cross_steps=not bool(
                kwargs.get("dapo_trajectory_level_advantage", False)
            ),
            sample_mask=sample_mask,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.VPR:
        vpr_outcome_scale = kwargs.get('vpr_outcome_reward_scale', 1.0)
        vpr_state_group_advantage_mode = kwargs.get('vpr_state_group_advantage_mode', 'group_whiten')
        advantages, returns = core_gigpo.compute_vpr_turn_level_advantage(
            data=data,
            min_group_size=4,
            outcome_reward_scale=vpr_outcome_scale,
            state_group_advantage_mode=vpr_state_group_advantage_mode,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        # Log VPR-specific metrics over REAL rows only (exclude divisibility padding).
        _vpr_pad = np.asarray(
            data.non_tensor_batch.get('is_padding', np.zeros(len(data), dtype=bool)), dtype=bool)
        _vpr_keep = ~_vpr_pad
        if 'vpr_oracle_reward' in data.non_tensor_batch:
            _vals = np.asarray(data.non_tensor_batch['vpr_oracle_reward'])[_vpr_keep]
            if _vals.size:
                data.meta_info['vpr_oracle_reward_mean'] = float(_vals.mean())
        if 'vpr_outcome_bonus' in data.non_tensor_batch:
            _vals = np.asarray(data.non_tensor_batch['vpr_outcome_bonus'])[_vpr_keep]
            if _vals.size:
                data.meta_info['vpr_outcome_bonus_mean'] = float(_vals.mean())
    elif adv_estimator == AdvantageEstimator.TurnLevelPPO:
        cfg = kwargs.get('turn_level_ppo', {})
        advantages, returns = core_gigpo.compute_turn_level_ppo_advantage(
            data=data,
            gamma=gamma,
            lam=lam,
            normalize_adv=cfg.get('normalize_adv', True),
            value_token=cfg.get('value_token', 'first'),
            reward_source=cfg.get('reward_source', 'non_tensor_rewards'),
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.VinePPO:
        cfg = kwargs.get('vineppo', {})
        advantages, returns = core_gigpo.compute_vineppo_advantage(
            data=data,
            gamma=cfg.get('gamma', gamma),
            normalize_adv=cfg.get('normalize_adv', True),
            eps=cfg.get('eps', 1e-8),
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator in [AdvantageEstimator.GAE, AdvantageEstimator.TurnLevelPPO]:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.DAPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO,
            AdvantageEstimator.VPR,
            AdvantageEstimator.VinePPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        rollout_mode = getattr(config.env.rollout, "mode", "vanilla")
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        if rollout_mode == "state_group":
            real_train_batch_size = config.data.train_batch_size
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        if config.algorithm.adv_estimator == AdvantageEstimator.VinePPO:
            if rollout_mode != "vanilla":
                raise ValueError("VinePPO MVP supports env.rollout.mode=vanilla only")
            if config.reward_model.enable:
                raise ValueError("VinePPO does not use reward_model")
            if getattr(config.env, "history_length", 0) != 0:
                raise ValueError("VinePPO requires env.history_length=0")
            entropy_coeff = float(config.actor_rollout_ref.actor.get("entropy_coeff", 0.0) or 0.0)
            if abs(entropy_coeff) > 0.0:
                raise ValueError(
                    "VinePPO requires actor_rollout_ref.actor.entropy_coeff=0; "
                    "entropy-only updates corrupt zero-advantage batches"
                )
            if config.algorithm.vineppo.get("max_states_per_batch", None) is not None:
                raise ValueError("VinePPO MVP requires algorithm.vineppo.max_states_per_batch=null")
            if int(config.algorithm.vineppo.get("state_stride", 1) or 1) != 1:
                raise ValueError("VinePPO MVP requires algorithm.vineppo.state_stride=1")

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            supported_multi_turn = [
                AdvantageEstimator.GRPO,
                AdvantageEstimator.DAPO,
                AdvantageEstimator.VPR,
                AdvantageEstimator.TurnLevelPPO,
                AdvantageEstimator.VinePPO,
            ]
            if config.algorithm.adv_estimator not in supported_multi_turn:
                assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=self._json_default) + "\n")

        print(f"Dumped generations to {filename}")

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        validation_extra_infos = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            is_agent_validation = self.traj_collector is not None
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if is_agent_validation:
                non_tensor_batch_keys_to_pop.append("data_source")
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")


            if is_agent_validation:
                test_output_gen_batch = self.traj_collector.multi_turn_loop(
                    gen_batch=test_gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.val_envs,
                    is_train=False,
                )
                del test_batch
                test_batch = test_output_gen_batch
            else:
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
                test_output_gen_batch_padded = (
                    self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                )
                test_output_gen_batch = unpad_dataproto(
                    test_output_gen_batch_padded, pad_size=pad_size
                )
                test_batch = test_batch.union(test_output_gen_batch)
            print("validation generation end")
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            prompt_ids = test_output_gen_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_ids]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if len(values) == reward_tensor.shape[0]:
                    validation_extra_infos[key].extend(list(values))

            response_mask = test_output_gen_batch.batch["attention_mask"][
                :, -output_ids.shape[-1]:
            ]
            response_lengths = response_mask.sum(dim=-1).cpu().tolist()
            validation_extra_infos["response_length"].extend(response_lengths)
            validation_extra_infos["max_response_length"].extend(
                [output_ids.shape[-1]] * len(response_lengths)
            )

            raw_keys = (
                "data_source", "traj_uid", "turn_index", "rewards", "active_masks",
                "is_terminal", "terminal_success", "episode_rewards", "episode_lengths",
                "is_action_valid", "tool_callings",
            )
            for key in raw_keys:
                values = test_output_gen_batch.non_tensor_batch.get(key)
                if values is not None:
                    validation_extra_infos[key].extend(list(values))

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            if all(
                key in test_output_gen_batch.non_tensor_batch
                for key in ("tool_callings", "traj_uid")
            ):
                tool_calling_list.append(
                    test_output_gen_batch.non_tensor_batch["tool_callings"]
                )
                traj_uid_list.append(
                    test_output_gen_batch.non_tensor_batch["traj_uid"]
                )
            if self.config.reward_model.get('reward_manager', 'episode') == 'turn':
                if not hasattr(self, '_val_episode_reward_list'):
                    self._val_episode_reward_list = []
                self._val_episode_reward_list.append(
                    test_output_gen_batch.non_tensor_batch.get('episode_rewards', np.full(len(test_output_gen_batch), np.nan))
                )
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k or k.startswith('env/'):
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        if traj_uid_list:
            tool_callings = np.concatenate(tool_calling_list, axis=0)
            traj_uids = np.concatenate(traj_uid_list, axis=0)
            _, unique_idx = np.unique(traj_uids, return_index=True)
            unique_data_sources = data_sources[unique_idx]
        else:
            tool_callings = None
            unique_idx = np.arange(len(data_sources))
            unique_data_sources = data_sources
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # For per-turn reward managers, validation test_score should remain an
        # episode-level outcome metric rather than average immediate turn reward.
        if self.config.reward_model.get('reward_manager', 'episode') == 'turn' and hasattr(self, '_val_episode_reward_list'):
            episode_rewards = np.concatenate(self._val_episode_reward_list, axis=0)
            delattr(self, '_val_episode_reward_list')
            eval_rewards = episode_rewards[unique_idx]
            eval_data_sources = unique_data_sources
        else:
            eval_rewards = reward_tensor.numpy()
            eval_data_sources = data_sources

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(len(eval_rewards)):
            data_source = eval_data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(float(eval_rewards[i]))

        # Agent rollouts expose trajectory-level tool call counts; standard DAPO does not.
        data_source_tool_calling = {}
        if tool_callings is not None:
            unique_tool_callings = tool_callings[unique_idx]
            for i in range(unique_tool_callings.shape[0]):
                data_source = unique_data_sources[i]
                if data_source not in data_source_tool_calling:
                    data_source_tool_calling[data_source] = []
                data_source_tool_calling[data_source].append(
                    unique_tool_callings[i].item()
                )

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        metric_dict.update(
            _compute_validation_diagnostics(
                data_sources=data_sources,
                validation_extra_infos=validation_extra_infos,
            )
        )

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        if validation_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=validation_extra_infos,
                dump_path=validation_data_dir,
            )
            metrics_path = os.path.join(validation_data_dir, f"{self.global_steps}.metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metric_dict, f, indent=2, sort_keys=True, default=self._json_default)
            print(f"Dumped validation metrics to {metrics_path}")

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            worker_role = "actor_rollout"
            if (
                self.config.trainer.get("val_only", False)
                and self.config.trainer.resume_mode == "disable"
            ):
                worker_role = "rollout"
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=worker_role,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if self.config.trainer.get("val_only", False):
            print("Validation-only run: skipping training dataloader state restore")
            return

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # Validation-only runs must never fall through into the training loop,
        # regardless of the regular val_before_train setting.
        val_only = self.config.trainer.get("val_only", False)
        if val_only and self.val_reward_fn is None:
            raise ValueError("trainer.val_only requires a validation reward function")
        if self.val_reward_fn is not None and (val_only or self.config.trainer.get("val_before_train", True)):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if val_only:
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        if hasattr(self.config, "env") and hasattr(self.config.env, "rollout"):
                            with open_dict(self.config.env.rollout):
                                self.config.env.rollout.current_step = int(self.global_steps)
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.VinePPO:
                        with _timer("vineppo_mc", timing_raw):
                            batch = self.traj_collector.estimate_vine_values_for_batch(
                                batch=batch,
                                actor_rollout_wg=self.actor_rollout_wg,
                                envs=self.envs,
                                vine_cfg=self.config.algorithm.vineppo,
                                generation_meta_info=gen_batch.meta_info,
                            )

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            if "rm_scores" not in batch.batch:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            vpr_outcome_reward_scale=self.config.algorithm.get('vpr', {}).get('outcome_reward_scale', 1.0),
                            vpr_state_group_advantage_mode=self.config.algorithm.get('vpr', {}).get('state_group_advantage_mode', 'group_whiten'),
                            turn_level_ppo=self.config.algorithm.get('turn_level_ppo', {}),
                            vineppo=self.config.algorithm.get('vineppo', {}),
                            dapo_trajectory_level_advantage=bool(
                                self.config.algorithm.get(
                                    "dapo_trajectory_level_advantage", False
                                )
                            ),
                        )
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.VinePPO and self.config.algorithm.vineppo.get('snapshot_fields_cleanup', True):
                            for _key in ['vine_pre_snapshot', 'vine_post_snapshot']:
                                if _key in batch.non_tensor_batch:
                                    batch.non_tensor_batch.pop(_key)

                        # Expose estimator-specific metrics.
                        for _key, _value in batch.meta_info.items():
                            if (
                                _key.startswith('turn_level_ppo/')
                                or _key.startswith('vineppo/')
                                or _key.startswith('dapo/')
                            ):
                                metrics[_key] = _value
                        skip_policy_update = False
                        if self.config.algorithm.adv_estimator == 'vpr':
                            if 'vpr_oracle_reward_mean' in batch.meta_info:
                                metrics['vpr/oracle_reward_mean'] = batch.meta_info['vpr_oracle_reward_mean']
                            if 'vpr_outcome_bonus_mean' in batch.meta_info:
                                metrics['vpr/outcome_bonus_mean'] = batch.meta_info['vpr_outcome_bonus_mean']
                            if 'state_group_random_select_prob' in batch.non_tensor_batch:
                                metrics['state_group/random_select_prob'] = float(
                                    np.mean(batch.non_tensor_batch['state_group_random_select_prob'])
                                )
                            for _key, _metric in {
                                'state_group_best_reward_mean': 'state_group/best_reward_mean',
                                'state_group_reward_std_mean': 'state_group/reward_std_mean',
                                'state_group_zero_std_rate': 'state_group/zero_std_rate',
                                'state_group_skipped_equal_reward_rate': 'state_group/skipped_equal_reward_rate',
                                'state_group_skipped_sample_rate': 'state_group/skipped_sample_rate',
                                'state_group_train_sample_rate': 'state_group/train_sample_rate',
                                'state_group_batch_adv_mean': 'state_group/batch_adv_mean',
                                'state_group_batch_adv_std': 'state_group/batch_adv_std',
                                'state_group_unique_action_rate': 'state_group/unique_action_rate',
                                'state_group_selected_oracle_rate': 'state_group/selected_oracle_rate',
                                'state_group_selected_safe_reveal_rate': 'state_group/selected_safe_reveal_rate',
                                'state_group_selected_certain_flag_rate': 'state_group/selected_certain_flag_rate',
                                'state_group_selected_guess_rate': 'state_group/selected_guess_rate',
                                'state_group_selected_non_oracle_reveal_rate': 'state_group/selected_non_oracle_reveal_rate',
                                'state_group_selected_non_oracle_flag_rate': 'state_group/selected_non_oracle_flag_rate',
                                'state_group_random_selected_rate': 'state_group/random_selected_rate',
                                'state_group_best_selected_rate': 'state_group/best_selected_rate',
                                'state_group_random_selected_oracle_rate': 'state_group/random_selected_oracle_rate',
                                'state_group_best_selected_oracle_rate': 'state_group/best_selected_oracle_rate',
                                'state_group_random_selected_valid_action_rate': 'state_group/random_selected_valid_action_rate',
                                'state_group_best_selected_valid_action_rate': 'state_group/best_selected_valid_action_rate',
                                'state_group_random_selected_non_oracle_reveal_rate': 'state_group/random_selected_non_oracle_reveal_rate',
                                'state_group_random_selected_non_oracle_flag_rate': 'state_group/random_selected_non_oracle_flag_rate',
                                'state_group_best_selected_non_oracle_reveal_rate': 'state_group/best_selected_non_oracle_reveal_rate',
                                'state_group_best_selected_non_oracle_flag_rate': 'state_group/best_selected_non_oracle_flag_rate',
                                'state_group_candidate_valid_action_rate': 'state_group/candidate_valid_action_rate',
                                'state_group_candidate_invalid_action_rate': 'state_group/candidate_invalid_action_rate',
                                'state_group_candidate_oracle_rate': 'state_group/candidate_oracle_rate',
                                'state_group_candidate_oracle_reveal_rate': 'state_group/candidate_oracle_reveal_rate',
                                'state_group_candidate_oracle_flag_rate': 'state_group/candidate_oracle_flag_rate',
                                'state_group_candidate_safe_reveal_rate': 'state_group/candidate_safe_reveal_rate',
                                'state_group_candidate_certain_flag_rate': 'state_group/candidate_certain_flag_rate',
                                'state_group_candidate_guess_rate': 'state_group/candidate_guess_rate',
                                'state_group_candidate_non_oracle_reveal_rate': 'state_group/candidate_non_oracle_reveal_rate',
                                'state_group_candidate_non_oracle_flag_rate': 'state_group/candidate_non_oracle_flag_rate',
                                'state_group_skipped_valid_action_rate': 'state_group/skipped_valid_action_rate',
                                'state_group_skipped_invalid_action_rate': 'state_group/skipped_invalid_action_rate',
                                'state_group_skipped_oracle_rate': 'state_group/skipped_oracle_rate',
                                'state_group_skipped_oracle_reveal_rate': 'state_group/skipped_oracle_reveal_rate',
                                'state_group_skipped_oracle_flag_rate': 'state_group/skipped_oracle_flag_rate',
                                'state_group_skipped_safe_reveal_rate': 'state_group/skipped_safe_reveal_rate',
                                'state_group_skipped_certain_flag_rate': 'state_group/skipped_certain_flag_rate',
                                'state_group_skipped_guess_rate': 'state_group/skipped_guess_rate',
                                'state_group_skipped_non_oracle_reveal_rate': 'state_group/skipped_non_oracle_reveal_rate',
                                'state_group_skipped_non_oracle_flag_rate': 'state_group/skipped_non_oracle_flag_rate',
                            }.items():
                                if _key in batch.meta_info:
                                    metrics[_metric] = batch.meta_info[_key]

                            skip_policy_update, skip_update_equal_rate, skip_update_product = _should_skip_vpr_state_group_update(
                                batch.meta_info,
                                self.config.algorithm.get('vpr', {}),
                            )
                            metrics['state_group/skip_update_equal_reward_rate'] = skip_update_equal_rate
                            metrics['state_group/skip_update_product'] = skip_update_product
                            metrics['training/skipped_update'] = float(skip_policy_update)
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.VinePPO:
                            skip_policy_update = bool(
                                float(batch.meta_info.get('vineppo/all_zero_advantage', 0.0) or 0.0)
                            )
                            metrics['training/skipped_update'] = float(skip_policy_update)
                        else:
                            metrics['training/skipped_update'] = 0.0

                    # update critic
                    if self.use_critic and not skip_policy_update:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if (not skip_policy_update) and self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # Ensure loss_mask is present when multi_turn is True (vllm rollout
                            # does not produce loss_mask; fall back to attention_mask)
                            if batch.meta_info["multi_turn"] and "loss_mask" not in batch.batch:
                                batch.batch["loss_mask"] = batch.batch["attention_mask"]
                            # VPR: exclude rows that should not contribute gradients. This includes
                            # divisibility-padding duplicates and, for state-group rollout, candidate groups
                            # whose rewards are all identical. Their advantages and response masks are already
                            # zeroed; loss_mask is handled separately because multi-turn actor loss falls back
                            # to attention_mask when loss_mask is absent from rollout.
                            if self.config.algorithm.adv_estimator in {'dapo', 'vpr', 'turn_level_ppo', 'vineppo'} and "loss_mask" in batch.batch:
                                _skip_loss = None
                                if self.config.algorithm.adv_estimator == 'dapo' and "dapo_skip_loss" in batch.non_tensor_batch:
                                    _skip_loss = np.asarray(batch.non_tensor_batch["dapo_skip_loss"], dtype=bool)
                                elif self.config.algorithm.adv_estimator == 'vpr' and "vpr_skip_loss" in batch.non_tensor_batch:
                                    _skip_loss = np.asarray(batch.non_tensor_batch["vpr_skip_loss"], dtype=bool)
                                elif self.config.algorithm.adv_estimator == 'vineppo' and "vineppo_skip_loss" in batch.non_tensor_batch:
                                    _skip_loss = np.asarray(batch.non_tensor_batch["vineppo_skip_loss"], dtype=bool)
                                elif "is_padding" in batch.non_tensor_batch:
                                    _skip_loss = np.asarray(batch.non_tensor_batch["is_padding"], dtype=bool)
                                if _skip_loss is not None:
                                    _skip = torch.tensor(
                                        _skip_loss, dtype=torch.bool, device=batch.batch["loss_mask"].device)
                                    if _skip.any():
                                        batch.batch["loss_mask"] = batch.batch["loss_mask"].clone()
                                        batch.batch["loss_mask"][_skip] = 0
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
