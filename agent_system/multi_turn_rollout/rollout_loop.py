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

from collections.abc import Mapping
import os
import json as _json_rl
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _resolve_train_rollout_limits(config, infos):
    """Return per-trajectory collection limits for mixed state-group training."""
    global_limit = int(config.env.max_steps)
    limits = np.full(len(infos), global_limit, dtype=np.int32)
    env_name = str(getattr(config.env, "env_name", "")).lower()
    if env_name == "tau_vpr":
        train_limit = int(config.env.tau.train_max_steps)
        if train_limit <= 0 or train_limit > global_limit:
            raise ValueError("env.tau.train_max_steps must be in [1, env.max_steps]")
        return np.full(len(infos), train_limit, dtype=np.int32)
    if env_name != "dapo_vpr_mixed":
        return limits

    for index, info in enumerate(infos):
        task = str(info.get("vpr_game") or "")
        task_config = getattr(config.env, task, None)
        if task_config is None:
            continue
        raw_limit = getattr(task_config, "train_rollout_max_steps", None)
        if raw_limit is None:
            continue
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, np.integer)):
            raise ValueError(
                f"env.{task}.train_rollout_max_steps must be a positive integer or null"
            )
        task_limit = int(raw_limit)
        environment_limit = int(getattr(task_config, "max_steps", global_limit))
        if task_limit <= 0:
            raise ValueError(
                f"env.{task}.train_rollout_max_steps must be positive"
            )
        if task_limit > environment_limit:
            raise ValueError(
                f"env.{task}.train_rollout_max_steps={task_limit} exceeds "
                f"env.{task}.max_steps={environment_limit}"
            )
        if task_limit > global_limit:
            raise ValueError(
                f"env.{task}.train_rollout_max_steps={task_limit} exceeds "
                f"env.max_steps={global_limit}"
            )
        limits[index] = task_limit
    return limits


def _positive_group_size(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_state_group_sizes(config, infos):
    """Resolve the candidate count for every base trajectory."""
    rollout_config = config.env.rollout
    default_size = _positive_group_size(rollout_config.n, "env.rollout.n")
    if str(getattr(config.env, "env_name", "")).lower() != "dapo_vpr_mixed":
        return np.full(len(infos), default_size, dtype=np.int32)

    math_size = _positive_group_size(
        getattr(rollout_config, "math_n", default_size),
        "env.rollout.math_n",
    )
    game_size = _positive_group_size(
        getattr(rollout_config, "game_n", default_size),
        "env.rollout.game_n",
    )
    supported_tasks = {"math", "sokoban", "sudoku", "minesweeper"}
    sizes = []
    for info in infos:
        task = str(info.get("vpr_game") or "")
        if task not in supported_tasks:
            raise ValueError(
                f"mixed state-group rollout received unknown task {task!r}"
            )
        sizes.append(math_size if task == "math" else game_size)
    return np.asarray(sizes, dtype=np.int32)


def _build_state_group_layout(active_indices, group_sizes):
    """Build contiguous variable-size candidate blocks for active states."""
    active_indices = np.asarray(active_indices, dtype=np.int64)
    group_sizes = np.asarray(group_sizes, dtype=np.int32)
    active_group_sizes = group_sizes[active_indices]
    if np.any(active_group_sizes <= 0):
        raise ValueError("state-group sizes must be positive")

    repeated_base_indices = np.repeat(active_indices, active_group_sizes)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(active_group_sizes)]
    )
    group_uids = np.asarray(
        [str(uuid.uuid4()) for _ in active_indices], dtype=object
    )
    state_group_uids = np.repeat(group_uids, active_group_sizes)
    candidate_ranks = np.concatenate(
        [np.arange(size, dtype=np.int32) for size in active_group_sizes]
    )
    return (
        active_group_sizes,
        repeated_base_indices,
        group_offsets,
        state_group_uids,
        candidate_ranks,
    )


def _normalize_tool_schemas(tools):
    if tools is None:
        return None
    if isinstance(tools, np.ndarray):
        tools = tools.tolist()
    if isinstance(tools, Mapping):
        tools = [tools]
    normalized = []
    for tool in tools:
        if hasattr(tool, "openai_schema"):
            tool = tool.openai_schema
        if isinstance(tool, np.ndarray):
            tool = tool.tolist()
        if not isinstance(tool, Mapping):
            raise TypeError(
                "agentic tools must be a sequence of mappings; "
                f"got element type {type(tool).__name__}"
            )
        normalized.append(dict(tool))
    return normalized


def _render_agentic_prompt(
    tokenizer,
    chat,
    prompt_rendering,
    chat_template_kwargs,
    tools=None,
):
    if prompt_rendering == "raw":
        return chat[0]["content"]
    if prompt_rendering == "chatml":
        template_kwargs = dict(chat_template_kwargs)
        normalized_tools = _normalize_tool_schemas(tools)
        if normalized_tools is not None:
            template_kwargs["tools"] = normalized_tools
        return tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
    raise ValueError(f"Unsupported agentic prompt rendering: {prompt_rendering!r}")


def _render_tau_prompt_with_budget(
    tokenizer,
    chat,
    chat_template_kwargs,
    *,
    tools,
    max_prompt_tokens,
):
    """Preserve Tau policy, task, and the latest complete interaction chunk."""

    def render(messages):
        return _render_agentic_prompt(
            tokenizer,
            messages,
            "chatml",
            chat_template_kwargs,
            tools=tools,
        )

    def token_length(prompt):
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    prompt = render(chat)
    if token_length(prompt) <= max_prompt_tokens:
        return prompt
    if not chat or chat[0].get("role") != "system":
        raise ValueError("Tau structured chat must start with a system message")

    first_user_index = next(
        (index for index, message in enumerate(chat) if message.get("role") == "user"),
        None,
    )
    if first_user_index is None:
        raise ValueError("Tau structured chat must contain the initial user request")
    pinned = [chat[0], chat[first_user_index]]
    tail = chat[first_user_index + 1 :]
    chunks = []
    current = []
    for message in tail:
        if message.get("role") == "assistant" and current:
            chunks.append(current)
            current = []
        current.append(message)
    if current:
        chunks.append(current)

    # Never remove the newest chunk: it contains the current tool result or user reply.
    while len(chunks) > 1:
        candidate = [*pinned, *(item for chunk in chunks for item in chunk)]
        prompt = render(candidate)
        if token_length(prompt) <= max_prompt_tokens:
            return prompt
        chunks.pop(0)

    candidate = [*pinned, *(item for chunk in chunks for item in chunk)]
    prompt = render(candidate)
    if token_length(prompt) <= max_prompt_tokens:
        return prompt
    raise ValueError(
        "Tau policy, initial request, tool schemas, and latest interaction do not fit "
        f"within data.max_prompt_length={max_prompt_tokens}; increase MAX_PROMPT or "
        "reduce the environment response size"
    )


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        apply_chat_template_kwargs_override: dict | None = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        if apply_chat_template_kwargs_override is None:
            apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        else:
            apply_chat_template_kwargs = apply_chat_template_kwargs_override
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_chats = obs.get('chat', None)
        obs_tools = obs.get('tools', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        obs_chat = obs_chats[item] if obs_chats is not None else None
        sample_tools = obs_tools[item] if obs_tools is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        if obs_chat is None:
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
        else:
            chat = np.asarray(obs_chat, dtype=object)
        
        prompt_rendering = self.config.env.agentic_eval.get("prompt_rendering", "chatml")
        chat_list = chat.tolist()
        env_name = str(getattr(self.config.env, "env_name", "")).lower()
        if env_name in {"tau_vpr", "tau_outcome"}:
            if prompt_rendering != "chatml":
                raise ValueError("Tau environments require ChatML prompt rendering")
            prompt_with_chat_template = _render_tau_prompt_with_budget(
                self.tokenizer,
                chat_list,
                apply_chat_template_kwargs,
                tools=sample_tools,
                max_prompt_tokens=int(self.config.data.max_prompt_length),
            )
        else:
            prompt_with_chat_template = _render_agentic_prompt(
                self.tokenizer,
                chat_list,
                prompt_rendering,
                apply_chat_template_kwargs,
                tools=sample_tools,
            )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict,
        apply_chat_template_kwargs_override: dict | None = None,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                apply_chat_template_kwargs_override=apply_chat_template_kwargs_override,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            vine_active_indices = np.where(active_masks)[0]
            vine_pre_snapshots_by_env = {}
            if self.config.algorithm.adv_estimator == "vineppo":
                pre_snapshots = envs.snapshot_states(active_indices=vine_active_indices)
                vine_pre_snapshots_by_env = {
                    int(env_idx): snapshot for env_idx, snapshot in zip(vine_active_indices, pre_snapshots)
                }

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            # Capture prompt text + action text for smoke evidence (only when VPR_SMOKE_EVIDENCE is set)
            _ev_path = os.environ.get("VPR_SMOKE_EVIDENCE", "")
            if _ev_path and obs.get("text"):
                _sidecar = _ev_path + ".prompts.jsonl"
                with open(_sidecar, "a") as _sf:
                    for _ei, (_pt, _at) in enumerate(zip(obs["text"], text_actions)):
                        # Capture the FULL prompt and action text (no truncation): the
                        # smoke verifier's prompt-locality check must be able to detect a
                        # prior action leaking anywhere into a later prompt, including past
                        # any fixed prefix window.
                        _sf.write(_json_rl.dumps({
                            "traj_uid": str(traj_uid[_ei]),
                            "turn_index": int(_step),
                            "prompt_prefix": (_pt or ""),
                            "action_prefix": (_at or ""),
                        }) + "\n")

            next_obs, rewards, dones, infos = envs.step(text_actions)

            vine_post_snapshots_by_env = {}
            if self.config.algorithm.adv_estimator == "vineppo":
                _dones_tmp = dones.squeeze(1) if hasattr(dones, "shape") and len(dones.shape) == 2 else dones
                post_indices = [int(i) for i in vine_active_indices if not bool(_dones_tmp[int(i)])]
                if post_indices:
                    post_snapshots = envs.snapshot_states(active_indices=post_indices)
                    vine_post_snapshots_by_env = {
                        int(env_idx): snapshot for env_idx, snapshot in zip(post_indices, post_snapshots)
                    }

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            for _info, _done in zip(infos, dones):
                _info["env_done"] = bool(_done)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            # VPR: track turn position and terminal state for per-turn advantage estimation
            batch.non_tensor_batch['turn_index'] = np.full(batch_size, _step, dtype=np.int32)
            _dones_np = torch_to_numpy(dones).astype(bool) if not isinstance(dones, np.ndarray) else dones.astype(bool)
            batch.non_tensor_batch['is_terminal'] = active_masks & _dones_np
            batch.non_tensor_batch['terminal_success'] = np.array(
                [bool(info.get('terminal_success', False)) for info in infos], dtype=bool
            )
            if self.config.algorithm.adv_estimator == "vineppo":
                pre_arr = np.empty(batch_size, dtype=object)
                post_arr = np.empty(batch_size, dtype=object)
                has_post = np.zeros(batch_size, dtype=bool)
                state_uid = np.empty(batch_size, dtype=object)
                next_state_uid = np.empty(batch_size, dtype=object)
                for i in range(batch_size):
                    pre_arr[i] = vine_pre_snapshots_by_env.get(i)
                    post_arr[i] = vine_post_snapshots_by_env.get(i)
                    has_post[i] = post_arr[i] is not None
                    state_uid[i] = f"{traj_uid[i]}:s:{_step}"
                    next_state_uid[i] = f"{traj_uid[i]}:s:{_step + 1}" if has_post[i] else ""
                batch.non_tensor_batch['vine_pre_snapshot'] = pre_arr
                batch.non_tensor_batch['vine_post_snapshot'] = post_arr
                batch.non_tensor_batch['vine_has_post_snapshot'] = has_post
                batch.non_tensor_batch['vine_state_uid'] = state_uid
                batch.non_tensor_batch['vine_next_state_uid'] = next_state_uid
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                if active_masks[i]:
                    total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def _state_group_multi_turn_loop_once(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """State-level group rollout for VPR environments.

        Each active environment state is expanded into its configured candidate
        count. All candidates train the policy, while one candidate is committed
        to the environment.
        """
        if not hasattr(envs, "state_group_step"):
            raise ValueError("state_group rollout requires an environment manager with state_group_step")

        batch_size = len(gen_batch.batch)

        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        length_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert batch_size == length_obs, f"gen_batch size {batch_size} does not match obs size {length_obs}"
        train_rollout_limits = _resolve_train_rollout_limits(self.config, infos)

        uid_batch = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        group_sizes = _resolve_state_group_sizes(self.config, infos)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        total_batch_list = [[] for _ in range(batch_size)]
        selected_total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        def _select_obs(source_obs, indices):
            selected = {}
            for key, value in source_obs.items():
                if value is None:
                    selected[key] = None
                elif isinstance(value, list):
                    selected[key] = [value[i] for i in indices]
                else:
                    arr = np.asarray(value, dtype=object)
                    selected[key] = arr[indices]
            return selected

        for _step in range(self.config.env.max_steps):
            active_indices = np.where((~is_done) & (_step < train_rollout_limits))[0]
            if len(active_indices) == 0:
                break

            (
                active_group_sizes,
                repeated_base_indices,
                group_offsets,
                state_group_uids,
                candidate_ranks,
            ) = _build_state_group_layout(active_indices, group_sizes)
            active_gen_batch = gen_batch.select_idxs(repeated_base_indices)
            active_obs = _select_obs(obs, repeated_base_indices)
            batch = self.preprocess_batch(gen_batch=active_gen_batch, obs=active_obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            flat_count = int(group_offsets[-1])
            batch.non_tensor_batch['uid'] = uid_batch[repeated_base_indices]
            batch.non_tensor_batch['traj_uid'] = traj_uid[repeated_base_indices]
            batch = batch.union(batch_output)


            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            candidate_action_groups = [
                text_actions[group_offsets[i]:group_offsets[i + 1]]
                for i in range(len(active_indices))
            ]
            observed_group_sizes = np.asarray(
                [len(group) for group in candidate_action_groups], dtype=np.int32
            )
            if not np.array_equal(observed_group_sizes, active_group_sizes):
                raise ValueError("candidate action groups do not match state-group layout")
            unique_action_rates = np.asarray([
                len(set(group)) / float(len(group)) if group else 0.0
                for group in candidate_action_groups
            ], dtype=np.float32)
            candidate_results, selected_indices, next_obs_active, selected_rewards, selected_dones, selected_infos = \
                envs.state_group_step(
                    candidate_action_groups,
                    active_indices=active_indices,
                )

            flat_rewards = []
            flat_dones = []
            flat_terminal_success = []
            flat_valid = []
            flat_selected = []
            flat_infos = []
            flat_vpr_game = []
            flat_move_optimal = []
            flat_legal_non_oracle = []
            flat_parsed_action = []
            flat_terminal_reason = []
            flat_oracle_tier = []
            flat_selection_type = []
            flat_random_selected = []
            flat_random_select_prob = []
            selection_types_by_group = [
                str(info.get('state_group_selection_type') or 'best')
                for info in selected_infos
            ]
            for group_pos, group in enumerate(candidate_results):
                for cand_idx, (_, reward, done, info) in enumerate(group):
                    flat_rewards.append(float(reward))
                    flat_dones.append(bool(done))
                    flat_terminal_success.append(bool(info.get('terminal_success', False)))
                    flat_valid.append(int(info.get('is_action_valid', 1)))
                    flat_selected.append(cand_idx == int(selected_indices[group_pos]))
                    flat_infos.append(info)
                    flat_vpr_game.append(str(info.get('vpr_game') or ''))
                    flat_move_optimal.append(bool(info.get('move_optimal', False)))
                    flat_legal_non_oracle.append(bool(info.get('legal_non_oracle', False)))
                    flat_parsed_action.append(str(info.get('parsed_action') or ''))
                    flat_terminal_reason.append(str(info.get('terminal_reason') or ''))
                    selection_type = selection_types_by_group[group_pos]
                    flat_selection_type.append(selection_type)
                    flat_random_selected.append(selection_type == 'random')
                    flat_random_select_prob.append(float(info.get('state_group_random_select_prob', 0.0) or 0.0))
                    flat_oracle_tier.append(str(info.get('oracle_tier') or info.get('sudoku_oracle_tier') or info.get('oracle_policy_tier') or ''))

            flat_rewards_np = np.asarray(flat_rewards, dtype=np.float32)
            flat_dones_np = np.asarray(flat_dones, dtype=bool)
            flat_selected_np = np.asarray(flat_selected, dtype=bool)

            for _info, _done in zip(selected_infos, selected_dones):
                _info["env_done"] = bool(_done)

            if 'tool_calling' in selected_infos[0]:
                tool_callings[active_indices] += np.asarray(
                    [info['tool_calling'] for info in selected_infos], dtype=np.float32)
            episode_rewards[active_indices] += torch_to_numpy(selected_rewards)
            episode_lengths[active_indices] += 1

            batch.non_tensor_batch['is_action_valid'] = np.asarray(flat_valid, dtype=bool)
            batch.non_tensor_batch['vpr_game'] = np.asarray(flat_vpr_game, dtype=object)
            batch.non_tensor_batch['rewards'] = flat_rewards_np
            batch.non_tensor_batch['active_masks'] = np.ones(flat_count, dtype=bool)
            batch.non_tensor_batch['turn_index'] = np.full(flat_count, _step, dtype=np.int32)
            batch.non_tensor_batch['is_terminal'] = flat_dones_np
            batch.non_tensor_batch['terminal_success'] = np.asarray(flat_terminal_success, dtype=bool)
            batch.non_tensor_batch['state_group_uid'] = state_group_uids
            batch.non_tensor_batch['state_group_selected'] = flat_selected_np
            batch.non_tensor_batch['state_group_rank'] = candidate_ranks
            batch.non_tensor_batch['state_group_base_index'] = repeated_base_indices.astype(np.int32)
            batch.non_tensor_batch['state_group_unique_action_rate'] = np.repeat(
                unique_action_rates, active_group_sizes
            )
            batch.non_tensor_batch['state_group_selection_type'] = np.asarray(flat_selection_type, dtype=object)
            batch.non_tensor_batch['state_group_random_selected'] = np.asarray(flat_random_selected, dtype=bool)
            batch.non_tensor_batch['state_group_random_select_prob'] = np.asarray(flat_random_select_prob, dtype=np.float32)
            batch.non_tensor_batch['move_optimal'] = np.asarray(flat_move_optimal, dtype=bool)
            batch.non_tensor_batch['legal_non_oracle'] = np.asarray(flat_legal_non_oracle, dtype=bool)
            batch.non_tensor_batch['parsed_action'] = np.asarray(flat_parsed_action, dtype=object)
            batch.non_tensor_batch['terminal_reason'] = np.asarray(flat_terminal_reason, dtype=object)
            batch.non_tensor_batch['oracle_tier'] = np.asarray(flat_oracle_tier, dtype=object)

            batch_list = to_list_of_dict(batch)
            for flat_idx, base_idx in enumerate(repeated_base_indices):
                total_batch_list[int(base_idx)].append(batch_list[flat_idx])
            for base_idx, info in zip(active_indices, selected_infos):
                selected_total_infos[int(base_idx)].append(info)

            is_done[active_indices] = np.logical_or(is_done[active_indices], selected_dones)
            for key, active_values in next_obs_active.items():
                if active_values is None:
                    obs[key] = None
                    continue
                current_values = obs.get(key)
                if current_values is None:
                    current_values = [None] * batch_size
                else:
                    current_values = list(current_values)
                for base_idx, next_value in zip(active_indices, active_values):
                    current_values[int(base_idx)] = next_value
                obs[key] = current_values

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=selected_total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def state_group_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ):
        is_mixed_dapo = (
            str(self.config.env.env_name).lower() == "dapo_vpr_mixed"
            and bool(self.config.algorithm.filter_groups.enable)
        )
        if not is_mixed_dapo:
            return self._state_group_multi_turn_loop_once(gen_batch, actor_rollout_wg, envs)

        from copy import deepcopy

        target_math = int(self.config.env.mixed.trajectory_counts.math)
        max_attempts = int(self.config.algorithm.filter_groups.max_num_gen_batches)
        if max_attempts <= 0:
            raise ValueError("mixed DAPO requires a positive max_num_gen_batches")

        retained_batches = []
        retained_rewards = []
        retained_lengths = []
        retained_traj_uids = []
        retained_tool_callings = []
        retained_success = {}
        retained_task_counts = {}
        retained_math = 0
        math_source_indices = None
        math_refill_envs = None

        def _generation_batch_for_attempt(attempt_index, source_indices=None):
            source = (
                gen_batch
                if source_indices is None
                else gen_batch.select_idxs(source_indices)
            )
            attempt_batch = deepcopy(source)
            env_kwargs = attempt_batch.non_tensor_batch.get("env_kwargs")
            if env_kwargs is None:
                raise ValueError(
                    "mixed DAPO requires env_kwargs for dynamic sampling"
                )
            updated_kwargs = []
            for item in env_kwargs:
                updated = dict(item)
                if str(updated.get("task")) == "math":
                    updated["dynamic_attempt"] = attempt_index
                updated_kwargs.append(updated)
            attempt_batch.non_tensor_batch["env_kwargs"] = np.asarray(
                updated_kwargs, dtype=object
            )
            return attempt_batch

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                attempt_batch = _generation_batch_for_attempt(0)
                attempt_envs = envs
            else:
                if math_source_indices is None or len(math_source_indices) != target_math:
                    raise ValueError(
                        "mixed DAPO could not identify the configured math slots"
                    )
                if math_refill_envs is None:
                    from agent_system.environments.env_package.math_reasoning import (
                        MathReasoningEnvironmentManager,
                        build_math_reasoning_envs,
                    )

                    math_refill_envs = MathReasoningEnvironmentManager(
                        build_math_reasoning_envs(env_num=target_math, group_n=1),
                        self.config,
                    )
                attempt_batch = _generation_batch_for_attempt(
                    attempt - 1, math_source_indices
                )
                attempt_envs = math_refill_envs
            result = self._state_group_multi_turn_loop_once(
                attempt_batch, actor_rollout_wg, attempt_envs
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uids, tool_callings = result
            trajectory_tasks = [
                str(trajectory[0].get("vpr_game") or "") if trajectory else ""
                for trajectory in batch_list
            ]
            if attempt == 1:
                math_source_indices = np.asarray(
                    [index for index, task in enumerate(trajectory_tasks) if task == "math"],
                    dtype=np.int64,
                )

            selected = []
            for index, task in enumerate(trajectory_tasks):
                if task != "math":
                    if attempt == 1:
                        selected.append(index)
                    continue
                rewards = [float(row["rewards"]) for row in batch_list[index]]
                if rewards and np.ptp(np.asarray(rewards, dtype=np.float32)) > 1e-8:
                    if retained_math < target_math:
                        selected.append(index)
                        retained_math += 1

            if selected:
                retained_batches.extend(batch_list[index] for index in selected)
                retained_rewards.append(episode_rewards[selected])
                retained_lengths.append(episode_lengths[selected])
                retained_traj_uids.append(traj_uids[selected])
                retained_tool_callings.append(tool_callings[selected])

                selected_set = set(selected)
                for task in {trajectory_tasks[index] for index in selected}:
                    retained_task_counts[task] = retained_task_counts.get(task, 0) + sum(
                        trajectory_tasks[index] == task for index in selected
                    )
                for key, values in success.items():
                    values = np.asarray(values)
                    selected_values = None
                    if len(values) == len(batch_list):
                        selected_values = values[selected]
                    else:
                        parts = key.split("/")
                        if len(parts) >= 3 and parts[0] == "env":
                            task = parts[1]
                            task_indices = [
                                index
                                for index, name in enumerate(trajectory_tasks)
                                if name == task
                            ]
                            if key.endswith("/trajectory_count"):
                                continue
                            if len(values) == len(task_indices):
                                local_selected = [
                                    local_index
                                    for local_index, global_index in enumerate(task_indices)
                                    if global_index in selected_set
                                ]
                                selected_values = values[local_selected]
                    if selected_values is not None and len(selected_values):
                        retained_success.setdefault(key, []).append(selected_values)

            print(
                f"mixed DAPO effective math groups: {retained_math}/{target_math} "
                f"after generation batch {attempt}/{max_attempts}"
            )
            if retained_math >= target_math:
                break

        if retained_math < target_math:
            raise ValueError(
                f"Only collected {retained_math}/{target_math} effective math groups "
                f"after {max_attempts} generation batches"
            )
        if not retained_batches:
            raise ValueError("mixed DAPO produced an empty training batch")

        combined_success = {
            key: np.concatenate(values) for key, values in retained_success.items()
        }
        for task, count in retained_task_counts.items():
            combined_success[f"env/{task}/trajectory_count"] = np.asarray(
                [count], dtype=np.float32
            )

        return (
            retained_batches,
            np.concatenate(retained_rewards),
            np.concatenate(retained_lengths),
            combined_success,
            np.concatenate(retained_traj_uids),
            np.concatenate(retained_tool_callings),
        )

    def mixed_outcome_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ):
        """Collect fixed game outcome groups and dynamically refill Math groups."""
        from copy import deepcopy

        group_size = int(self.config.env.rollout.n)
        if group_size <= 0:
            raise ValueError("mixed outcome DAPO requires env.rollout.n > 0")
        target_counts = {
            task: int(getattr(self.config.env.mixed.trajectory_counts, task))
            for task in ("math", "sokoban", "sudoku", "minesweeper")
        }
        target_math = target_counts["math"]
        max_attempts = int(self.config.algorithm.filter_groups.max_num_gen_batches)
        if target_math <= 0 or max_attempts <= 0:
            raise ValueError(
                "mixed outcome DAPO requires positive math groups and max_num_gen_batches"
            )

        env_kwargs = gen_batch.non_tensor_batch.get("env_kwargs")
        if env_kwargs is None:
            raise ValueError("mixed outcome DAPO requires env_kwargs")
        base_tasks = [str(item.get("task") or "") for item in env_kwargs]
        observed_counts = {task: base_tasks.count(task) for task in target_counts}
        if observed_counts != target_counts:
            raise ValueError(
                f"mixed outcome batch tasks {observed_counts} do not match configured "
                f"counts {target_counts}"
            )
        math_source_indices = np.asarray(
            [index for index, task in enumerate(base_tasks) if task == "math"],
            dtype=np.int64,
        )

        retained_batches = []
        retained_rewards = []
        retained_lengths = []
        retained_traj_uids = []
        retained_tool_callings = []
        retained_success = {}
        retained_task_counts = {}
        retained_math = 0
        math_refill_envs = None

        def _generation_batch_for_attempt(attempt_index, source_indices=None):
            source = (
                gen_batch
                if source_indices is None
                else gen_batch.select_idxs(source_indices)
            )
            attempt_base = deepcopy(source)
            source_kwargs = attempt_base.non_tensor_batch.get("env_kwargs")
            if source_kwargs is None:
                raise ValueError("mixed outcome DAPO requires env_kwargs")
            updated_kwargs = []
            for item in source_kwargs:
                updated = dict(item)
                if str(updated.get("task")) == "math":
                    updated["dynamic_attempt"] = attempt_index
                updated_kwargs.append(updated)
            attempt_base.non_tensor_batch["env_kwargs"] = np.asarray(
                updated_kwargs, dtype=object
            )
            attempt_tasks = [str(item.get("task") or "") for item in updated_kwargs]
            return (
                attempt_base.repeat(repeat_times=group_size, interleave=True),
                np.repeat(np.asarray(attempt_tasks, dtype=object), group_size),
            )

        def _retain_success_metrics(success, trajectory_tasks, selected):
            selected_set = set(selected)
            for key, raw_values in success.items():
                if key.endswith("/trajectory_count"):
                    continue
                values = np.asarray(raw_values)
                selected_values = None
                if len(values) == len(trajectory_tasks):
                    selected_values = values[selected]
                else:
                    parts = key.split("/")
                    if len(parts) >= 3 and parts[0] == "env":
                        task = parts[1]
                        task_indices = [
                            index
                            for index, name in enumerate(trajectory_tasks)
                            if name == task
                        ]
                        if len(values) == len(task_indices):
                            local_selected = [
                                local_index
                                for local_index, global_index in enumerate(task_indices)
                                if global_index in selected_set
                            ]
                            selected_values = values[local_selected]
                if selected_values is None or not len(selected_values):
                    continue
                retained_success.setdefault(key, []).append(selected_values)
                if (
                    all(trajectory_tasks[index] == "math" for index in selected)
                    and key.startswith("env/")
                    and key.count("/") == 1
                ):
                    suffix = key.removeprefix("env/")
                    retained_success.setdefault(f"env/math/{suffix}", []).append(
                        selected_values
                    )

        try:
            for attempt in range(1, max_attempts + 1):
                if attempt == 1:
                    attempt_batch, trajectory_tasks = _generation_batch_for_attempt(0)
                    attempt_envs = envs
                else:
                    if math_refill_envs is None:
                        from agent_system.environments.env_package.math_reasoning import (
                            MathReasoningEnvironmentManager,
                            build_math_reasoning_envs,
                        )

                        math_refill_envs = MathReasoningEnvironmentManager(
                            build_math_reasoning_envs(
                                env_num=target_math, group_n=group_size
                            ),
                            self.config,
                        )
                    attempt_batch, trajectory_tasks = _generation_batch_for_attempt(
                        attempt - 1, math_source_indices
                    )
                    attempt_envs = math_refill_envs

                result = self.vanilla_multi_turn_loop(
                    attempt_batch, actor_rollout_wg, attempt_envs
                )
                (
                    batch_list,
                    episode_rewards,
                    episode_lengths,
                    success,
                    traj_uids,
                    tool_callings,
                ) = result
                if len(batch_list) != len(trajectory_tasks):
                    raise ValueError("mixed outcome task ordering does not match rollouts")
                if len(batch_list) % group_size:
                    raise ValueError("mixed outcome rollouts do not form complete groups")

                selected = []
                for group_start in range(0, len(batch_list), group_size):
                    group_indices = list(
                        range(group_start, group_start + group_size)
                    )
                    group_tasks = {
                        str(trajectory_tasks[index]) for index in group_indices
                    }
                    if len(group_tasks) != 1:
                        raise ValueError("mixed outcome group crosses task boundaries")
                    task = group_tasks.pop()
                    if task != "math":
                        if attempt == 1:
                            selected.extend(group_indices)
                        continue
                    group_rewards = np.asarray(
                        episode_rewards[group_indices], dtype=np.float32
                    )
                    if np.ptp(group_rewards) > 1e-8 and retained_math < target_math:
                        selected.extend(group_indices)
                        retained_math += 1

                if selected:
                    retained_batches.extend(batch_list[index] for index in selected)
                    retained_rewards.append(np.asarray(episode_rewards)[selected])
                    retained_lengths.append(np.asarray(episode_lengths)[selected])
                    retained_traj_uids.append(np.asarray(traj_uids)[selected])
                    retained_tool_callings.append(np.asarray(tool_callings)[selected])
                    for index in selected:
                        task = str(trajectory_tasks[index])
                        retained_task_counts[task] = retained_task_counts.get(task, 0) + 1
                    _retain_success_metrics(success, trajectory_tasks, selected)

                print(
                    f"mixed outcome DAPO effective math groups: "
                    f"{retained_math}/{target_math} after generation batch "
                    f"{attempt}/{max_attempts}"
                )
                if retained_math >= target_math:
                    break
        finally:
            if math_refill_envs is not None:
                math_refill_envs.close()

        if retained_math < target_math:
            raise ValueError(
                f"Only collected {retained_math}/{target_math} effective math groups "
                f"after {max_attempts} generation batches"
            )
        expected_trajectories = sum(target_counts.values()) * group_size
        if len(retained_batches) != expected_trajectories:
            raise ValueError(
                f"mixed outcome DAPO retained {len(retained_batches)} trajectories; "
                f"expected {expected_trajectories}"
            )

        combined_success = {
            key: np.concatenate(values) for key, values in retained_success.items()
        }
        for task, count in retained_task_counts.items():
            combined_success[f"env/{task}/trajectory_count"] = np.asarray(
                [count], dtype=np.float32
            )
        return (
            retained_batches,
            np.concatenate(retained_rewards),
            np.concatenate(retained_lengths),
            combined_success,
            np.concatenate(retained_traj_uids),
            np.concatenate(retained_tool_callings),
        )

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            remaining = (
                self.config.data.train_batch_size * self.config.env.rollout.n
                - len(total_batch_list)
            )
            if len(batch_list) > remaining:
                if remaining % self.config.env.rollout.n != 0:
                    raise ValueError(
                        "dynamic rollout target must contain complete rollout groups"
                    )
                original_size = len(batch_list)
                keep = np.arange(remaining, dtype=np.int64)
                batch_list = batch_list[:remaining]
                episode_rewards = episode_rewards[keep]
                episode_lengths = episode_lengths[keep]
                traj_uid = traj_uid[keep]
                tool_callings = tool_callings[keep]
                success = {
                    key: value[keep] if len(value) == original_size else value
                    for key, value in success.items()
                }

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def _select_obs(self, obs: Dict, indices: np.ndarray) -> Dict:
        selected = {}
        for key, value in obs.items():
            if value is None:
                selected[key] = None
            else:
                selected[key] = [value[int(i)] for i in indices]
        return selected

    def _make_vine_seed_batch(self, n: int, data_sources: np.ndarray, meta_info: dict | None = None) -> DataProto:
        tensors = {
            "input_ids": torch.zeros((n, 1), dtype=torch.long),
            "attention_mask": torch.ones((n, 1), dtype=torch.long),
            "position_ids": torch.zeros((n, 1), dtype=torch.long),
        }
        prompts = np.asarray([[{"role": "user", "content": ""}] for _ in range(n)], dtype=object)
        batch = DataProto.from_dict(
            tensors=tensors,
            non_tensors={
                "raw_prompt": prompts,
                "data_source": np.asarray(data_sources, dtype=object),
            },
        )
        batch.meta_info = dict(meta_info or {})
        return batch

    def _generate_one_step_actions(
            self,
            gen_batch: DataProto,
            obs: Dict,
            actor_rollout_wg,
            apply_chat_template_kwargs: dict | None = None,
            ) -> list[str]:
        batch = self.preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            apply_chat_template_kwargs_override=apply_chat_template_kwargs,
        )
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for key in ["multi_modal_data", "raw_prompt", "tools_kwargs"]:
            if key in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        return self.tokenizer.batch_decode(batch_output.batch['responses'], skip_special_tokens=True)

    def estimate_vine_values_for_batch(
            self,
            batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            vine_cfg,
            generation_meta_info: dict | None = None,
            ) -> DataProto:
        """Estimate MC state values for VinePPO without adding MC rows to PPO data.

        The first MVP implementation rolled out each state independently. That
        was correct but extremely inefficient under hybrid FSDP+vLLM because
        every tiny generate call enters and exits the rollout sharding manager,
        repeatedly waking/sleeping vLLM. This implementation expands unique
        states into MC samples, then advances each frontier in batches bounded by
        the number of available environment workers.
        """
        if 'vine_pre_snapshot' not in batch.non_tensor_batch:
            raise KeyError("VinePPO rollout batch missing vine_pre_snapshot")
        k = int(vine_cfg.get('num_rollouts_per_state', 1))
        max_steps = int(getattr(self.config.env, 'max_steps', 1))
        max_states = vine_cfg.get('max_states_per_batch', None)
        stride = int(vine_cfg.get('state_stride', 1) or 1)
        if k <= 0:
            raise ValueError("VinePPO requires algorithm.vineppo.num_rollouts_per_state > 0")
        if max_states is not None:
            raise ValueError("VinePPO MVP requires algorithm.vineppo.max_states_per_batch=null")
        if stride != 1:
            raise ValueError("VinePPO MVP requires algorithm.vineppo.state_stride=1")
        drop_padding = bool(vine_cfg.get('drop_padding', True))
        gamma = float(vine_cfg.get('gamma', 1.0))
        mc_apply_chat_template_kwargs = None
        mc_enable_thinking = vine_cfg.get('mc_enable_thinking', None)
        if mc_enable_thinking is not None:
            base_kwargs = {}
            if hasattr(self.config, 'data'):
                base_kwargs = dict(self.config.data.get('apply_chat_template_kwargs', {}) or {})
            if isinstance(mc_enable_thinking, str):
                mc_enable_thinking = mc_enable_thinking.strip().lower() in {'1', 'true', 'yes', 'y'}
            mc_apply_chat_template_kwargs = dict(base_kwargs)
            mc_apply_chat_template_kwargs['enable_thinking'] = bool(mc_enable_thinking)
        is_padding = np.asarray(batch.non_tensor_batch.get('is_padding', np.zeros(len(batch), dtype=bool)), dtype=bool)
        keep = ~is_padding if drop_padding else np.ones(len(batch), dtype=bool)

        pre_snapshots = np.asarray(batch.non_tensor_batch['vine_pre_snapshot'], dtype=object)
        post_snapshots = np.asarray(batch.non_tensor_batch['vine_post_snapshot'], dtype=object)
        has_post = np.asarray(batch.non_tensor_batch.get('vine_has_post_snapshot', np.zeros(len(batch), dtype=bool)), dtype=bool)
        state_uids = np.asarray(batch.non_tensor_batch['vine_state_uid'], dtype=object)
        next_state_uids = np.asarray(batch.non_tensor_batch['vine_next_state_uid'], dtype=object)
        traj_uids = np.asarray(batch.non_tensor_batch.get('traj_uid', np.asarray([''] * len(batch), dtype=object)), dtype=object)
        data_sources = np.asarray(batch.non_tensor_batch.get('data_source', np.asarray(['unknown'] * len(batch), dtype=object)), dtype=object)

        candidate_trajs = np.asarray([uid for uid in np.unique(traj_uids[keep]) if str(uid)], dtype=object)
        max_train_trajs = vine_cfg.get('max_train_trajectories', None)
        selected_trajs = candidate_trajs
        if len(candidate_trajs) == 0:
            train_mask = keep.copy()
        else:
            if max_train_trajs is not None:
                max_train_trajs = int(max_train_trajs)
                if max_train_trajs <= 0:
                    raise ValueError("algorithm.vineppo.max_train_trajectories must be positive or null")
                if len(candidate_trajs) > max_train_trajs:
                    selected_trajs = np.random.choice(candidate_trajs, size=max_train_trajs, replace=False)
            train_mask = keep & np.isin(traj_uids, selected_trajs)
        batch.non_tensor_batch['vine_train_mask'] = train_mask.astype(bool)
        batch.non_tensor_batch['vineppo_skip_loss'] = (~train_mask).astype(bool)

        states: dict[str, tuple[object, int]] = {}
        for i in np.where(train_mask)[0]:
            if pre_snapshots[i] is not None and state_uids[i]:
                states.setdefault(str(state_uids[i]), (pre_snapshots[i], int(i)))
            if has_post[i] and post_snapshots[i] is not None and next_state_uids[i]:
                states.setdefault(str(next_state_uids[i]), (post_snapshots[i], int(i)))

        values: dict[str, float] = {}
        mc_returns_by_state: dict[str, list[float]] = {uid: [] for uid in states}
        mc_first_actions_by_state: dict[str, list[str]] = {uid: [] for uid in states}
        mc_generate_calls = 0
        mc_generated_batch_sizes: list[int] = []
        mc_chunks = 0

        if states:
            main_snapshots = envs.snapshot_states()
            worker_capacity = len(main_snapshots)
            if worker_capacity <= 0:
                raise ValueError("VinePPO MC rollout requires at least one environment worker")
            mc_entries: list[tuple[str, object, int]] = []
            for uid, (snapshot, source_idx) in states.items():
                for _ in range(k):
                    mc_entries.append((uid, snapshot, source_idx))

            try:
                for chunk_start in range(0, len(mc_entries), worker_capacity):
                    chunk = mc_entries[chunk_start:chunk_start + worker_capacity]
                    mc_chunks += 1
                    active_uids = np.asarray([entry[0] for entry in chunk], dtype=object)
                    active_snapshots = [entry[1] for entry in chunk]
                    active_source_indices = np.asarray([entry[2] for entry in chunk], dtype=np.int64)
                    active_returns = np.zeros(len(chunk), dtype=np.float32)
                    active_discounts = np.ones(len(chunk), dtype=np.float32)

                    for _step in range(max_steps):
                        active_count = len(active_snapshots)
                        if active_count == 0:
                            break

                        obs, _ = envs.restore_states(active_snapshots)
                        source = self._make_vine_seed_batch(
                            active_count,
                            data_sources[active_source_indices],
                            meta_info=generation_meta_info,
                        )
                        actions = self._generate_one_step_actions(
                            source,
                            obs,
                            actor_rollout_wg,
                            apply_chat_template_kwargs=mc_apply_chat_template_kwargs,
                        )
                        if _step == 0:
                            for uid, action in zip(active_uids, actions):
                                mc_first_actions_by_state[str(uid)].append(str(action))
                        mc_generate_calls += 1
                        mc_generated_batch_sizes.append(active_count)

                        _obs, rewards, dones, _infos = envs.step(actions)
                        rewards = np.asarray(rewards).reshape(-1)[:active_count].astype(np.float32)
                        dones = np.asarray(dones).reshape(-1)[:active_count].astype(bool)
                        active_returns += active_discounts * rewards

                        done_indices = np.where(dones)[0]
                        for done_idx in done_indices:
                            mc_returns_by_state[str(active_uids[done_idx])].append(float(active_returns[done_idx]))

                        live_indices = np.where(~dones)[0]
                        if len(live_indices) == 0:
                            active_snapshots = []
                            break

                        active_discounts[live_indices] *= gamma
                        live_snapshots = envs.snapshot_states(active_indices=live_indices)
                        active_snapshots = live_snapshots
                        active_uids = active_uids[live_indices]
                        active_source_indices = active_source_indices[live_indices]
                        active_returns = active_returns[live_indices]
                        active_discounts = active_discounts[live_indices]

                    if len(active_snapshots) > 0:
                        for uid, value in zip(active_uids, active_returns):
                            mc_returns_by_state[str(uid)].append(float(value))
            finally:
                envs.restore_states(main_snapshots)

            values = {
                uid: (float(np.mean(returns)) if returns else 0.0)
                for uid, returns in mc_returns_by_state.items()
            }

        v_curr = np.zeros(len(batch), dtype=np.float32)
        v_next = np.zeros(len(batch), dtype=np.float32)
        for i in range(len(batch)):
            if not is_padding[i]:
                v_curr[i] = float(values.get(str(state_uids[i]), 0.0))
                if has_post[i]:
                    v_next[i] = float(values.get(str(next_state_uids[i]), 0.0))
        batch.non_tensor_batch['vine_v_curr'] = v_curr
        batch.non_tensor_batch['vine_v_next'] = v_next
        selected_row_count = int(train_mask.sum())
        candidate_row_count = int(keep.sum())
        batch.meta_info['vine_num_states'] = float(len(states))
        batch.meta_info['vine_num_mc_rollouts'] = float(len(states) * k)
        batch.meta_info['vineppo/selected_traj_count'] = float(len(selected_trajs))
        batch.meta_info['vineppo/candidate_traj_count'] = float(len(candidate_trajs))
        batch.meta_info['vineppo/selected_traj_rate'] = float(len(selected_trajs) / len(candidate_trajs)) if len(candidate_trajs) else 0.0
        batch.meta_info['vineppo/selected_row_count'] = float(selected_row_count)
        batch.meta_info['vineppo/selected_row_rate'] = float(selected_row_count / candidate_row_count) if candidate_row_count else 0.0
        batch.meta_info['vineppo/estimated_state_rate'] = float(len(states) / max(candidate_row_count, 1))
        batch.meta_info['vineppo/mc_chunks'] = float(mc_chunks)
        batch.meta_info['vineppo/mc_generate_calls'] = float(mc_generate_calls)
        batch.meta_info['vineppo/mc_mean_batch_size'] = float(np.mean(mc_generated_batch_sizes)) if mc_generated_batch_sizes else 0.0
        batch.meta_info['vineppo/mc_max_batch_size'] = float(np.max(mc_generated_batch_sizes)) if mc_generated_batch_sizes else 0.0

        all_mc_returns = np.asarray(
            [value for returns in mc_returns_by_state.values() for value in returns],
            dtype=np.float32,
        )
        if all_mc_returns.size:
            batch.meta_info['vineppo/mc_return_mean'] = float(all_mc_returns.mean())
            batch.meta_info['vineppo/mc_return_std'] = float(all_mc_returns.std())
            batch.meta_info['vineppo/mc_positive_return_rate'] = float(np.mean(all_mc_returns > 0))
            batch.meta_info['vineppo/mc_negative_return_rate'] = float(np.mean(all_mc_returns < 0))
            batch.meta_info['vineppo/mc_zero_return_rate'] = float(np.mean(np.abs(all_mc_returns) <= 1e-8))
        else:
            batch.meta_info['vineppo/mc_return_mean'] = 0.0
            batch.meta_info['vineppo/mc_return_std'] = 0.0
            batch.meta_info['vineppo/mc_positive_return_rate'] = 0.0
            batch.meta_info['vineppo/mc_negative_return_rate'] = 0.0
            batch.meta_info['vineppo/mc_zero_return_rate'] = 0.0

        state_return_is_constant = [
            float(np.ptp(np.asarray(returns, dtype=np.float32)) <= 1e-8)
            for returns in mc_returns_by_state.values()
            if returns
        ]
        first_action_unique_rates = [
            len(set(actions)) / len(actions)
            for actions in mc_first_actions_by_state.values()
            if actions
        ]
        batch.meta_info['vineppo/mc_constant_return_state_rate'] = (
            float(np.mean(state_return_is_constant)) if state_return_is_constant else 0.0
        )
        batch.meta_info['vineppo/mc_first_action_unique_rate'] = (
            float(np.mean(first_action_unique_rates)) if first_action_unique_rates else 0.0
        )
        return batch

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        rollout_mode = getattr(self.config.env.rollout, "mode", "vanilla")
        env_name = str(getattr(self.config.env, "env_name", "")).lower()
        if is_train and rollout_mode == "state_group":
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.state_group_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        elif (
            is_train
            and env_name == "dapo_games_non_vpr_mixed"
            and bool(self.config.algorithm.filter_groups.enable)
        ):
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.mixed_outcome_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        else:
            if is_train:
                gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
            # Initial observations from the environment
            if self.config.algorithm.filter_groups.enable and is_train:
                # Dynamic Sampling (for DAPO and Dynamic GiGPO)
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            else:
                # Vanilla Sampling   
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
