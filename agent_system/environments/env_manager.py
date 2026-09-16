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
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import *
from agent_system.memory import SearchMemory, SimpleMemory


def _trainer_validation_enabled(config) -> bool:
    """Whether this run can invoke validation at any point."""
    trainer = config.trainer
    return bool(
        trainer.get("val_only", False)
        or trainer.get("val_before_train", True)
        or int(trainer.get("test_freq", -1)) > 0
    )


def select_agentic_prompt_template(
    config,
    action_tag_template,
    boxed_template,
    legacy_template,
):
    if not config.env.agentic_eval.native_action_protocol:
        return legacy_template

    action_format = config.env.agentic_eval.get("action_format", "action_tag")
    if action_format == "action_tag":
        return action_tag_template
    if action_format == "boxed":
        return boxed_template
    raise ValueError(f"Unsupported agentic action format: {action_format!r}")


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                template = select_agentic_prompt_template(
                    self.config,
                    ALFWORLD_NATIVE_ACTION_TEMPLATE_NO_HIS,
                    ALFWORLD_NATIVE_BOXED_TEMPLATE_NO_HIS,
                    ALFWORLD_TEMPLATE_NO_HIS,
                )
                obs = template.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                template = select_agentic_prompt_template(
                    self.config,
                    ALFWORLD_NATIVE_ACTION_TEMPLATE,
                    ALFWORLD_NATIVE_BOXED_TEMPLATE,
                    ALFWORLD_TEMPLATE,
                )
                obs = template.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        self.available_actions = [
            self.format_avail_actions(info['available_actions']) for info in infos
        ]
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.available_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.available_actions = [
            self.format_avail_actions(info['available_actions']) for info in infos
        ]

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:  # noqa: E722 - Preserve this unrelated legacy adapter's behavior.
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                template = select_agentic_prompt_template(
                    self.config,
                    WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS,
                    WEBSHOP_NATIVE_BOXED_TEMPLATE_NO_HIS,
                    WEBSHOP_TEMPLATE_NO_HIS,
                )
                obs = template.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                template = select_agentic_prompt_template(
                    self.config,
                    WEBSHOP_NATIVE_ACTION_TEMPLATE,
                    WEBSHOP_NATIVE_BOXED_TEMPLATE,
                    WEBSHOP_TEMPLATE,
                )
                obs = template.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    fallback_template = select_agentic_prompt_template(
                        self.config,
                        WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS,
                        WEBSHOP_NATIVE_BOXED_TEMPLATE_NO_HIS,
                        WEBSHOP_TEMPLATE_NO_HIS,
                    )
                    obs = fallback_template.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def _validate_awm_context_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    model_length = int(config.actor_rollout_ref.rollout.max_model_len)
    lengths = {
        "data.max_prompt_length": prompt_length,
        "data.max_response_length": response_length,
        "actor_rollout_ref.rollout.max_model_len": model_length,
    }
    non_positive = [name for name, value in lengths.items() if value <= 0]
    if non_positive:
        raise ValueError(
            "AWM context lengths must be positive: " + ", ".join(non_positive)
        )
    if prompt_length + response_length > model_length:
        raise ValueError(
            "AWM context budget requires data.max_prompt_length + "
            "data.max_response_length <= "
            "actor_rollout_ref.rollout.max_model_len"
        )


def _validate_teacher_reward(config):
    from agent_system.environments.teacher_reward import (
        validate_teacher_reward_config,
    )

    reward = config.env.teacher_reward
    validate_teacher_reward_config(
        str(reward.mode),
        float(reward.frequency_bonus_scale),
    )


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    rollout_mode = getattr(config.env.rollout, "mode", "vanilla")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if rollout_mode == "state_group":
        group_n = 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    mixed_env_name = config.env.env_name.lower()
    if mixed_env_name == "awm_envscaler_agentic_opd":
        if bool(
            getattr(config.env.awm.oracle, "use_privileged_context", False)
        ):
            raise ValueError(
                "AWM does not support privileged teacher context; set "
                "env.awm.oracle.use_privileged_context=false"
            )
        if rollout_mode != "state_group":
            raise ValueError(
                "awm_envscaler_agentic_opd requires env.rollout.mode=state_group"
            )
        if str(config.algorithm.adv_estimator) != "dapo":
            raise ValueError(
                "awm_envscaler_agentic_opd requires algorithm.adv_estimator=dapo"
            )
        if int(config.env.rollout.n) != 4:
            raise ValueError(
                "mixed agentic OPD protocol requires four candidates per state"
            )
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError(
                "mixed agentic OPD protocol requires rollout.n=1 at inference"
            )
        if not bool(config.actor_rollout_ref.rollout.multi_turn.enable):
            raise ValueError("mixed agentic OPD training requires multi-turn rollout")
        _validate_awm_context_budget(config)
        context = config.env.context
        if str(context.history_policy) != "token_budget":
            raise ValueError(
                "mixed agentic OPD training requires token-budget context"
            )
        if (
            context.max_history_exchanges is not None
            and int(context.max_history_exchanges) < 0
        ):
            raise ValueError("max_history_exchanges must be non-negative")
        counts = OmegaConf.to_container(
            config.env.agentic_mix.trajectory_counts, resolve=True
        )
        if sum(int(value) for value in counts.values()) != int(
            config.data.train_batch_size
        ):
            raise ValueError(
                "mixed AWM/EnvScaler counts must sum to train batch size"
            )
        if set(counts) != {"awm", "envscaler"}:
            raise ValueError(
                "mixed agentic OPD counts must contain awm and envscaler"
            )
        if str(config.env.awm.verifier_mode) != "sql":
            raise ValueError("mixed AWM training requires SQL+LLM verification")
        if str(config.env.awm.reward_mode) != "semantic":
            raise ValueError("mixed AWM training requires semantic rewards")
        _validate_teacher_reward(config)
        if int(config.env.awm.train_max_steps) != 20:
            raise ValueError("mixed AWM protocol requires 20 decisions")
        if int(config.env.envscaler.train_max_steps) != 40:
            raise ValueError("EnvScaler conversation protocol requires 40 decisions")
        terminal_judge = config.env.awm.terminal_judge
        runtime_failures = config.env.awm.runtime_failures
        if (
            not bool(terminal_judge.enabled)
            or not bool(runtime_failures.enabled)
            or not bool(runtime_failures.judge.enabled)
        ):
            raise ValueError(
                "mixed agentic OPD training requires AWM terminal/runtime judges"
            )
        envscaler_runtime = config.env.envscaler.runtime_failures
        envscaler_judge = envscaler_runtime.judge
        if not bool(envscaler_runtime.enabled) or not bool(
            envscaler_judge.enabled
        ):
            raise ValueError(
                "mixed agentic OPD training requires the EnvScaler tool-exception judge"
            )
        envscaler_confidence = int(envscaler_judge.confidence_threshold)
        if not 0 <= envscaler_confidence <= 100:
            raise ValueError(
                "EnvScaler runtime judge confidence threshold must be in [0, 100]"
            )

        from agent_system.environments.env_package.awm.runtime.oracle import (
            DeepSeekAWMOracleActor,
        )
        from agent_system.environments.env_package.envscaler.envs import (
            build_mixed_agentic_envs,
        )
        from agent_system.environments.env_package.envscaler.manager import (
            MixedAgenticEnvironmentManager,
            awm_projection,
        )
        from agent_system.environments.env_package.envscaler.source import (
            validate_envscaler_source,
        )

        validate_envscaler_source(config.env.envscaler.source_root)
        val_only = bool(config.trainer.get("val_only", False))
        oracle_actor = None
        if not val_only:
            oracle = config.env.awm.oracle
            runtime_judge = runtime_failures.judge
            oracle_actor = DeepSeekAWMOracleActor.remote(
                model=str(oracle.model),
                provider=str(oracle.provider),
                api_base=str(oracle.api_base),
                api_key_env=str(oracle.api_key_env),
                samples=int(oracle.samples),
                enable_thinking=bool(oracle.enable_thinking),
                reasoning_effort=(str(oracle.reasoning_effort) if oracle.reasoning_effort is not None else None),
                thinking_budget=(int(oracle.thinking_budget) if oracle.thinking_budget is not None else None),
                temperature=(float(oracle.temperature) if oracle.temperature is not None else None),
                top_p=(float(oracle.top_p) if oracle.top_p is not None else None),
                presence_penalty=(float(oracle.presence_penalty) if oracle.presence_penalty is not None else None),
                max_tokens=int(oracle.max_tokens),
                matcher_provider=str(oracle.matcher_provider),
                matcher_model=str(oracle.matcher_model),
                matcher_api_base=str(oracle.matcher_api_base),
                matcher_api_key_env=str(oracle.matcher_api_key_env),
                matcher_enable_thinking=oracle.get("matcher_enable_thinking"),
                matcher_reasoning_effort=oracle.get("matcher_reasoning_effort"),
                matcher_max_tokens=oracle.get("matcher_max_tokens"),
                matcher_max_concurrent_requests=oracle.get("matcher_max_concurrent_requests", 32),
                cache_path=str(oracle.cache_path),
                teacher_cache_import_paths=list(oracle.get("teacher_cache_import_paths", [])),
                teacher_cache_import_prompt_hashes=list(oracle.get("teacher_cache_import_prompt_hashes", [])),
                matcher_cache_path=str(oracle.matcher_cache_path),
                timeout_seconds=float(oracle.timeout_seconds),
                max_retries=int(oracle.max_retries),
                teacher_validity_max_retries=int(
                    oracle.teacher_validity_max_retries
                ),
                max_concurrent_requests=int(oracle.max_concurrent_requests),
                teacher_multi_call_fallback_enabled=bool(
                    config.env.rollout.teacher_multi_call_fallback.enabled
                ),
                teacher_multi_call_fallback_min_repeat_streak=int(
                    config.env.rollout.teacher_multi_call_fallback.min_repeat_streak
                ),
                runtime_judge_enabled=True,
                runtime_judge_data_dir=str(runtime_judge.data_dir),
                runtime_judge_reference_trials_path=(
                    str(runtime_judge.reference_trials_path)
                    if runtime_judge.reference_trials_path
                    else None
                ),
                runtime_judge_cache_path=str(runtime_judge.cache_path),
                runtime_judge_provider=str(runtime_judge.provider),
                runtime_judge_model=str(runtime_judge.model),
                runtime_judge_api_base=str(runtime_judge.api_base),
                runtime_judge_api_key_env=str(runtime_judge.api_key_env),
                runtime_judge_reasoning_effort=str(
                    runtime_judge.reasoning_effort
                ),
                runtime_judge_max_tokens=int(runtime_judge.max_tokens),
                runtime_judge_max_format_retries=int(runtime_judge.max_format_retries),
            )
        envs = None
        if not val_only:
            vector = build_mixed_agentic_envs(
                seed=int(config.env.seed),
                counts=counts,
                env_config=config.env,
                oracle_actor=oracle_actor,
            )
            envs = MixedAgenticEnvironmentManager(
                vector,
                awm_projection,
                config,
                oracle_actor=oracle_actor,
            )

        if not _trainer_validation_enabled(config):
            return envs, None

        if str(config.env.validation.env_name).lower() != "tau":
            raise ValueError(
                "mixed AWM/EnvScaler periodic validation must use Tau"
            )
        from agent_system.environments.env_package.tau_bench.envs import (
            build_tau_bench_envs,
            validate_tau_runtime_config,
            validate_tau_source,
        )
        from agent_system.environments.env_package.tau_bench.manager import (
            TauBenchEnvironmentManager,
            tau_projection,
        )

        validate_tau_source(config.env.tau.source_root)
        validate_tau_runtime_config(config.env.tau, require_oracle=False)
        validation_counts = OmegaConf.to_container(
            config.env.tau.validation_counts, resolve=True
        )
        if sum(int(value) for value in validation_counts.values()) != int(
            config.data.val_batch_size
        ):
            raise ValueError(
                "Tau validation counts must sum to data.val_batch_size"
            )
        val_vector = build_tau_bench_envs(
            seed=int(config.env.tau.eval_seed),
            counts=validation_counts,
            env_config=config.env,
            group_n=1,
            is_train=False,
            oracle_actor=None,
        )
        val_envs = TauBenchEnvironmentManager(
            val_vector, tau_projection, config
        )
        return envs, val_envs
    if mixed_env_name in {"awm_agentic_opd", "awm_outcome"}:
        if bool(
            getattr(config.env.awm.oracle, "use_privileged_context", False)
        ):
            raise ValueError(
                "AWM does not support privileged teacher context; set "
                "env.awm.oracle.use_privileged_context=false"
            )
        expected_mode = "state_group" if mixed_env_name == "awm_agentic_opd" else "vanilla"
        if rollout_mode != expected_mode:
            raise ValueError(f"{mixed_env_name} requires env.rollout.mode={expected_mode}")
        expected_reward_mode = (
            "semantic" if mixed_env_name == "awm_agentic_opd" else "outcome"
        )
        if str(config.env.awm.reward_mode) != expected_reward_mode:
            raise ValueError(
                f"{mixed_env_name} requires env.awm.reward_mode={expected_reward_mode}"
            )
        if str(config.env.awm.verifier_mode) != "sql":
            raise ValueError("AWM training requires env.awm.verifier_mode=sql")
        terminal_judge = getattr(config.env.awm, "terminal_judge", None)
        if terminal_judge is None or not bool(terminal_judge.enabled):
            raise ValueError("AWM training requires the terminal SQL+LLM judge")
        for field in ("model", "api_base", "api_key_env"):
            if not str(getattr(terminal_judge, field, "") or "").strip():
                raise ValueError(f"AWM terminal judge {field} must be non-empty")
        if str(terminal_judge.reasoning_effort) != "max":
            raise ValueError("AWM terminal judge requires reasoning_effort=max")
        if int(terminal_judge.max_tokens) < 8192:
            raise ValueError("AWM terminal judge requires max_tokens >= 8192")
        if float(terminal_judge.timeout_seconds) <= 0:
            raise ValueError("AWM terminal judge timeout_seconds must be positive")
        if int(terminal_judge.max_retries) < 0:
            raise ValueError("AWM terminal judge max_retries must be non-negative")
        context = getattr(config.env, "context", None)
        if context is None or str(context.history_policy) != "token_budget":
            raise ValueError("AWM training requires env.context.history_policy=token_budget")
        if (
            context.max_history_exchanges is not None
            and int(context.max_history_exchanges) < 0
        ):
            raise ValueError("max_history_exchanges must be non-negative")
        if int(config.env.awm.train_max_steps) != 20:
            raise ValueError("AWM training protocol requires env.awm.train_max_steps=20")
        if int(config.env.max_steps) != 20:
            raise ValueError("AWM training protocol requires env.max_steps=20")
        if int(config.env.rollout.n) != 4:
            raise ValueError("AWM training protocol requires env.rollout.n=4")
        _validate_awm_context_budget(config)
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError(
                "AWM protocol requires actor_rollout_ref.rollout.n=1; "
                "env.rollout.n controls the four candidates"
            )
        if not bool(config.actor_rollout_ref.rollout.multi_turn.enable):
            raise ValueError(
                "AWM protocol requires actor_rollout_ref.rollout.multi_turn.enable=true"
            )
        expected_estimator = "dapo" if mixed_env_name == "awm_agentic_opd" else "grpo"
        if str(config.algorithm.adv_estimator) != expected_estimator:
            raise ValueError(
                f"{mixed_env_name} requires algorithm.adv_estimator={expected_estimator}"
            )
        if mixed_env_name == "awm_agentic_opd":
            _validate_teacher_reward(config)
            runtime_failures = getattr(config.env.awm, "runtime_failures", None)
            if runtime_failures is None or not bool(runtime_failures.enabled):
                raise ValueError("AWM agentic OPD training requires runtime-failure handling")
            if int(runtime_failures.protocol_version) != 2:
                raise ValueError("AWM runtime-failure protocol mismatch")
            if not str(runtime_failures.path).strip():
                raise ValueError("AWM runtime-failure path must be non-empty")
            runtime_judge = getattr(runtime_failures, "judge", None)
            if runtime_judge is None or not bool(runtime_judge.enabled):
                raise ValueError("AWM agentic OPD training requires runtime 5xx judge")
            confidence_threshold = int(runtime_judge.confidence_threshold)
            if not 0 <= confidence_threshold <= 100:
                raise ValueError(
                    "AWM runtime judge confidence threshold must be in [0, 100]"
                )
            if str(runtime_judge.reasoning_effort) != "max":
                raise ValueError(
                    "AWM runtime judge requires reasoning_effort=max"
                )
            if int(runtime_judge.max_tokens) < 8192:
                raise ValueError("AWM runtime judge requires max_tokens >= 8192")
            for field in ("data_dir", "cache_path"):
                if not str(getattr(runtime_judge, field, "") or "").strip():
                    raise ValueError(
                        f"AWM runtime judge {field} must be non-empty"
                    )

        from agent_system.environments.env_package.awm.runtime.envs import build_awm_envs
        from agent_system.environments.env_package.awm.runtime.manager import (
            AWMEnvironmentManager,
            awm_projection,
        )

        oracle_actor = None
        val_only = bool(config.trainer.get("val_only", False))
        if mixed_env_name == "awm_agentic_opd" and not val_only:
            from agent_system.environments.env_package.awm.runtime.oracle import (
                DeepSeekAWMOracleActor,
            )

            oracle_actor = DeepSeekAWMOracleActor.remote(
                model=str(config.env.awm.oracle.model),
                provider=str(config.env.awm.oracle.provider),
                api_base=str(config.env.awm.oracle.api_base),
                api_key_env=str(config.env.awm.oracle.api_key_env),
                samples=int(config.env.awm.oracle.samples),
                enable_thinking=bool(config.env.awm.oracle.enable_thinking),
                reasoning_effort=(str(config.env.awm.oracle.reasoning_effort) if config.env.awm.oracle.reasoning_effort is not None else None),
                thinking_budget=(int(config.env.awm.oracle.thinking_budget) if config.env.awm.oracle.thinking_budget is not None else None),
                temperature=(float(config.env.awm.oracle.temperature) if config.env.awm.oracle.temperature is not None else None),
                top_p=(float(config.env.awm.oracle.top_p) if config.env.awm.oracle.top_p is not None else None),
                presence_penalty=(float(config.env.awm.oracle.presence_penalty) if config.env.awm.oracle.presence_penalty is not None else None),
                max_tokens=int(config.env.awm.oracle.max_tokens),
                matcher_provider=str(config.env.awm.oracle.matcher_provider),
                matcher_model=str(config.env.awm.oracle.matcher_model),
                matcher_api_base=str(config.env.awm.oracle.matcher_api_base),
                matcher_api_key_env=str(config.env.awm.oracle.matcher_api_key_env),
                matcher_enable_thinking=config.env.awm.oracle.get("matcher_enable_thinking"),
                matcher_reasoning_effort=config.env.awm.oracle.get("matcher_reasoning_effort"),
                matcher_max_tokens=config.env.awm.oracle.get("matcher_max_tokens"),
                matcher_max_concurrent_requests=config.env.awm.oracle.get("matcher_max_concurrent_requests", 32),
                cache_path=str(config.env.awm.oracle.cache_path),
                teacher_cache_import_paths=list(config.env.awm.oracle.get("teacher_cache_import_paths", [])),
                teacher_cache_import_prompt_hashes=list(config.env.awm.oracle.get("teacher_cache_import_prompt_hashes", [])),
                matcher_cache_path=str(config.env.awm.oracle.matcher_cache_path),
                timeout_seconds=float(config.env.awm.oracle.timeout_seconds),
                max_retries=int(config.env.awm.oracle.max_retries),
                teacher_validity_max_retries=int(
                    config.env.awm.oracle.teacher_validity_max_retries
                ),
                max_concurrent_requests=int(
                    config.env.awm.oracle.max_concurrent_requests
                ),
                teacher_multi_call_fallback_enabled=bool(
                    config.env.rollout.teacher_multi_call_fallback.enabled
                ),
                teacher_multi_call_fallback_min_repeat_streak=int(
                    config.env.rollout.teacher_multi_call_fallback.min_repeat_streak
                ),
                runtime_judge_enabled=bool(runtime_failures.judge.enabled),
                runtime_judge_data_dir=str(runtime_failures.judge.data_dir),
                runtime_judge_reference_trials_path=(
                    str(runtime_failures.judge.reference_trials_path)
                    if runtime_failures.judge.reference_trials_path
                    else None
                ),
                runtime_judge_cache_path=str(runtime_failures.judge.cache_path),
                runtime_judge_provider=str(runtime_failures.judge.provider),
                runtime_judge_model=str(runtime_failures.judge.model),
                runtime_judge_api_base=str(runtime_failures.judge.api_base),
                runtime_judge_api_key_env=str(runtime_failures.judge.api_key_env),
                runtime_judge_reasoning_effort=str(
                    runtime_failures.judge.reasoning_effort
                ),
                runtime_judge_max_tokens=int(runtime_failures.judge.max_tokens),
                runtime_judge_max_format_retries=int(runtime_failures.judge.max_format_retries),
            )
        _envs = None
        if not val_only:
            _envs = build_awm_envs(
                seed=int(config.env.seed),
                count=int(config.data.train_batch_size),
                env_config=config.env,
                is_train=True,
                group_n=group_n,
                oracle_actor=oracle_actor,
            )
        envs = (
            None
            if val_only
            else AWMEnvironmentManager(
                _envs,
                awm_projection,
                config,
                oracle_actor=oracle_actor,
            )
        )
        if not _trainer_validation_enabled(config):
            return envs, None
        validation_env_name = str(config.env.validation.env_name).lower()
        if validation_env_name == "awm":
            _val_envs = build_awm_envs(
                seed=int(config.env.awm.eval_seed),
                count=int(config.data.val_batch_size),
                env_config=config.env,
                is_train=False,
                group_n=1,
                oracle_actor=None,
            )
            val_envs = AWMEnvironmentManager(_val_envs, awm_projection, config)
        elif validation_env_name == "tau":
            from agent_system.environments.env_package.tau_bench.envs import (
                build_tau_bench_envs,
                validate_tau_runtime_config,
                validate_tau_source,
            )
            from agent_system.environments.env_package.tau_bench.manager import (
                TauBenchEnvironmentManager,
                tau_projection,
            )

            validate_tau_source(config.env.tau.source_root)
            validate_tau_runtime_config(config.env.tau, require_oracle=False)
            validation_counts = OmegaConf.to_container(
                config.env.tau.validation_counts, resolve=True
            )
            if sum(int(value) for value in validation_counts.values()) != int(
                config.data.val_batch_size
            ):
                raise ValueError(
                    "Tau validation counts must sum to data.val_batch_size"
                )
            _val_envs = build_tau_bench_envs(
                seed=int(config.env.tau.eval_seed),
                counts=validation_counts,
                env_config=config.env,
                group_n=1,
                is_train=False,
                oracle_actor=None,
            )
            val_envs = TauBenchEnvironmentManager(
                _val_envs, tau_projection, config
            )
        else:
            raise ValueError(
                f"unsupported AWM validation environment: {validation_env_name}"
            )
        return envs, val_envs
    elif mixed_env_name in {"tau_agentic_opd", "tau_outcome"}:
        _validate_teacher_reward(config)
        expected_mode = "state_group" if mixed_env_name == "tau_agentic_opd" else "vanilla"
        if rollout_mode != expected_mode:
            raise ValueError(
                f"{mixed_env_name} requires env.rollout.mode={expected_mode}"
            )
        from agent_system.environments.env_package.tau_bench.envs import (
            build_tau_bench_envs,
            validate_tau_runtime_config,
            validate_tau_source,
        )
        from agent_system.environments.env_package.tau_bench.manager import (
            TauBenchEnvironmentManager,
            tau_projection,
        )

        validate_tau_source(config.env.tau.source_root)
        validate_tau_runtime_config(
            config.env.tau,
            require_oracle=mixed_env_name == "tau_agentic_opd",
        )
        train_counts = OmegaConf.to_container(
            config.env.tau.trajectory_counts, resolve=True
        )
        validation_counts = OmegaConf.to_container(
            config.env.tau.validation_counts, resolve=True
        )
        if sum(int(value) for value in train_counts.values()) != int(
            config.data.train_batch_size
        ):
            raise ValueError("Tau training counts must sum to data.train_batch_size")
        if sum(int(value) for value in validation_counts.values()) != int(
            config.data.val_batch_size
        ):
            raise ValueError(
                "Tau validation counts must sum to data.val_batch_size"
            )

        oracle_actor = None
        val_only = bool(config.trainer.get("val_only", False))
        if mixed_env_name == "tau_agentic_opd" and not val_only:
            from agent_system.environments.env_package.tau_bench.oracle import (
                TauTeacherActor,
            )

            oracle_actor = TauTeacherActor.remote(
                model=str(config.env.tau.oracle.model),
                api_base=str(config.env.tau.oracle.api_base),
                api_key_env=str(config.env.tau.oracle.api_key_env),
                samples=int(config.env.tau.oracle.samples),
                temperature=float(config.env.tau.oracle.temperature),
                top_p=float(config.env.tau.oracle.top_p),
                top_k=int(config.env.tau.oracle.top_k),
                min_p=float(config.env.tau.oracle.min_p),
                enable_thinking=bool(config.env.tau.oracle.enable_thinking),
                max_tokens=int(config.env.tau.oracle.max_tokens),
                cache_path=str(config.env.tau.oracle.cache_path),
                teacher_cache_import_paths=list(config.env.tau.oracle.get("teacher_cache_import_paths", [])),
                matcher_enabled=not bool(config.env.tau.get("mask_matcher_required_groups", False)),
                matcher_cache_path=str(config.env.tau.oracle.matcher_cache_path),
                matcher_provider=str(config.env.tau.oracle.get("matcher_provider", "openai-compatible")),
                matcher_model=config.env.tau.oracle.get("matcher_model"),
                matcher_api_base=config.env.tau.oracle.get("matcher_api_base"),
                matcher_api_key_env=config.env.tau.oracle.get("matcher_api_key_env"),
                matcher_enable_thinking=config.env.tau.oracle.get("matcher_enable_thinking"),
                matcher_max_tokens=config.env.tau.oracle.get("matcher_max_tokens"),
                matcher_reasoning_effort=config.env.tau.oracle.get("matcher_reasoning_effort"),
                matcher_max_concurrent_requests=int(config.env.tau.oracle.get("matcher_max_concurrent_requests", 32)),
                timeout_seconds=float(config.env.tau.oracle.timeout_seconds),
                max_retries=int(config.env.tau.oracle.max_retries),
                teacher_validity_max_retries=int(
                    config.env.tau.oracle.teacher_validity_max_retries
                ),
                max_concurrent_requests=int(
                    config.env.tau.oracle.max_concurrent_requests
                ),
            )
        _envs = None
        if not val_only:
            _envs = build_tau_bench_envs(
                seed=int(config.env.seed),
                counts=train_counts,
                group_n=group_n,
                env_config=config.env,
                is_train=True,
                oracle_actor=oracle_actor,
            )
        _val_envs = build_tau_bench_envs(
            seed=int(config.env.tau.eval_seed),
            counts=validation_counts,
            env_config=config.env,
            group_n=1,
            is_train=False,
            oracle_actor=None,
        )
        envs = (
            None
            if val_only
            else TauBenchEnvironmentManager(
                _envs,
                tau_projection,
                config,
                oracle_actor=oracle_actor,
            )
        )
        val_envs = TauBenchEnvironmentManager(_val_envs, tau_projection, config)
        return envs, val_envs
    elif mixed_env_name in {"dapo_vpr_mixed", "dapo_games_non_vpr_mixed"}:
        expected_mode = (
            "state_group" if mixed_env_name == "dapo_vpr_mixed" else "vanilla"
        )
        if rollout_mode != expected_mode:
            raise ValueError(
                f"{mixed_env_name} requires env.rollout.mode={expected_mode}"
            )
        from agent_system.environments.env_package.vpr_games.mixed import (
            MixedVPRManager,
            build_mixed_vpr_envs,
            mixed_vpr_projection,
        )

        train_counts = OmegaConf.to_container(
            config.env.mixed.trajectory_counts, resolve=True
        )
        validation_counts = OmegaConf.to_container(
            config.env.mixed.validation_counts, resolve=True
        )
        if sum(int(value) for value in train_counts.values()) != int(config.data.train_batch_size):
            raise ValueError("mixed training counts must sum to data.train_batch_size")
        if sum(int(value) for value in validation_counts.values()) != int(config.data.val_batch_size):
            raise ValueError("mixed validation counts must sum to data.val_batch_size")
        _envs = build_mixed_vpr_envs(
            seed=config.env.seed,
            counts=train_counts,
            env_config=config.env,
            is_train=True,
            group_n=group_n,
        )
        _val_envs = build_mixed_vpr_envs(
            seed=config.env.seed + 1000,
            counts=validation_counts,
            env_config=config.env,
            is_train=False,
            group_n=1,
        )
        envs = MixedVPRManager(_envs, mixed_vpr_projection, config)
        val_envs = MixedVPRManager(_val_envs, mixed_vpr_projection, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
            'deterministic_eval': config.env.alfworld.get('deterministic_eval', False),
        }
        val_only = config.trainer.get('val_only', False)
        _envs = None
        if not val_only:
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(
            alfworld_projection,
            native_action_protocol=config.env.agentic_eval.native_action_protocol,
        )
        envs = None if val_only else AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "vpr_sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.vpr_games.sokoban.envs import build_sokoban_envs
        from agent_system.environments.env_package.vpr_games.sokoban.manager import (
            SokobanEnvironmentManager as VPRSokobanEnvironmentManager,
        )
        from agent_system.environments.env_package.vpr_games.sokoban.manager import (
            sokoban_projection as vpr_sokoban_projection,
        )
        _envs = build_sokoban_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                    group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_sokoban_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                        group_n=1, is_train=False, env_config=config.env)
        envs = VPRSokobanEnvironmentManager(_envs, partial(vpr_sokoban_projection), config)
        val_envs = VPRSokobanEnvironmentManager(_val_envs, partial(vpr_sokoban_projection), config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        data_dir = config.env.webshop.get('data_dir')
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data')
        if config.env.webshop.use_small:
            file_path = os.path.join(data_dir, 'items_shuffle_1000.json')
            attr_path = os.path.join(data_dir, 'items_ins_v2_1000.json')
        else:
            file_path = os.path.join(data_dir, 'items_shuffle.json')
            attr_path = os.path.join(data_dir, 'items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path,
                    'deterministic_eval': config.env.webshop.get('deterministic_eval', False),
                    'shared_server': config.env.webshop.get('shared_server', False),
                    }
        val_only = config.trainer.get('val_only', False)
        _envs = None
        if not val_only:
            _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(
            webshop_projection,
            native_action_protocol=config.env.agentic_eval.native_action_protocol,
        )
        envs = None if val_only else WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        train_env_count = 0 if val_only else config.data.train_batch_size * group_n
        time.sleep((train_env_count + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import appworld_projection, build_appworld_envs
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "vpr_tictactoe" in config.env.env_name.lower():
        from agent_system.environments.env_package.vpr_games.tictactoe.envs import build_tictactoe_envs
        from agent_system.environments.env_package.vpr_games.tictactoe.manager import (
            TicTacToeEnvironmentManager,
            tictactoe_projection,
        )
        _envs = build_tictactoe_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                      group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_tictactoe_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                          group_n=1, is_train=False, env_config=config.env)
        envs = TicTacToeEnvironmentManager(_envs, partial(tictactoe_projection), config)
        val_envs = TicTacToeEnvironmentManager(_val_envs, partial(tictactoe_projection), config)
        return envs, val_envs
    elif "vpr_sudoku" in config.env.env_name.lower():
        from agent_system.environments.env_package.vpr_games.sudoku.envs import build_sudoku_envs
        from agent_system.environments.env_package.vpr_games.sudoku.manager import (
            SudokuEnvironmentManager,
            sudoku_projection,
        )
        _envs = build_sudoku_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                   group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_sudoku_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                       group_n=1, is_train=False, env_config=config.env)
        envs = SudokuEnvironmentManager(_envs, partial(sudoku_projection), config)
        val_envs = SudokuEnvironmentManager(_val_envs, partial(sudoku_projection), config)
        return envs, val_envs
    elif "vpr_minesweeper" in config.env.env_name.lower():
        from agent_system.environments.env_package.vpr_games.minesweeper.envs import build_minesweeper_envs
        from agent_system.environments.env_package.vpr_games.minesweeper.manager import (
            MinesweeperEnvironmentManager,
            minesweeper_projection,
        )
        _envs = build_minesweeper_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                        group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_minesweeper_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                            group_n=1, is_train=False, env_config=config.env)
        envs = MinesweeperEnvironmentManager(_envs, partial(minesweeper_projection), config)
        val_envs = MinesweeperEnvironmentManager(_val_envs, partial(minesweeper_projection), config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
