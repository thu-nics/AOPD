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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_NATIVE_ACTION_TEMPLATE_NO_HIS = """
Solve the following WebShop action-selection problem step by step. Put your answer inside \\boxed{{}}.

Shopping goal: {task_description}
Current observation: {current_observation}
Admissible actions:
[
{available_actions}
].

Choose exactly one executable action. A click action must exactly match one of the admissible actions. A search action must replace `<your query>` with actual search terms; never output the literal search template.
Put only plain action text inside the box, without additional LaTeX commands, quotes, or backticks.
Remember to put your answer inside \\boxed{{}}.
"""

WEBSHOP_NATIVE_ACTION_TEMPLATE = """
Solve the following WebShop action-selection problem step by step. Put your answer inside \\boxed{{}}.

Shopping goal: {task_description}
Recent interaction history ({history_length} of {step_count} previous steps): {action_history}
Current observation: {current_observation}
Admissible actions:
[
{available_actions}
].

Choose exactly one executable action. A click action must exactly match one of the admissible actions. A search action must replace `<your query>` with actual search terms; never output the literal search template.
Put only plain action text inside the box, without additional LaTeX commands, quotes, or backticks.
Remember to put your answer inside \\boxed{{}}.
"""
