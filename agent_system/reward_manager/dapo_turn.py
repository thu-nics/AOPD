"""DAPO reward shaping for agent state-group rollout rows."""

from __future__ import annotations

from collections import defaultdict

import torch

from verl import DataProto


class DAPOTurnRewardManager:
    """Write environment rewards to response tails with DAPO overlong shaping."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length=False,
        *,
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        if overlong_buffer_cfg is not None and max_resp_len is None:
            raise ValueError("max_resp_len is required when DAPO overlong shaping is configured")

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch:
            result = {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return result if return_dict else result["reward_tensor"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra_info = defaultdict(list)
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            if response_length <= 0:
                continue

            reward = float(item.non_tensor_batch["rewards"])
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                buffer_length = int(self.overlong_buffer_cfg.len)
                expected_length = int(self.max_resp_len) - buffer_length
                exceed_length = response_length - expected_length
                overlong_reward = min(
                    -exceed_length / buffer_length * float(self.overlong_buffer_cfg.penalty_factor),
                    0.0,
                )
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    extra_info["overlong_reward"].append(overlong_reward)
                    extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[index, response_length - 1] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra_info}
        return reward_tensor
