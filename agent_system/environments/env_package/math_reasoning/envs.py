"""One-turn rule-reward environment for mathematical reasoning RL."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase
from verl.utils.reward_score.math_dapo import compute_score


class MathReasoningEnvs:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self._items: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(0)

    def reset(self, kwargs: list[dict[str, Any]]):
        if kwargs is None:
            raise ValueError("math_reasoning requires env_kwargs in every data row")
        if len(kwargs) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} math items, received {len(kwargs)}"
            )
        self._items = [self._resolve_prompt_pool(item) for item in kwargs]
        observations = [str(item["question"]) for item in self._items]
        infos = [
            {
                "data_source": str(item.get("data_source", "math_dapo")),
                "observation": observation,
                "vpr_game": "math",
            }
            for item, observation in zip(self._items, observations, strict=True)
        ]
        return observations, infos

    @staticmethod
    def _resolve_prompt_pool(item: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(item)
        questions = list(resolved.get("question_pool") or [])
        if not questions:
            return resolved

        ground_truths = list(resolved.get("ground_truth_pool") or [])
        data_sources = list(resolved.get("data_source_pool") or [])
        if not (len(questions) == len(ground_truths) == len(data_sources)):
            raise ValueError("math prompt pools must have equal lengths")
        pool_index = int(resolved.get("dynamic_attempt", 0)) % len(questions)
        resolved["question"] = questions[pool_index]
        resolved["ground_truth"] = ground_truths[pool_index]
        resolved["data_source"] = data_sources[pool_index]
        return resolved

    @staticmethod
    def _evaluate(action: str, item: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        result = compute_score(str(action), str(item["ground_truth"]))
        score = float(result["score"])
        correct = bool(result["acc"])
        parse_ok = str(result.get("pred", "[INVALID]")) != "[INVALID]"
        return score, {
            "data_source": str(item.get("data_source", "math_dapo")),
            "won": correct,
            "terminal_success": correct,
            "is_action_valid": parse_ok,
            "parse_ok": parse_ok,
            "illegal_action": not parse_ok,
            "observation": "",
            "vpr_game": "math",
        }

    def step(self, actions: list[str]):
        if len(actions) != len(self._items):
            raise ValueError(
                f"Expected {len(self._items)} math actions, received {len(actions)}"
            )
        rewards: list[float] = []
        infos: list[dict[str, Any]] = []
        for action, item in zip(actions, self._items, strict=True):
            result = compute_score(str(action), str(item["ground_truth"]))
            score = float(result["score"])
            correct = bool(result["acc"])
            parse_ok = str(result.get("pred", "[INVALID]")) != "[INVALID]"
            rewards.append(score)
            infos.append(
                {
                    "data_source": str(item.get("data_source", "math_dapo")),
                    "won": correct,
                    "terminal_success": correct,
                    "is_action_valid": parse_ok,
                    "parse_ok": parse_ok,
                    "illegal_action": not parse_ok,
                }
            )
        return [""] * len(actions), rewards, [True] * len(actions), infos

    def step_candidate_groups(
        self,
        candidate_action_groups: list[list[str]],
        active_indices=None,
        selection_mode: str = "best",
        random_select_prob: float = 0.0,
    ):
        if active_indices is None:
            active_indices = range(len(candidate_action_groups))
        item_indices = [int(index) for index in active_indices]
        if len(item_indices) != len(candidate_action_groups):
            raise ValueError("active_indices must align with candidate_action_groups")

        candidate_results = []
        selected_indices = []
        observations = []
        rewards = []
        dones = []
        infos = []
        for item_index, actions in zip(item_indices, candidate_action_groups, strict=True):
            if not actions:
                raise ValueError("math candidate groups must be non-empty")
            item = self._items[item_index]
            candidates = []
            for candidate_index, action in enumerate(actions):
                score, info = self._evaluate(action, item)
                info["candidate_index"] = candidate_index
                info["env_done"] = True
                candidates.append(("", score, True, info))

            best_index = int(np.argmax([candidate[1] for candidate in candidates]))
            selected_index = best_index
            selection_type = "best"
            if selection_mode == "random" or (
                selection_mode == "mixed" and self._rng.random() < float(random_select_prob)
            ):
                selected_index = int(self._rng.integers(len(actions)))
                selection_type = "random"

            selected_reward = float(candidates[selected_index][1])
            selected_info = dict(candidates[selected_index][3])
            selected_info.update(
                {
                    "best_candidate_index": best_index,
                    "state_group_selected": True,
                    "state_group_selection_type": selection_type,
                    "state_group_random_selected": selection_type == "random",
                    "state_group_random_select_prob": float(random_select_prob),
                }
            )
            candidate_results.append(candidates)
            selected_indices.append(selected_index)
            observations.append("")
            rewards.append(selected_reward)
            dones.append(True)
            infos.append(selected_info)

        return (
            candidate_results,
            np.asarray(selected_indices, dtype=np.int32),
            observations,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        self._items = []


class MathReasoningEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs: MathReasoningEnvs, config) -> None:
        super().__init__(envs, projection_f=None, config=config)

    def reset(self, kwargs):
        observations, infos = self.envs.reset(kwargs)
        return {
            "text": observations,
            "image": None,
            "anchor": observations.copy(),
        }, infos

    def step(self, text_actions: list[str]):
        observations, rewards, dones, infos = self.envs.step(text_actions)
        return {
            "text": observations,
            "image": None,
            "anchor": observations.copy(),
        }, np.asarray(rewards), np.asarray(dones), infos

    def success_evaluator(self, total_infos=None, **kwargs):
        if total_infos is None:
            return {"env/success_rate": np.asarray([], dtype=np.float32)}
        success = []
        valid_action_rate = []
        for episode in total_infos:
            success.append(
                float(
                    any(
                        bool(info.get("terminal_success", False))
                        for info in episode
                    )
                )
            )
            valid_action_rate.append(
                float(
                    np.mean(
                        [bool(info.get("is_action_valid", True)) for info in episode]
                    )
                )
                if episode
                else 0.0
            )
        return {
            "env/success_rate": np.asarray(success, dtype=np.float32),
            "env/valid_action_rate": np.asarray(
                valid_action_rate, dtype=np.float32
            ),
        }

    def state_group_step(self, candidate_text_action_groups, active_indices=None):
        rollout_config = self.config.env.rollout
        results = self.envs.step_candidate_groups(
            candidate_text_action_groups,
            active_indices=active_indices,
            selection_mode=rollout_config.selection_mode,
            random_select_prob=rollout_config.random_select_prob,
        )
        candidate_results, selected_indices, observations, rewards, dones, infos = results
        return (
            candidate_results,
            selected_indices,
            {"text": observations, "image": None, "anchor": None},
            rewards,
            dones,
            infos,
        )


def build_math_reasoning_envs(*, env_num: int, group_n: int):
    return MathReasoningEnvs(batch_size=env_num * group_n)
