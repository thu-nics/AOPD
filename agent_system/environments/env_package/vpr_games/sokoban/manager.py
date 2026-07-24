"""Sokoban environment manager for VPR training."""

from __future__ import annotations

from typing import Dict, List

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import get_vpr_game_template


def sokoban_projection(text_actions: List[str]):
    """Pass raw text to workers; workers parse <action> tags."""
    return text_actions, [True] * len(text_actions)


class SokobanEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_sokoban."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        obs_list = []
        template = get_vpr_game_template(
            "sokoban", getattr(self.config.env, "game_action_format", "action_tag")
        )
        for info in infos:
            board = info.get("observation", "")
            oracle_actions = info.get("oracle_valid_actions", [])
            oracle_text = ", ".join(oracle_actions) if oracle_actions else "unknown"
            num_boxes = info.get("num_boxes")
            if num_boxes is None:
                cfg = getattr(self.config.env, "sokoban", None)
                num_boxes = getattr(cfg, "num_boxes", 1) if cfg is not None else 1
            obs_list.append(template.format(
                board=board,
                num_boxes=int(num_boxes),
                oracle_hint=oracle_text,
            ))
        return obs_list

    def state_group_step(self, candidate_text_action_groups: List[List[str]], active_indices=None) -> tuple:
        rollout_cfg = getattr(self.config.env, "rollout", None)
        selection_mode = getattr(rollout_cfg, "selection_mode", "best") if rollout_cfg is not None else "best"
        random_select_prob = getattr(rollout_cfg, "random_select_prob", 0.0) if rollout_cfg is not None else 0.0
        candidate_results, selected_indices, next_obs, rewards, dones, infos = \
            self.envs.step_candidate_groups(
                candidate_text_action_groups,
                active_indices=active_indices,
                selection_mode=selection_mode,
                random_select_prob=random_select_prob
            )
        for group in candidate_results:
            for _, _, _, info in group:
                info["is_action_valid"] = int(info.get("parse_ok", True) and not info.get("illegal_action", False))
        for info in infos:
            info["is_action_valid"] = int(info.get("parse_ok", True) and not info.get("illegal_action", False))
        next_observations = {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }
        return candidate_results, selected_indices, next_observations, rewards, dones, infos

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        completion = 0.0
        boxes_on_target = 0.0
        terminal_reason = None
        for si in reversed(episode_info_list):
            if si.get("completion_rate") is not None:
                completion = float(si["completion_rate"])
                boxes_on_target = float(si.get("boxes_on_target") or 0.0)
                break
        for si in reversed(episode_info_list):
            reason = si.get("terminal_reason")
            if reason is not None and reason != "already_done":
                terminal_reason = str(reason)
                break

        measured = [si for si in episode_info_list if si.get("move_optimal") is not None]
        denom = float(len(measured)) if measured else 0.0
        pre_exec_matches = sum(
            1 for si in measured if bool(si.get("pre_exec_oracle_match", si.get("move_optimal")))
        )
        legal_non_oracle = sum(1 for si in measured if bool(si.get("legal_non_oracle")))
        oracle_set_sizes = [
            float(si.get("oracle_action_set_size"))
            for si in measured
            if si.get("oracle_action_set_size") is not None
        ]
        path_lens = [
            float(si.get("sokoban_shortest_path_len"))
            for si in measured
            if si.get("sokoban_shortest_path_len") is not None
        ]

        step_denom = float(len(episode_info_list)) if episode_info_list else 0.0
        parse_errors = sum(1 for si in episode_info_list if not bool(si.get("parse_ok", True)))
        illegal_actions = sum(1 for si in episode_info_list if bool(si.get("illegal_action", False)))
        ineffective_actions = sum(1 for si in episode_info_list if si.get("action_effective") is False)
        reward_noise = sum(1 for si in measured if bool(si.get("reward_noise_applied", False)))

        return {
            "env/completion_rate": completion,
            "env/boxes_on_target": boxes_on_target,
            "env/pre_exec_oracle_match_rate": (pre_exec_matches / denom) if denom else 0.0,
            "env/legal_non_oracle_rate": (legal_non_oracle / denom) if denom else 0.0,
            "env/reward_noise_rate": (reward_noise / denom) if denom else 0.0,
            "env/sokoban_oracle_action_set_size_mean": (
                sum(oracle_set_sizes) / len(oracle_set_sizes) if oracle_set_sizes else 0.0
            ),
            "env/sokoban_shortest_path_len_mean": (
                sum(path_lens) / len(path_lens) if path_lens else 0.0
            ),
            "env/parse_error_rate": (parse_errors / step_denom) if step_denom else 0.0,
            "env/illegal_action_rate": (illegal_actions / step_denom) if step_denom else 0.0,
            "env/ineffective_action_rate": (ineffective_actions / step_denom) if step_denom else 0.0,
            "env/terminal_complete_rate": 1.0 if terminal_reason == "complete" else 0.0,
            "env/terminal_timeout_rate": 1.0 if terminal_reason == "timeout" else 0.0,
            "env/terminal_invalid_action_rate": 1.0 if terminal_reason == "invalid_action" else 0.0,
            "env/terminal_deadlock_rate": 1.0 if terminal_reason == "deadlock" else 0.0,
        }
