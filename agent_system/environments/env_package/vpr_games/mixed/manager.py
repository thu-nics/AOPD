"""Environment manager for heterogeneous VPR rollouts."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.env_package.vpr_games.minesweeper.manager import MinesweeperEnvironmentManager
from agent_system.environments.env_package.vpr_games.sokoban.manager import SokobanEnvironmentManager
from agent_system.environments.env_package.vpr_games.sudoku.manager import SudokuEnvironmentManager
from agent_system.environments.prompts.vpr_games import get_vpr_game_template


def mixed_vpr_projection(text_actions: List[str]):
    return text_actions, [True] * len(text_actions)


class MixedVPRManager(VPRBaseEnvironmentManager):
    def reset(self, kwargs=None) -> tuple:
        _, infos = self.envs.reset(kwargs=kwargs)
        return {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }, infos

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        observations = []
        action_format = getattr(self.config.env, "game_action_format", "action_tag")
        for info in infos:
            game = info.get("vpr_game")
            if game == "math":
                observations.append(str(info.get("observation", "")))
            elif game == "sokoban":
                oracle_actions = info.get("oracle_valid_actions", [])
                observations.append(
                    get_vpr_game_template("sokoban", action_format).format(
                        board=info.get("observation", ""),
                        num_boxes=int(info.get("num_boxes", self.config.env.sokoban.num_boxes)),
                        oracle_hint=", ".join(oracle_actions) if oracle_actions else "unknown",
                    )
                )
            elif game == "sudoku":
                blanks = info.get("available_actions", [])
                blank_text = ", ".join(blanks[:20])
                if len(blanks) > 20:
                    blank_text += f"... ({len(blanks)} total)"
                observations.append(
                    get_vpr_game_template("sudoku", action_format).format(
                        grid=info.get("observation", ""),
                        blank_cells=blank_text,
                    )
                )
            elif game == "minesweeper":
                available = info.get("available_actions", [])
                available_text = ", ".join(available[:15])
                if len(available) > 15:
                    available_text += f"... ({len(available)} total)"
                flagged = info.get("flagged_cells", [])
                flagged_text = ", ".join(flagged[:15]) if flagged else "none"
                if len(flagged) > 15:
                    flagged_text += f"... ({len(flagged)} total)"
                cfg = self.config.env.minesweeper
                observations.append(
                    get_vpr_game_template("minesweeper", action_format).format(
                        rows=cfg.rows,
                        cols=cfg.cols,
                        mines=cfg.mines,
                        board=info.get("observation", ""),
                        unrevealed_cells=available_text or "none",
                        flagged_cells=flagged_text,
                    )
                )
            else:
                raise ValueError(f"unknown mixed VPR game: {game!r}")
        return observations

    def state_group_step(self, candidate_text_action_groups, active_indices=None):
        rollout_config = self.config.env.rollout
        results = self.envs.step_candidate_groups(
            candidate_text_action_groups,
            active_indices=active_indices,
            selection_mode=rollout_config.selection_mode,
            random_select_prob=rollout_config.random_select_prob,
        )
        candidate_results, selected_indices, _, rewards, dones, infos = results
        for group in candidate_results:
            for _, _, _, info in group:
                info["is_action_valid"] = int(
                    info.get("parse_ok", True) and not info.get("illegal_action", False)
                )
        for info in infos:
            info["is_action_valid"] = int(
                info.get("parse_ok", True) and not info.get("illegal_action", False)
            )
        observations = {"text": self.build_text_obs(infos), "image": None, "anchor": None}
        return candidate_results, selected_indices, observations, rewards, dones, infos

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        game = next((info.get("vpr_game") for info in episode_info_list if info.get("vpr_game")), None)
        if game == "sokoban":
            return SokobanEnvironmentManager._trajectory_metrics(self, episode_info_list)
        if game == "sudoku":
            return SudokuEnvironmentManager._trajectory_metrics(self, episode_info_list)
        if game == "minesweeper":
            return MinesweeperEnvironmentManager._trajectory_metrics(self, episode_info_list)
        return {}

    def success_evaluator(self, total_infos=None, **kwargs):
        metrics = super().success_evaluator(total_infos=total_infos, **kwargs)
        if total_infos is None:
            return metrics
        games = np.asarray(
            [
                next((info.get("vpr_game") for info in episode if info.get("vpr_game")), "unknown")
                for episode in total_infos
            ],
            dtype=object,
        )
        common_metric_keys = (
            "env/success_rate",
            "env/valid_action_rate",
            "env/oracle_hit_rate",
            "env/completion_rate",
        )
        output = {
            key: metrics[key] for key in common_metric_keys if key in metrics
        }
        for game in ("math", "sokoban", "sudoku", "minesweeper"):
            game_mask = games == game
            output[f"env/{game}/trajectory_count"] = np.asarray([game_mask.sum()], dtype=np.float32)
            if not game_mask.any():
                continue
            game_infos = [episode for episode, selected in zip(total_infos, game_mask) if selected]
            game_metrics = VPRBaseEnvironmentManager.success_evaluator(
                self, total_infos=game_infos
            )
            for key, values in game_metrics.items():
                suffix = key.removeprefix("env/")
                output[f"env/{game}/{suffix}"] = values
        return output
