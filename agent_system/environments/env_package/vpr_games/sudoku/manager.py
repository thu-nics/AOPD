"""Sudoku environment manager for VPR training."""

from __future__ import annotations

from typing import List, Dict

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import SUDOKU_TEMPLATE


def sudoku_projection(text_actions: List[str]):
    """Pass raw text to workers (workers handle parsing internally)."""
    return text_actions, [True] * len(text_actions)


class SudokuEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_sudoku."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        obs_list = []
        for info in infos:
            grid = info.get("observation", "")
            blanks = info.get("available_actions", [])
            blank_str = ", ".join(blanks[:20])  # cap at 20 to keep prompt bounded
            if len(blanks) > 20:
                blank_str += f"... ({len(blanks)} total)"
            obs_list.append(SUDOKU_TEMPLATE.format(
                grid=grid,
                blank_cells=blank_str,
            ))
        return obs_list

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        """Final board completion (fraction of the initial blanks correctly filled)."""
        completion = 0.0
        for si in reversed(episode_info_list):
            if si.get("completion_rate") is not None:
                completion = float(si["completion_rate"])
                break
        return {"env/completion_rate": completion}
