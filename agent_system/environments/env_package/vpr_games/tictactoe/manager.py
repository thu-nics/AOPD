"""TicTacToe environment manager for VPR training."""

from __future__ import annotations

from typing import List, Dict

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import TICTACTOE_TEMPLATE


def tictactoe_projection(text_actions: List[str]):
    """Pass raw text to workers (workers handle parsing internally)."""
    return text_actions, [True] * len(text_actions)


class TicTacToeEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_tictactoe."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        out = []
        for info in infos:
            mark = info.get("agent_player", "X")
            opp = "O" if mark == "X" else "X"
            out.append(TICTACTOE_TEMPLATE.format(
                board=info.get("observation", ""), mark=mark, opp=opp))
        return out

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        """Win / draw / loss breakdown plus the rate of episodes ended by an illegal move."""
        result = None
        for si in reversed(episode_info_list):
            if si.get("game_result") in ("win", "loss", "draw"):
                result = si["game_result"]
                break
        ended_illegal = any(si.get("terminal_reason") == "invalid_action"
                            for si in episode_info_list)
        return {
            "env/win_rate": 1.0 if result == "win" else 0.0,
            "env/draw_rate": 1.0 if result == "draw" else 0.0,
            "env/loss_rate": 1.0 if result == "loss" else 0.0,
            "env/illegal_end_rate": 1.0 if ended_illegal else 0.0,
        }
