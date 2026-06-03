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
