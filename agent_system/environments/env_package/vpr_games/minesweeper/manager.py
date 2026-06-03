"""Minesweeper environment manager for VPR training."""

from __future__ import annotations

from typing import List, Dict

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import MINESWEEPER_TEMPLATE


def minesweeper_projection(text_actions: List[str]):
    """Pass raw text to workers (workers handle parsing internally)."""
    return text_actions, [True] * len(text_actions)


class MinesweeperEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_minesweeper."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        obs_list = []
        for info in infos:
            rows = self.config.env.get("rows", 5) if hasattr(self.config.env, "get") else 5
            cols = self.config.env.get("cols", 5) if hasattr(self.config.env, "get") else 5
            mines = self.config.env.get("mines", 5) if hasattr(self.config.env, "get") else 5
            try:
                ms_cfg = getattr(self.config.env, "minesweeper", None)
                if ms_cfg:
                    rows = getattr(ms_cfg, "rows", rows)
                    cols = getattr(ms_cfg, "cols", cols)
                    mines = getattr(ms_cfg, "mines", mines)
            except Exception:
                pass

            board = info.get("observation", "")
            available = info.get("available_actions", [])
            unrevealed_str = ", ".join(available[:15])
            if len(available) > 15:
                unrevealed_str += f"... ({len(available)} total)"
            flagged = info.get("flagged_cells", [])
            flagged_str = ", ".join(flagged[:15]) if flagged else "none"
            if len(flagged) > 15:
                flagged_str += f"... ({len(flagged)} total)"

            obs_list.append(MINESWEEPER_TEMPLATE.format(
                rows=rows, cols=cols, mines=mines,
                board=board,
                unrevealed_cells=unrevealed_str if unrevealed_str else "none",
                flagged_cells=flagged_str,
            ))
        return obs_list

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        """Final fraction of safe cells revealed, plus the rate of mine-hit endings."""
        completion = 0.0
        for si in reversed(episode_info_list):
            if si.get("completion_rate") is not None:
                completion = float(si["completion_rate"])
                break
        mine_hit = any(si.get("terminal_reason") == "mine_hit" for si in episode_info_list)
        return {
            "env/completion_rate": completion,
            "env/mine_hit_rate": 1.0 if mine_hit else 0.0,
        }
