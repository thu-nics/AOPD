"""Minesweeper environment manager for VPR training."""

from __future__ import annotations

from typing import List, Dict

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import get_vpr_game_template


def minesweeper_projection(text_actions: List[str]):
    """Pass raw text to workers (workers handle parsing internally)."""
    return text_actions, [True] * len(text_actions)


def _resolve_random_select_prob(rollout_cfg) -> float:
    if rollout_cfg is None:
        return 0.0
    prob = float(getattr(rollout_cfg, "random_select_prob", 0.0) or 0.0)
    schedule = str(getattr(rollout_cfg, "random_select_prob_schedule", "") or "").strip()
    if not schedule:
        return prob

    current_step = int(getattr(rollout_cfg, "current_step", 0) or 0)
    for item in schedule.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "env.rollout.random_select_prob_schedule entries must be '<step>:<prob>', "
                f"got {item!r}"
            )
        start_step, scheduled_prob = item.split(":", 1)
        if current_step >= int(start_step.strip()):
            prob = float(scheduled_prob.strip())
    return prob


class MinesweeperEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_minesweeper."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        obs_list = []
        template = get_vpr_game_template(
            "minesweeper", getattr(self.config.env, "game_action_format", "action_tag")
        )
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

            obs_list.append(template.format(
                rows=rows, cols=cols, mines=mines,
                board=board,
                unrevealed_cells=unrevealed_str if unrevealed_str else "none",
                flagged_cells=flagged_str,
            ))
        return obs_list

    def state_group_step(self, candidate_text_action_groups: List[List[str]], active_indices=None):
        rollout_cfg = getattr(self.config.env, "rollout", None)
        selection_mode = getattr(rollout_cfg, "selection_mode", "best") if rollout_cfg is not None else "best"
        random_select_prob = _resolve_random_select_prob(rollout_cfg)
        candidate_results, selected_indices, next_obs, rewards, dones, infos = (
            self.envs.step_candidate_groups(
                candidate_text_action_groups,
                active_indices=active_indices,
                selection_mode=selection_mode,
                random_select_prob=random_select_prob
            )
        )
        for group in candidate_results:
            for _, _, _, info in group:
                info["is_action_valid"] = int(
                    info.get("parse_ok", True) and not info.get("illegal_action", False)
                )
                info["state_group_random_select_prob"] = float(random_select_prob)
        for info in infos:
            info["is_action_valid"] = int(
                info.get("parse_ok", True) and not info.get("illegal_action", False)
            )
            info["state_group_random_select_prob"] = float(random_select_prob)
        next_observations = {
            "text": self.build_text_obs(infos),
            "image": None,
            "anchor": None,
        }
        return candidate_results, selected_indices, next_observations, rewards, dones, infos

    def _trajectory_metrics(self, episode_info_list: List[Dict]) -> Dict[str, float]:
        """Final board progress plus reveal/flag oracle-action breakdown."""
        completion = 0.0
        for si in reversed(episode_info_list):
            if si.get("completion_rate") is not None:
                completion = float(si["completion_rate"])
                break

        terminal_reason = None
        terminal_success = False
        last_env_done = False
        saw_already_done = False
        for si in episode_info_list:
            reason = si.get("terminal_reason")
            if reason:
                reason = str(reason)
                if reason == "already_done":
                    saw_already_done = True
                else:
                    terminal_reason = reason
            if si.get("terminal_success") is not None:
                terminal_success = bool(si.get("terminal_success"))
            if "env_done" in si:
                last_env_done = bool(si.get("env_done"))
        if terminal_reason is not None or terminal_success:
            last_env_done = True
        invalid_or_illegal_reasons = {
            "invalid_action",
            "out_of_bounds",
            "cell_already_revealed",
            "cannot_reveal_flagged_cell",
        }
        mine_hit = terminal_reason == "mine_hit"
        terminal_invalid_or_illegal = terminal_reason in invalid_or_illegal_reasons
        terminal_known_failure = (
            mine_hit
            or terminal_invalid_or_illegal
            or terminal_reason == "non_oracle_flag"
            or terminal_reason == "timeout"
        )
        terminal_missing_reason = bool(
            episode_info_list
            and last_env_done
            and terminal_reason is None
            and not terminal_success
        )
        terminal_unfinished = bool(
            episode_info_list
            and not last_env_done
            and terminal_reason is None
            and not terminal_success
        )
        terminal_other = bool(
            terminal_reason is not None
            and not terminal_success
            and not terminal_known_failure
            and terminal_reason != "complete"
        )

        measured = [si for si in episode_info_list if si.get("move_optimal") is not None]
        denom = float(len(measured)) if measured else 0.0
        oracle_reveal = 0
        oracle_flag = 0
        flag_actions = 0
        pre_exec_matches = 0
        legal_non_oracle = 0
        oracle_guess = 0
        safe_reveal_available = 0
        certain_flag_available = 0
        guess_required = 0
        safe_reveal_hits = 0
        certain_flag_hits = 0
        guess_hits = 0
        non_oracle_reveal = 0
        non_oracle_flag = 0
        reveal_posterior_margins = []
        action_posteriors = []
        min_posteriors = []
        oracle_action_set_sizes = []
        hit_by_size = {"1": [0, 0], "2_4": [0, 0], "gt4": [0, 0]}
        for si in measured:
            parsed = str(si.get("parsed_action") or "").strip().lower()
            action_type = parsed.split(maxsplit=1)[0] if parsed else ""
            action_set_size = int(si.get("oracle_action_set_size", 0) or 0)
            oracle_action_set_sizes.append(float(action_set_size))
            pre_exec_match = si.get("pre_exec_oracle_match", si.get("move_optimal"))
            is_match = bool(pre_exec_match)
            if is_match:
                pre_exec_matches += 1
            if bool(si.get("legal_non_oracle")):
                legal_non_oracle += 1
                if action_type == "reveal":
                    non_oracle_reveal += 1
                elif action_type == "flag":
                    non_oracle_flag += 1
            if bool(si.get("oracle_guess")):
                oracle_guess += 1
            if bool(si.get("safe_reveal_available")):
                safe_reveal_available += 1
            if bool(si.get("certain_flag_available")):
                certain_flag_available += 1
            if bool(si.get("guess_required")):
                guess_required += 1
            oracle_tier = si.get("oracle_tier")
            if oracle_tier == "safe_reveal":
                safe_reveal_hits += 1
            elif oracle_tier == "certain_flag":
                certain_flag_hits += 1
            elif oracle_tier == "guess":
                guess_hits += 1
            if action_type == "flag":
                flag_actions += 1
            if bool(si.get("move_optimal")):
                if action_type == "reveal":
                    oracle_reveal += 1
                elif action_type == "flag":
                    oracle_flag += 1
            margin = si.get("reveal_posterior_margin")
            if margin is not None:
                reveal_posterior_margins.append(float(margin))
            action_posterior = si.get("posterior_prob_for_action")
            if action_posterior is not None:
                action_posteriors.append(float(action_posterior))
            min_posterior = si.get("posterior_min_prob")
            if min_posterior is not None:
                min_posteriors.append(float(min_posterior))
            if action_set_size == 1:
                bucket = "1"
            elif 2 <= action_set_size <= 4:
                bucket = "2_4"
            else:
                bucket = "gt4"
            hit_by_size[bucket][1] += 1
            if is_match:
                hit_by_size[bucket][0] += 1

        non_oracle_mine_hit = any(
            si.get("terminal_reason") == "mine_hit" and bool(si.get("legal_non_oracle"))
            for si in episode_info_list
        )
        avg_oracle_action_set_size = (
            sum(oracle_action_set_sizes) / denom if denom else 0.0
        )
        reveal_posterior_margin_mean = (
            sum(reveal_posterior_margins) / len(reveal_posterior_margins)
            if reveal_posterior_margins else 0.0
        )
        action_posterior_mean = (
            sum(action_posteriors) / len(action_posteriors) if action_posteriors else 0.0
        )
        min_posterior_mean = (
            sum(min_posteriors) / len(min_posteriors) if min_posteriors else 0.0
        )
        oracle_hit_size_1 = (hit_by_size["1"][0] / hit_by_size["1"][1]) if hit_by_size["1"][1] else 0.0
        oracle_hit_size_2_4 = (hit_by_size["2_4"][0] / hit_by_size["2_4"][1]) if hit_by_size["2_4"][1] else 0.0
        oracle_hit_size_gt4 = (hit_by_size["gt4"][0] / hit_by_size["gt4"][1]) if hit_by_size["gt4"][1] else 0.0

        return {
            "env/completion_rate": completion,
            "env/mine_hit_rate": 1.0 if mine_hit else 0.0,
            "env/terminal_complete_rate": 1.0 if terminal_success else 0.0,
            "env/terminal_mine_hit_rate": 1.0 if mine_hit else 0.0,
            "env/terminal_invalid_action_rate": 1.0 if terminal_reason == "invalid_action" else 0.0,
            "env/terminal_out_of_bounds_rate": 1.0 if terminal_reason == "out_of_bounds" else 0.0,
            "env/terminal_cell_already_revealed_rate": 1.0 if terminal_reason == "cell_already_revealed" else 0.0,
            "env/terminal_cannot_reveal_flagged_cell_rate": 1.0 if terminal_reason == "cannot_reveal_flagged_cell" else 0.0,
            "env/terminal_invalid_or_illegal_action_rate": 1.0 if terminal_invalid_or_illegal else 0.0,
            "env/terminal_non_oracle_flag_rate": 1.0 if terminal_reason == "non_oracle_flag" else 0.0,
            "env/terminal_timeout_rate": 1.0 if terminal_reason == "timeout" else 0.0,
            "env/terminal_already_done_rate": 1.0 if saw_already_done else 0.0,
            "env/terminal_missing_reason_rate": 1.0 if terminal_missing_reason else 0.0,
            "env/terminal_unfinished_rate": 1.0 if terminal_unfinished else 0.0,
            "env/terminal_other_rate": 1.0 if terminal_other else 0.0,
            "env/oracle_reveal_rate": (oracle_reveal / denom) if denom else 0.0,
            "env/oracle_flag_rate": (oracle_flag / denom) if denom else 0.0,
            "env/flag_action_rate": (flag_actions / denom) if denom else 0.0,
            "env/pre_exec_oracle_match_rate": (pre_exec_matches / denom) if denom else 0.0,
            "env/legal_non_oracle_rate": (legal_non_oracle / denom) if denom else 0.0,
            "env/oracle_action_set_size_mean": avg_oracle_action_set_size,
            "env/oracle_guess_rate": (oracle_guess / denom) if denom else 0.0,
            "env/safe_reveal_available_rate": (safe_reveal_available / denom) if denom else 0.0,
            "env/certain_flag_available_rate": (certain_flag_available / denom) if denom else 0.0,
            "env/guess_required_rate": (guess_required / denom) if denom else 0.0,
            "env/safe_reveal_hit_rate": (safe_reveal_hits / safe_reveal_available) if safe_reveal_available else 0.0,
            "env/certain_flag_hit_rate": (certain_flag_hits / certain_flag_available) if certain_flag_available else 0.0,
            "env/guess_hit_rate": (guess_hits / guess_required) if guess_required else 0.0,
            "env/non_oracle_reveal_rate": (non_oracle_reveal / denom) if denom else 0.0,
            "env/non_oracle_flag_rate": (non_oracle_flag / denom) if denom else 0.0,
            "env/reveal_posterior_margin_mean": reveal_posterior_margin_mean,
            "env/action_posterior_mean": action_posterior_mean,
            "env/min_posterior_mean": min_posterior_mean,
            "env/oracle_hit_rate_action_set_size_1": oracle_hit_size_1,
            "env/oracle_hit_rate_action_set_size_2_4": oracle_hit_size_2_4,
            "env/oracle_hit_rate_action_set_size_gt4": oracle_hit_size_gt4,
            "env/non_oracle_mine_hit_rate": 1.0 if non_oracle_mine_hit else 0.0,
        }
