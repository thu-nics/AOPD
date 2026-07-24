"""Sudoku environment manager for VPR training."""

from __future__ import annotations

from typing import List, Dict

from agent_system.environments.env_package.vpr_games.common.base_manager import VPRBaseEnvironmentManager
from agent_system.environments.prompts.vpr_games import get_vpr_game_template


def sudoku_projection(text_actions: List[str]):
    """Pass raw text to workers (workers handle parsing internally)."""
    return text_actions, [True] * len(text_actions)


class SudokuEnvironmentManager(VPRBaseEnvironmentManager):
    """Environment manager for vpr_sudoku."""

    def build_text_obs(self, infos: List[Dict]) -> List[str]:
        obs_list = []
        template = get_vpr_game_template(
            "sudoku", getattr(self.config.env, "game_action_format", "action_tag")
        )
        for info in infos:
            grid = info.get("observation", "")
            blanks = info.get("available_actions", [])
            blank_str = ", ".join(blanks[:20])  # cap at 20 to keep prompt bounded
            if len(blanks) > 20:
                blank_str += f"... ({len(blanks)} total)"
            obs_list.append(template.format(
                grid=grid,
                blank_cells=blank_str,
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
        """Final board progress plus Sudoku oracle-action diagnostics."""
        completion = 0.0
        blanks_remaining = 0.0
        initial_blanks = 0.0
        correct_fills = 0.0
        outcome_target = 0.0
        outcome_target_completion = 0.0
        terminal_reason = None
        for si in reversed(episode_info_list):
            if si.get("completion_rate") is not None:
                completion = float(si["completion_rate"])
                blanks_remaining = float(si.get("num_blanks_remaining") or 0.0)
                initial_blanks = float(si.get("initial_blank_count") or 0.0)
                correct_fills = float(si.get("correct_fills") or 0.0)
                outcome_target = float(si.get("outcome_success_correct_fills") or 0.0)
                outcome_target_completion = float(
                    si.get("outcome_target_completion_rate") or 0.0
                )
                break
        for si in reversed(episode_info_list):
            reason = si.get("terminal_reason")
            if reason is not None and reason != "already_done":
                terminal_reason = str(reason)
                break

        measured = [si for si in episode_info_list if si.get("move_optimal") is not None]
        denom = float(len(measured)) if measured else 0.0
        pre_exec_matches = 0
        legal_non_oracle = 0
        forced_available = 0
        mrv_actions = 0
        forced_oracle = 0
        mrv_oracle = 0
        mrv_min_candidate_sum = 0.0
        mrv_min_candidate_count = 0
        oracle_action_set_size_sum = 0.0
        oracle_action_set_size_count = 0
        for si in measured:
            pre_exec_match = si.get("pre_exec_oracle_match", si.get("move_optimal"))
            if bool(pre_exec_match):
                pre_exec_matches += 1
            if bool(si.get("legal_non_oracle")):
                legal_non_oracle += 1
            if bool(si.get("sudoku_forced_cell_available")):
                forced_available += 1
            if bool(si.get("sudoku_action_is_mrv_cell")):
                mrv_actions += 1
            tier = si.get("sudoku_oracle_tier")
            if tier == "forced":
                forced_oracle += 1
            elif tier == "mrv":
                mrv_oracle += 1
            min_candidates = si.get("sudoku_mrv_min_candidates")
            if min_candidates is not None:
                mrv_min_candidate_sum += float(min_candidates)
                mrv_min_candidate_count += 1
            oracle_set_size = si.get("oracle_action_set_size")
            if oracle_set_size is not None:
                oracle_action_set_size_sum += float(oracle_set_size)
                oracle_action_set_size_count += 1

        step_denom = float(len(episode_info_list)) if episode_info_list else 0.0
        parse_errors = sum(1 for si in episode_info_list if not bool(si.get("parse_ok", True)))
        illegal_actions = sum(1 for si in episode_info_list if bool(si.get("illegal_action", False)))
        blanks_remaining_rate = (blanks_remaining / initial_blanks) if initial_blanks > 0 else 0.0

        return {
            "env/completion_rate": completion,
            "env/num_blanks_remaining": blanks_remaining,
            "env/blanks_remaining_rate": blanks_remaining_rate,
            "env/sudoku_correct_fills": correct_fills,
            "env/sudoku_outcome_target": outcome_target,
            "env/sudoku_outcome_target_completion_rate": outcome_target_completion,
            "env/pre_exec_oracle_match_rate": (pre_exec_matches / denom) if denom else 0.0,
            "env/legal_non_oracle_rate": (legal_non_oracle / denom) if denom else 0.0,
            "env/sudoku_forced_cell_available_rate": (forced_available / denom) if denom else 0.0,
            "env/sudoku_mrv_action_rate": (mrv_actions / denom) if denom else 0.0,
            "env/sudoku_forced_oracle_rate": (forced_oracle / denom) if denom else 0.0,
            "env/sudoku_mrv_oracle_rate": (mrv_oracle / denom) if denom else 0.0,
            "env/sudoku_mrv_min_candidates_mean": (mrv_min_candidate_sum / mrv_min_candidate_count) if mrv_min_candidate_count else 0.0,
            "env/sudoku_oracle_action_set_size_mean": (oracle_action_set_size_sum / oracle_action_set_size_count) if oracle_action_set_size_count else 0.0,
            "env/parse_error_rate": (parse_errors / step_denom) if step_denom else 0.0,
            "env/illegal_action_rate": (illegal_actions / step_denom) if step_denom else 0.0,
            "env/terminal_complete_rate": 1.0 if terminal_reason == "complete" else 0.0,
            "env/terminal_outcome_target_rate": 1.0 if terminal_reason == "outcome_target" else 0.0,
            "env/terminal_timeout_rate": 1.0 if terminal_reason == "timeout" else 0.0,
            "env/terminal_wrong_digit_rate": 1.0 if terminal_reason == "wrong_digit" else 0.0,
            "env/terminal_invalid_action_rate": 1.0 if terminal_reason == "invalid_action" else 0.0,
            "env/terminal_out_of_range_rate": 1.0 if terminal_reason == "out_of_range" else 0.0,
            "env/terminal_cell_not_blank_rate": 1.0 if terminal_reason == "cell_not_blank" else 0.0,
        }
