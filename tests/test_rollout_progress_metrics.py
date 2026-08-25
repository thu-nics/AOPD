import pytest

from agent_system.environments.teacher_reward import teacher_selection_diagnostics


def test_rollout_progress_metrics_count_selected_groups_once():
    group = [
        {
            "state_group_uid": "g1",
            "action_kind": "tool",
            "teacher_multiset": [{"kind": "tool", "name": "lookup", "arguments": {}}],
            "state_group_selected": index == 0,
            "prospective_no_progress_repeat": index == 0,
            "repeat_reward_capped": index == 0,
            "selection_score": 0.0 if index == 0 else -1.0,
            "teacher_frequency": 1 if index == 0 else 0,
            "nonrepeat_alternative_available": index == 0,
            "nonrepeat_preference_applied": index == 0,
        }
        for index in range(4)
    ]

    metrics = teacher_selection_diagnostics([group], [[group[0]]])

    assert metrics["nonrepeat_argmax_available_rate"] == 1.0
    assert metrics["nonrepeat_commit_rate"] == 1.0
    assert metrics["repeat_reward_capped_candidate_rate"] == pytest.approx(0.25)
    assert metrics["repeat_reward_capped_group_rate"] == 1.0
    assert metrics["no_progress_repeat_candidate_teacher_match_rate"] == 1.0
    assert metrics["top_reward_no_progress_repeat_group_rate"] == 1.0
    assert metrics["top_reward_no_progress_repeat_teacher_match_rate"] == 1.0
