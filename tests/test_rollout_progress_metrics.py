import pytest

from agent_system.environments.teacher_reward import teacher_selection_diagnostics


def test_rollout_progress_metrics_count_selected_groups_once():
    group = [
        {
            "state_group_uid": "g1",
            "action_kind": "tool",
            "teacher_multiset": [{"kind": "tool", "name": "lookup", "arguments": {}}],
            "state_group_selected": index == 0,
            "no_progress_resample_triggered": True,
            "no_progress_resample_rounds": 1,
            "no_progress_resample_recovered": True,
            "no_progress_resample_still_collapsed": False,
            "pre_resample_unique_action_count": 1,
            "post_resample_unique_action_count": 3,
            "nonrepeat_alternative_available": index == 0,
            "nonrepeat_preference_applied": index == 0,
        }
        for index in range(4)
    ]

    metrics = teacher_selection_diagnostics([group], [[group[0]]])

    assert metrics["nonrepeat_argmax_available_rate"] == 1.0
    assert metrics["nonrepeat_commit_rate"] == 1.0
    assert metrics["no_progress_resample_trigger_count"] == 1.0
    assert metrics["no_progress_resample_trigger_rate"] == 1.0
    assert metrics["no_progress_resample_recovery_rate"] == 1.0
    assert metrics["no_progress_resample_still_collapsed_rate"] == 0.0
    assert metrics["no_progress_resample_extra_candidate_count"] == 4.0
    assert metrics["pre_resample_unique_action_count_mean"] == pytest.approx(1.0)
    assert metrics["post_resample_unique_action_count_mean"] == pytest.approx(3.0)
