import os
from pathlib import Path

import pytest

from agent_system.environments.env_package.awm.evaluation.native import (
    DEFAULT_VERIFIER_MODE,
    _fit_context,
    _model_artifact_identity,
    _selection_task_ids,
    _verifier_reset_kwargs,
    _verifier_summary,
)


class _Tokenizer:
    def __init__(self):
        self.kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return list(range(len(messages)))


def test_eval_budget_uses_same_thinking_template_as_generation():
    tokenizer = _Tokenizer()
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

    assert _fit_context(tokenizer, chat, tools) == chat
    assert tokenizer.kwargs[-1]["enable_thinking"] is True


def test_eval_history_window_defaults_to_six_and_is_configurable():
    tokenizer = _Tokenizer()
    chat = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *[{"role": "assistant", "content": f"action-{index}"} for index in range(8)],
    ]

    assert _fit_context(tokenizer, chat, []) == [*chat[:2], *chat[-6:]]
    assert _fit_context(tokenizer, chat, [], history_window=2) == [
        *chat[:2],
        *chat[-2:],
    ]
    assert _fit_context(tokenizer, chat, [], history_window=0) == chat[:2]


def test_local_checkpoint_identity_hashes_weight_content(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    first = _model_artifact_identity(str(tmp_path))
    stat = weight.stat()
    weight.write_bytes(b"other")
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = _model_artifact_identity(str(tmp_path))

    assert first["artifact_manifest_sha256"] != second["artifact_manifest_sha256"]


def test_native_eval_defaults_to_sql_and_passes_judge_credentials_only_to_reset():
    assert DEFAULT_VERIFIER_MODE == "sql"
    assert _verifier_reset_kwargs(
        verifier_mode="sql",
        judge_api_base="https://api.deepseek.com",
        judge_api_key="secret",
        judge_model="deepseek-v4-flash",
    ) == {
        "llm_base_url": "https://api.deepseek.com",
        "llm_api_key": "secret",
        "llm_model": "deepseek-v4-flash",
    }
    assert (
        _verifier_reset_kwargs(
            verifier_mode="code",
            judge_api_base=None,
            judge_api_key=None,
            judge_model=None,
        )
        == {}
    )
    with pytest.raises(ValueError, match="SQL verification requires"):
        _verifier_reset_kwargs(
            verifier_mode="sql",
            judge_api_base="https://api.deepseek.com",
            judge_api_key=None,
            judge_model="deepseek-v4-flash",
        )


def test_native_eval_reads_current_healthy_pool_selection():
    assert _selection_task_ids(
        {
            "kind": "awm_healthy_task_pool",
            "training_pool_task_ids": ["scenario:0", "scenario:1"],
        }
    ) == ["scenario:0", "scenario:1"]
    assert _selection_task_ids(
        {
            "kind": "awm_training_pool_slice",
            "task_ids": ["scenario:2"],
        }
    ) == ["scenario:2"]


def test_native_eval_reports_terminal_judge_coverage_and_valid_success_rate():
    summary = _verifier_summary(
        [
            {"reward_type": "complete"},
            {"reward_type": "incomplete"},
            {"reward_type": "agent_error"},
            {"reward_type": "server_error"},
            {"reward_type": "judge_error"},
        ]
    )

    assert summary == {
        "verifier_label_counts": {
            "agent_error": 1,
            "complete": 1,
            "incomplete": 1,
            "judge_error": 1,
            "server_error": 1,
        },
        "terminal_judge_valid_tasks": 3,
        "terminal_judge_coverage": 0.6,
        "success_rate_valid": 1 / 3,
    }


def test_native_eval_launcher_defaults_to_sql_and_forwards_judge_identity():
    launcher = (Path(__file__).parents[2] / "examples" / "awm" / "eval" / "run_eval.sh").read_text(encoding="utf-8")

    assert 'VERIFIER_MODE="${VERIFIER_MODE:-sql}"' in launcher
    assert 'JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-DEEPSEEK_API_KEY}"' in launcher
    assert '--expected-terminal-model "$JUDGE_MODEL"' in launcher
    assert '--verifier-mode "$VERIFIER_MODE"' in launcher
    assert '--judge-api-key-env "$JUDGE_API_KEY_ENV"' in launcher
