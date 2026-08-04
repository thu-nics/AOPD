import os

from agent_system.environments.env_package.awm.evaluation import _fit_context, _model_artifact_identity


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
