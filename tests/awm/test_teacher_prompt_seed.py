import copy
import json

import pytest

from agent_system.environments.env_package.awm.runtime.oracle import (
    ORACLE_PROTOCOL_VERSION,
    DeepSeekAWMOracleClient,
)
from agent_system.environments.teacher_cache_import import (
    LEGACY_SINGLE_ACTION_PROMPT_HASH,
    TeacherCacheImport,
    teacher_import_protocol_matches,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "parent_id": {"type": ["integer", "null"]},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    }
]
CHAT = [{"role": "user", "content": "Update record 1."}]
ALLOW = [LEGACY_SINGLE_ACTION_PROMPT_HASH]


def response(_):
    return {"model": "deepseek-v4-flash", "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"function": {"name": "write", "arguments": '{"id":1,"parent_id":null}'}}]}}]}


def make_seed(path):
    client = DeepSeekAWMOracleClient(cache_path=str(path), request_fn=response)
    client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
    r = json.loads(path.read_text())
    r["protocol_version"] = 13
    r["teacher_prompt_revision"] = "single_action_v1"
    r["teacher_prompt_hash"] = LEGACY_SINGLE_ACTION_PROMPT_HASH
    r["teacher_protocol_config"].update(teacher_prompt_revision=r["teacher_prompt_revision"], teacher_prompt_hash=r["teacher_prompt_hash"])
    r["teacher_protocol_config"].pop("teacher_validity_max_retries")
    # Simulate the old lossy normalization; import must use raw arguments.
    for vote in r["teacher_samples"]:
        vote["action"]["arguments"].pop("parent_id")
    path.write_text(json.dumps(r) + "\n")
    return r


def test_old_prompt_seed_preserves_origin_votes_and_resume(tmp_path):
    source, target = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    make_seed(source)
    before = source.read_bytes()
    client = DeepSeekAWMOracleClient(cache_path=str(target), teacher_cache_import_paths=[str(source)], teacher_cache_import_prompt_hashes=ALLOW, request_fn=lambda _: pytest.fail("must reuse seed"))
    votes = client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
    assert len(votes) == 3
    assert all(v["action"]["arguments"] == {"id": 1, "parent_id": None} for v in votes)
    assert all(v["generation_provenance"]["teacher_prompt_hash"] == ALLOW[0] for v in votes)
    assert source.read_bytes() == before
    assert client.stats()["teacher_cache_hit_rate"] == 1.0
    assert client.stats()["teacher_cache_misses"] == 0
    assert json.loads(target.read_text())["protocol_version"] == ORACLE_PROTOCOL_VERSION
    resumed = DeepSeekAWMOracleClient(cache_path=str(target), teacher_cache_import_prompt_hashes=ALLOW, request_fn=lambda _: pytest.fail("must reuse revalidated cache"))
    assert resumed.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS) == votes
    assert resumed.stats()["teacher_cache_hits"] == 1
    # A migration flag cannot be silently lost on resume or on another import.
    for kwargs in ({"teacher_cache_import_paths": [str(target)]}, {"cache_path": str(target)}):
        strict = DeepSeekAWMOracleClient(**kwargs, request_fn=response)
        strict.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
        assert strict.stats()["teacher_requests"] == 3


@pytest.mark.parametrize("change", ["no_opt_in", "hash", "revision", "endpoint", "model", "schema", "decoding", "fallback"])
def test_seed_rejects_unapproved_protocol_changes(tmp_path, change):
    source = tmp_path / "source.jsonl"
    r = make_seed(source)
    if change == "hash":
        r["teacher_protocol_config"]["teacher_prompt_hash"] = "f" * 64
    elif change == "revision":
        r["teacher_prompt_revision"] = "other"
    elif change == "endpoint":
        r["api_base"] = "https://other.invalid/v1"
    elif change == "model":
        r["model"] = "other-model"
    elif change == "schema":
        r["native_tool_schema_hash"] = "different"
    elif change == "decoding":
        r["decoding_config"]["max_tokens"] += 1
    elif change == "fallback":
        r["teacher_protocol_config"]["multi_call_fallback_enabled"] = False
    source.write_text(json.dumps(r) + "\n")
    client = DeepSeekAWMOracleClient(teacher_cache_import_paths=[str(source)], teacher_cache_import_prompt_hashes=[] if change == "no_opt_in" else ALLOW, request_fn=response)
    client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
    assert client.stats()["teacher_requests"] == 3


@pytest.mark.parametrize("bad_vote", ["schema_invalid", "truncated", "unknown_identity"])
def test_seed_refills_only_bad_vote_with_current_prompt(tmp_path, bad_vote):
    source = tmp_path / "source.jsonl"
    r = make_seed(source)
    v = r["teacher_samples"][1]
    if bad_vote == "schema_invalid":
        v["raw_tool_calls"][0]["function"]["arguments"] = '{"id":"not an integer"}'
    elif bad_vote == "truncated":
        v.update(raw_tool_calls=[], raw_content="Incomplete", finish_reason="length")
    else:
        v["provider_identity"]["model"] = "different-model"
    source.write_text(json.dumps(r) + "\n")
    client = DeepSeekAWMOracleClient(teacher_cache_import_paths=[str(source)], teacher_cache_import_prompt_hashes=ALLOW, request_fn=response)
    votes = client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
    assert len(votes) == 3 and [v["sample_index"] for v in votes] == [0, 1, 2]
    assert client.stats()["teacher_requests"] == 1
    assert client.stats()["teacher_cache_imported_votes"] == 2
    assert client.stats()["teacher_cache_partial_hits"] == 1
    assert client.stats()["teacher_cache_refill_attempts"] == 1
    assert "generation_provenance" not in votes[1]


def test_import_index_preserves_progress_variants(tmp_path):
    source = tmp_path / "source.jsonl"
    first = make_seed(source)
    second = copy.deepcopy(first)
    second["progress_context"] = {"multi_call_fallback_eligible": True, "previous_canonical_action": "previous"}
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    records = list(TeacherCacheImport([source]).records("state"))
    assert len(records) == 2
    client = DeepSeekAWMOracleClient(teacher_cache_import_paths=[str(source)], teacher_cache_import_prompt_hashes=ALLOW, request_fn=lambda _: pytest.fail("must find non-fallback variant"))
    assert len(client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)) == 3


def test_prompt_upgrade_cannot_silently_accept_a_future_prompt(tmp_path):
    record = make_seed(tmp_path / "source.jsonl")
    current = DeepSeekAWMOracleClient(request_fn=response)._teacher_protocol_config()
    current["teacher_prompt_hash"] = "0" * 64
    assert not teacher_import_protocol_matches(record, current, ALLOW)


def test_legacy_votes_can_be_refilled_with_observed_response_alias(tmp_path):
    source, target = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    record = make_seed(source)
    record["teacher_samples"] = record["teacher_samples"][:2]
    record["valid_samples"] = 2
    source.write_text(json.dumps(record) + "\n")
    before = source.read_bytes()
    client = DeepSeekAWMOracleClient(cache_path=str(target), teacher_cache_import_paths=[str(source)], teacher_cache_import_prompt_hashes=ALLOW, request_fn=lambda p: {**response(p), "model": "deepseek-flash"})
    votes = client.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS)
    assert [v["provider_identity"]["model"] for v in votes] == ["deepseek-v4-flash", "deepseek-v4-flash", "deepseek-flash"]
    assert client.stats()["teacher_requests"] == 1
    assert client.stats()["teacher_cache_imported_votes"] == 2
    assert all("generation_provenance" in v for v in votes[:2])
    assert "generation_provenance" not in votes[2]
    assert source.read_bytes() == before
    reloaded = DeepSeekAWMOracleClient(cache_path=str(target), teacher_cache_import_prompt_hashes=ALLOW, request_fn=lambda _: pytest.fail("must reuse all three votes"))
    assert reloaded.sample_multiset(state_fingerprint="state", messages=CHAT, tools=TOOLS) == votes
