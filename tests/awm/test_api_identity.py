import json

import pytest

from agent_system.environments.env_package.awm.runtime.api_identity import (
    checked_provider_identity,
    deepseek_identity_preflight,
    response_model_matches,
)
from agent_system.environments.env_package.awm.runtime.oracle import DeepSeekAWMOracleClient

REQUESTED = "deepseek-v4-flash"
RETURNED = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com"


@pytest.mark.parametrize("path", ["", "/", "/v1", "/v1/", "/chat/completions", "/v1/chat/completions"])
def test_observed_alias_is_accepted_on_official_endpoint(path):
    assert response_model_matches(provider="deepseek", api_base=ENDPOINT + path, requested_model=REQUESTED, returned_model=RETURNED)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.deepseek.com",
        "https://api.deepseek.com:444",
        "https://gateway.invalid/v1",
        "https://api.deepseek.com.evil.invalid",
        "https://user@api.deepseek.com",
        "https://api.deepseek.com/?route=other",
        "https://api.deepseek.com/#other",
        "https://api.deepseek.com/other",
        "https://api.deepseek.com:bad",
    ],
)
def test_alias_does_not_relax_other_endpoints(endpoint):
    assert not response_model_matches(provider="deepseek", api_base=endpoint, requested_model=REQUESTED, returned_model=RETURNED)


@pytest.mark.parametrize(
    "provider,requested,returned",
    [
        ("dashscope", REQUESTED, RETURNED),
        ("zai", REQUESTED, RETURNED),
        ("deepseek", RETURNED, REQUESTED),
        ("deepseek", REQUESTED, "deepseek-v4-pro"),
        ("deepseek", REQUESTED, "deepseek-flash-other"),
        ("deepseek", "", ""),
        ("deepseek", REQUESTED, None),
    ],
)
def test_no_other_name_mapping(provider, requested, returned):
    assert not response_model_matches(provider=provider, api_base=ENDPOINT, requested_model=requested, returned_model=returned)


def test_exact_identity_still_works_for_other_providers_and_endpoints():
    assert response_model_matches(provider="zai", api_base="https://gateway.invalid", requested_model="glm-5.3-flash", returned_model="glm-5.3-flash")


def test_keeps_actual_name_and_rejects_conflicting_requested_name():
    identity = checked_provider_identity({"model": RETURNED, "system_fingerprint": "fp-a"}, provider="deepseek", api_base=ENDPOINT, requested_model=REQUESTED, role="teacher")
    assert identity == {"model": RETURNED, "requested_model": REQUESTED, "system_fingerprint": "fp-a"}
    assert checked_provider_identity(identity, provider="deepseek", api_base=ENDPOINT, requested_model=REQUESTED, role="teacher") == identity
    with pytest.raises(RuntimeError, match="returned model"):
        checked_provider_identity({**identity, "requested_model": RETURNED}, provider="deepseek", api_base=ENDPOINT, requested_model=REQUESTED, role="teacher")


def _reply(content, model=RETURNED):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def __init__(self, reply=None):
        self.calls = []
        self.reply = _reply("private-generated-text") if reply is None else reply
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _Response({"data": [{"id": REQUESTED}, {"id": RETURNED}]})

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _Response(self.reply)


def test_preflight_checks_both_modes_and_keeps_canonical_requests():
    session = _Session()
    report = deepseek_identity_preflight(api_base=ENDPOINT, model=REQUESTED, api_key="test-secret", session=session)
    assert report["status"] == "passed"
    assert report["request_name_advertised"] is True
    assert session.trust_env is False
    assert [(method, url) for method, url, _ in session.calls] == [("GET", ENDPOINT + "/models"), ("POST", ENDPOINT + "/chat/completions"), ("POST", ENDPOINT + "/chat/completions")]
    payloads = [kwargs["json"] for method, _, kwargs in session.calls if method == "POST"]
    assert [p["thinking"] for p in payloads] == [{"type": "disabled"}, {"type": "enabled"}]
    assert payloads[1]["reasoning_effort"] == "max"
    assert all(p["model"] == REQUESTED for p in payloads)
    assert all(c["model"] == RETURNED and c["requested_model"] == REQUESTED and c["response_name_alias"] for c in report["checks"])
    assert "test-secret" not in json.dumps(report)
    assert "private-generated-text" not in json.dumps(report)


@pytest.mark.parametrize("reply", [_reply("Done", "wrong-model"), {"model": RETURNED, "choices": []}, {"model": RETURNED, "choices": [None]}, []])
def test_preflight_fails_unknown_identity_or_malformed_reply(reply):
    with pytest.raises(RuntimeError):
        deepseek_identity_preflight(api_base=ENDPOINT, model=REQUESTED, api_key="test-secret", session=_Session(reply))


def test_teacher_and_matcher_alias_cache_survives_restart(tmp_path):
    payloads = []

    def request(payload):
        payloads.append(payload)
        return _reply("Done" if "tools" in payload else '{"equivalent":true}')

    paths = {"cache_path": str(tmp_path / "teacher.jsonl"), "matcher_cache_path": str(tmp_path / "matcher.jsonl")}
    client = DeepSeekAWMOracleClient(**paths, request_fn=request)
    kwargs = {"state_fingerprint": "state", "messages": [{"role": "user", "content": "Complete the task."}], "tools": []}
    votes = client.sample_multiset(**kwargs)
    assert len(votes) == 3
    assert client.match_message_pairs(["Done"], ["Completed"])["counts"] == [1]
    assert len(payloads) == 4
    assert all(p["model"] == REQUESTED for p in payloads)
    assert all(v["provider_identity"]["model"] == RETURNED for v in votes)
    assert json.loads((tmp_path / "teacher.jsonl").read_text())["model"] == REQUESTED
    assert json.loads((tmp_path / "matcher.jsonl").read_text())["provider_identity"]["model"] == RETURNED

    def no_requests(_):
        pytest.fail("valid persisted alias response must remain cache-first")

    reloaded = DeepSeekAWMOracleClient(**paths, request_fn=no_requests)
    assert reloaded.sample_multiset(**kwargs) == votes
    assert reloaded.match_message_pairs(["Done"], ["Completed"])["counts"] == [1]
    assert reloaded.stats()["teacher_cache_hits"] == 1
    imported = DeepSeekAWMOracleClient(teacher_cache_import_paths=[paths["cache_path"]], request_fn=no_requests)
    assert len(imported.sample_multiset(**kwargs)) == 3
    assert imported.stats()["teacher_cache_imported_votes"] == 3


@pytest.mark.parametrize("prefix", ["matcher", "runtime_judge", "envscaler_runtime_judge"])
def test_each_auxiliary_uses_own_provider_and_endpoint(prefix):
    client = DeepSeekAWMOracleClient(
        provider="zai",
        model="glm-5.3-flash",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        matcher_provider="deepseek",
        matcher_model=REQUESTED,
        matcher_api_base=ENDPOINT,
        runtime_judge_provider="deepseek",
        runtime_judge_model=REQUESTED,
        runtime_judge_api_base=ENDPOINT,
        request_fn=lambda _: _reply("unused"),
    )
    assert client._accept_provider_identity({"model": RETURNED}, prefix=prefix)["model"] == RETURNED
    with pytest.raises(RuntimeError, match="returned model"):
        client._accept_provider_identity({"model": RETURNED}, prefix="teacher")
