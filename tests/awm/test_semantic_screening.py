import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.awm.data.pools import (
    _validate_semantic_partition_identity,
)
from agent_system.environments.env_package.awm.screening.semantic.evidence import (
    SOURCE_ARTIFACT_NAMES,
    ReplayDriftError,
    SourceCatalog,
    _validated_session_dir,
    require_replay_match,
    sha256_file,
    sqlite_diff,
    validate_evidence_packet,
)
from agent_system.environments.env_package.awm.screening.semantic.judgments import (
    JUDGMENT_PROTOCOL_VERSION,
    reviewer_consensus,
    validate_judgment,
)
from agent_system.environments.env_package.awm.screening.semantic.pipeline import (
    _derive_consensus_record,
    _rebuild_queue,
    next_review,
    select_success_controls,
    verify_consensus_ledger,
)


def _judgment(
    slot,
    verdict,
    relevance="not_applicable",
    confidence=0.95,
    cohort=None,
    *,
    evidence_sha256="evidence",
    evidence_refs=None,
):
    return {
        "protocol_version": JUDGMENT_PROTOCOL_VERSION,
        "task_id": "scenario:0",
        "review_slot": slot,
        "evidence_sha256": evidence_sha256,
        "verdict": verdict,
        "path_relevance": relevance,
        "confidence": confidence,
        "evidence_refs": list(evidence_refs or ["/fresh_replay/verify"]),
        "cohort_keys": list(cohort or []),
        "rationale": "Evidence supports this verdict.",
        "required_path_explanation": "The affected path is classified explicitly.",
    }


def _packet_fixture(root: Path, task_id: str = "scenario:0"):
    artifacts = {}
    for name in sorted(SOURCE_ARTIFACT_NAMES):
        path = root / "source_bundles" / name / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"name": name}
        if name == "tasks":
            value = {"scenario": "scenario", "tasks": ["perform the task"]}
        elif name == "code_verifier":
            value = {"scenario": "scenario", "task_idx": 0, "verification": {}}
        path.write_text(json.dumps(value), encoding="utf-8")
        artifacts[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
    packet = {
        "protocol_version": 2,
        "task_id": task_id,
        "scenario": "scenario",
        "task_idx": 0,
        "task": "perform the task",
        "screening_manifest_sha256": "screening",
        "screening_trial": {},
        "fresh_replay": {"validation": "matched", "verify": {"reward_type": "others"}},
        "source_artifacts": artifacts,
        "cohort_keys": {"environment_source:abc": "same environment"},
    }
    evidence_path = root / "evidence" / "scenario__0.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(packet), encoding="utf-8")
    plan = {
        "screening_manifest_sha256": "screening",
        "targets": [
            {
                "task_id": task_id,
                "basis": "policy_failure",
                "success_control": False,
            }
        ],
    }
    return plan, evidence_path, packet


def _write_judgment(root: Path, value: dict) -> Path:
    path = root / "judgments" / value["review_slot"] / f"{value['task_id'].replace(':', '__')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_dual_review_exclusion_requires_agreement_confidence_and_shared_cohort():
    key = "environment_source:abc"
    agreed = [
        validate_judgment(_judgment("A", "task_infeasible", cohort=[key])),
        validate_judgment(_judgment("B", "task_infeasible", cohort=[key])),
    ]
    consensus = reviewer_consensus(agreed)
    assert consensus.membership == "excluded"
    assert consensus.cohort_keys == (key,)

    low = [agreed[0], validate_judgment(_judgment("B", "task_infeasible", confidence=0.89, cohort=[key]))]
    assert reviewer_consensus(low).membership == "pending"
    disagreed = [agreed[0], validate_judgment(_judgment("B", "confirmed_policy_failure"))]
    assert reviewer_consensus(disagreed).membership == "pending"


def test_avoidable_environment_bug_is_included_but_required_bug_is_excluded():
    avoidable = [validate_judgment(_judgment(slot, "environment_semantic_bug", "avoidable")) for slot in ("A", "B")]
    assert reviewer_consensus(avoidable).membership == "included"
    key = "environment_source:abc"
    required = [validate_judgment(_judgment(slot, "environment_semantic_bug", "required", cohort=[key])) for slot in ("A", "B")]
    assert reviewer_consensus(required).membership == "excluded"


def test_judgment_rejects_unknown_fields_and_unbound_cohort():
    value = _judgment("A", "task_infeasible", cohort=["environment_source:abc"])
    value["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_judgment(value)
    value.pop("extra")
    with pytest.raises(ValueError, match="unknown cohort"):
        validate_judgment(value, allowed_cohort_keys=[])


def test_success_controls_are_deterministic_and_stratified():
    rows = [{"task_id": f"scenario:{index}", "native_prompt_tokens": 100 + index} for index in range(40)]
    trials = {row["task_id"]: {"result": {"decisions": 1 + (index % 15)}} for index, row in enumerate(rows)}
    first, strata = select_success_controls(rows, trials, fraction=0.1)
    second, _ = select_success_controls(list(reversed(rows)), trials, fraction=0.1)
    assert set(first) == set(second)
    assert len(first) >= 4
    assert set(first) <= set(strata)


def test_sqlite_diff_records_added_and_removed_rows(tmp_path):
    initial = tmp_path / "initial.db"
    final = tmp_path / "final.db"
    for path, values in ((initial, [(1, "old")]), (final, [(2, "new")])):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT)")
            connection.executemany("INSERT INTO records VALUES (?, ?)", values)
    diff = sqlite_diff(initial, final)
    assert diff["changed_tables"]["records"]["added"] == [[2, "new"]]
    assert diff["changed_tables"]["records"]["removed"] == [[1, "old"]]
    assert json.dumps(diff)


def test_sqlite_diff_preserves_duplicate_row_multiplicity(tmp_path):
    initial = tmp_path / "initial.db"
    final = tmp_path / "final.db"
    for path, count in ((initial, 1), (final, 2)):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (value TEXT)")
            connection.executemany("INSERT INTO records VALUES (?)", [("same",)] * count)
    diff = sqlite_diff(initial, final)
    assert diff["changed_tables"]["records"]["added"] == [["same"]]


def test_session_cleanup_guard_only_accepts_expected_tmp_prefix(tmp_path):
    safe = tmp_path / "openenv_awm_sample_abc"
    safe.mkdir()
    if tmp_path.resolve().is_relative_to("/tmp"):
        assert _validated_session_dir(str(safe), "sample") == safe.resolve()
    with pytest.raises(RuntimeError, match="unsafe"):
        _validated_session_dir("/var/tmp/openenv_awm_sample_abc", "sample")


@pytest.mark.parametrize(
    "component",
    [
        "canonical tool schema",
        "raw tool schema",
        "tool observation",
        "verifier observation",
    ],
)
def test_replay_drift_is_rejected_for_every_bound_component(component):
    with pytest.raises(ReplayDriftError, match=component):
        require_replay_match(component, expected="before", actual="after")
    require_replay_match(component, expected="same", actual="same")


def test_source_catalog_collapses_identical_duplicates_but_rejects_conflicts():
    assert SourceCatalog._unique([{"value": 1}, {"value": 1}], "verifier") == {"value": 1}
    with pytest.raises(RuntimeError, match="distinct values"):
        SourceCatalog._unique([{"value": 1}, {"value": 2}], "verifier")


def test_evidence_packet_binds_every_source_artifact(tmp_path):
    _, _, packet = _packet_fixture(tmp_path)
    validate_evidence_packet(
        packet,
        output_dir=tmp_path,
        task_id="scenario:0",
        screening_manifest_sha256="screening",
    )
    source = tmp_path / packet["source_artifacts"]["environment"]["path"]
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        validate_evidence_packet(packet, output_dir=tmp_path)


def test_invalid_or_stale_judgment_remains_actionable_in_queue(tmp_path):
    plan, evidence_path, _ = _packet_fixture(tmp_path)
    queue = _rebuild_queue(tmp_path, plan, persist=False, write_prompts=False)
    assert {item["status"] for item in queue} == {"ready"}

    path = tmp_path / "judgments" / "A" / "scenario__0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    queue = _rebuild_queue(tmp_path, plan, persist=False, write_prompts=False)
    assert next(item for item in queue if item["review_slot"] == "A")["status"] == "invalid"

    stale = _judgment(
        "A",
        "confirmed_policy_failure",
        evidence_sha256="stale",
    )
    path.write_text(json.dumps(stale), encoding="utf-8")
    queue = _rebuild_queue(tmp_path, plan, persist=False, write_prompts=False)
    assert next(item for item in queue if item["review_slot"] == "A")["status"] == "invalid"

    invalid_pointer = _judgment(
        "A",
        "confirmed_policy_failure",
        evidence_sha256=sha256_file(evidence_path),
        evidence_refs=["/missing"],
    )
    path.write_text(json.dumps(invalid_pointer), encoding="utf-8")
    queue = _rebuild_queue(tmp_path, plan, persist=False, write_prompts=False)
    assert next(item for item in queue if item["review_slot"] == "A")["status"] == "invalid"


def test_next_review_uses_slot_cursor_and_advances_after_valid_judgment(tmp_path, monkeypatch):
    plan, evidence_path, _ = _packet_fixture(tmp_path)
    _rebuild_queue(tmp_path, plan, write_prompts=False)
    monkeypatch.setattr(
        "agent_system.environments.env_package.awm.screening.semantic.pipeline._load_bound_plan",
        lambda *_args, **_kwargs: plan,
    )
    args = SimpleNamespace(output_dir=tmp_path, slot="A")
    prompt = next_review(args)
    assert "Reviewer slot: A" in prompt
    assert "Reviewer slot: B" not in prompt

    _write_judgment(
        tmp_path,
        _judgment(
            "A",
            "confirmed_policy_failure",
            evidence_sha256=sha256_file(evidence_path),
        ),
    )
    assert next_review(args) == ""
    cursor = json.loads((tmp_path / "review_cursor_A.json").read_text())
    assert cursor["offset"] == (tmp_path / "review_queue.jsonl").stat().st_size


def test_consensus_ledger_rejects_evidence_and_judgment_tampering(tmp_path):
    plan, evidence_path, packet = _packet_fixture(tmp_path)
    evidence_sha = sha256_file(evidence_path)
    judgments = {}
    for slot in ("A", "B"):
        value = _judgment(
            slot,
            "confirmed_policy_failure",
            evidence_sha256=evidence_sha,
        )
        judgments[slot] = value
        _write_judgment(tmp_path, value)
    record, _ = _derive_consensus_record(tmp_path, plan, "scenario:0")
    verify_consensus_ledger(tmp_path, plan=plan, records=[record])

    judgments["A"]["rationale"] = "Changed but still schema-valid evidence rationale."
    _write_judgment(tmp_path, judgments["A"])
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_consensus_ledger(tmp_path, plan=plan, records=[record])
    judgments["A"]["rationale"] = "Evidence supports this verdict."
    _write_judgment(tmp_path, judgments["A"])

    (tmp_path / "judgments" / "B" / "scenario__0.json").unlink()
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_consensus_ledger(tmp_path, plan=plan, records=[record])
    _write_judgment(tmp_path, judgments["B"])

    evidence_path.write_text(json.dumps({**packet, "task": "tampered"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_consensus_ledger(tmp_path, plan=plan, records=[record])


def test_consensus_ledger_rederives_membership(tmp_path):
    plan, evidence_path, _ = _packet_fixture(tmp_path)
    for slot in ("A", "B"):
        _write_judgment(
            tmp_path,
            _judgment(
                slot,
                "confirmed_policy_failure",
                evidence_sha256=sha256_file(evidence_path),
            ),
        )
    record, _ = _derive_consensus_record(tmp_path, plan, "scenario:0")
    record["membership"] = "excluded"
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_consensus_ledger(tmp_path, plan=plan, records=[record])


def test_semantic_partitions_require_exact_pinned_10000_ids_and_order():
    source_ids = [f"scenario:{index}" for index in range(10000)]
    names = ("included", "excluded", "pending", "out_of_context")
    partitions = {
        "included": source_ids[::2],
        "excluded": source_ids[1::2],
        "pending": [],
        "out_of_context": [],
    }
    _validate_semantic_partition_identity(source_ids, source_ids, partitions, names)

    changed_ids = [*source_ids[:-1], "different:9999"]
    with pytest.raises(RuntimeError, match="pinned all-task IDs mismatch"):
        _validate_semantic_partition_identity(source_ids, changed_ids, partitions, names)

    changed_partitions = {**partitions, "included": [*partitions["included"]]}
    changed_partitions["included"][-1] = "different:9998"
    with pytest.raises(RuntimeError, match="exactly cover"):
        _validate_semantic_partition_identity(source_ids, source_ids, changed_partitions, names)

    reordered = {**partitions, "included": list(reversed(partitions["included"]))}
    with pytest.raises(RuntimeError, match="partition order mismatch"):
        _validate_semantic_partition_identity(source_ids, source_ids, reordered, names)
