"""Read-only, offset-indexed cache import; clients own protocol validation."""

from __future__ import annotations

import json
from pathlib import Path

LEGACY_SINGLE_ACTION_PROMPT_HASH = "c31012bf8ebc959b5c5381207bb50c44597b952a850f331771471a80fc29d3b7"


def teacher_import_protocol_matches(record, current, allowed_prompt_hashes=()):
    """Explicit seed-label reuse, never a claim of equal sampling distributions.

    Only the known v13 single-action prompt can opt into the strict-JSON
    prompt upgrade. All decoding, model, endpoint, schema and state checks
    remain the caller's responsibility. Votes must be reparsed before use.
    """
    source = record.get("teacher_protocol_config")
    if source == current and record.get("protocol_version") in {14, 15}:
        return True
    if not isinstance(source, dict):
        return False
    old_hash = source.get("teacher_prompt_hash")
    if (
        record.get("protocol_version") != 13
        or old_hash != LEGACY_SINGLE_ACTION_PROMPT_HASH
        or old_hash not in allowed_prompt_hashes
        or current.get("teacher_prompt_revision") != "single_action_strict_json"
        or current.get("teacher_prompt_hash") != "9d0f51c36e7d197884b9acdab30754ed02f61e7e837cddd42300eb8a85e78667"
        or source.get("teacher_prompt_revision") != "single_action_v1"
        or record.get("teacher_prompt_hash") != old_hash
        or record.get("teacher_prompt_revision") != "single_action_v1"
        or ("teacher_validity_max_retries" in source and source["teacher_validity_max_retries"] != current["teacher_validity_max_retries"])
    ):
        return False
    # v13 did not implement validity retry. Missing/invalid votes are now
    # refilled using the current policy; already valid votes retain their origin.
    upgraded = {
        **source,
        "teacher_prompt_hash": current["teacher_prompt_hash"],
        "teacher_prompt_revision": current["teacher_prompt_revision"],
        "teacher_validity_max_retries": current["teacher_validity_max_retries"],
    }
    return upgraded == current


class TeacherCacheImport:
    """Index once without retaining large raw responses or modifying source files."""

    def __init__(self, paths=(), *, destination=None):
        self._index = {}
        self._source_settings = {}
        destination = Path(destination).resolve() if destination else None
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path == destination:
                raise ValueError("teacher cache import must not be its writable destination")
            if not path.is_file():
                raise FileNotFoundError(f"teacher cache import does not exist: {path}")
            config = path.parent.parent / "hydra" / ".hydra" / "config.yaml"
            if config.is_file():
                import yaml

                with config.open() as handle:
                    settings = yaml.safe_load(handle)
                self._source_settings[path] = (settings or {}).get("env", {}).get("awm", {}).get("oracle", {})
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(record, dict) and isinstance(record.get("state_fingerprint"), str):
                        context = json.dumps(record.get("progress_context"), sort_keys=True)
                        self._index.setdefault(record["state_fingerprint"], {})[(path, context)] = offset

    def records(self, fingerprint):
        for (path, _), offset in reversed(list(self._index.get(fingerprint, {}).items())):
            with path.open("rb") as handle:
                handle.seek(offset)
                record = json.loads(handle.readline())
            if record.get("state_fingerprint") == fingerprint:
                settings = self._source_settings.get(path, {})
                if "api_base" not in record and settings.get("model") == record.get("model") and settings.get("provider") == record.get("provider"):
                    # Historical AWM records omitted endpoint identity. The
                    # originating run's resolved Hydra config is evidence;
                    # without it the client must decline the import.
                    record["api_base"] = settings.get("api_base")
                yield record


def valid_vote_records(record, samples):
    votes = record.get("teacher_samples")
    if not isinstance(votes, list) or len(votes) > samples or record.get("valid_samples") != len(votes):
        return False
    indices = [vote.get("sample_index") for vote in votes if isinstance(vote, dict)]
    return len(indices) == len(votes) and all(type(i) is int and 0 <= i < samples for i in indices) and len(set(indices)) == len(indices)
