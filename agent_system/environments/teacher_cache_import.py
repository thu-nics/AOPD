"""Read-only, offset-indexed cache import; clients own protocol validation."""

from __future__ import annotations

import json
from pathlib import Path


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
                        self._index.setdefault(record["state_fingerprint"], {})[path] = offset

    def records(self, fingerprint):
        for path, offset in reversed(list(self._index.get(fingerprint, {}).items())):
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
