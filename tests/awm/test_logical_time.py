from datetime import datetime, timezone

from examples.awm.logical_time import (
    FALLBACK_LOGICAL_TIME,
    freeze_temporal_source,
    freeze_temporal_structure,
    freeze_temporal_text,
    infer_logical_time,
)


def test_infers_snapshot_time_from_latest_explicit_record_timestamp():
    sample = {
        "tables": [
            {
                "table_name": "records",
                "insert_statements": [
                    "INSERT INTO records (id, created_at, updated_at) VALUES (1, '2026-03-01 08:00:00', '2026-03-02 09:05:01');",
                    "INSERT INTO records (id, created_at, updated_at) VALUES (2, datetime('now', '-1 day'), datetime('now'));",
                ],
            }
        ]
    }

    logical_time, source = infer_logical_time(sample)

    assert logical_time == datetime(2026, 3, 2, 9, 5, 2)
    assert source == "sample_created_or_updated_at"


def test_uses_pinned_fallback_when_sample_only_has_relative_time():
    sample = [{"insert_statements": ["INSERT INTO records (id, created_at) VALUES (1, datetime('now'));"]}]

    assert infer_logical_time(sample) == (
        FALLBACK_LOGICAL_TIME,
        "dataset_fallback",
    )


def test_freezes_python_datetime_and_date_calls_without_changing_types():
    source = """
import datetime
import datetime as dt
from datetime import date, datetime, timezone

values = (
    datetime.utcnow(),
    datetime.now(),
    datetime.now(timezone.utc),
    date.today(),
    dt.datetime.utcnow(),
    dt.date.today(),
)
"""
    logical_time = datetime(2026, 3, 2, 9, 5, 2)

    frozen = freeze_temporal_source(source, logical_time)
    namespace = {}
    exec(compile(frozen, "<frozen-clock-test>", "exec"), namespace)

    values = namespace["values"]
    assert values[:2] == (logical_time, logical_time)
    assert values[2] == logical_time.replace(tzinfo=timezone.utc)
    assert values[3].isoformat() == "2026-03-02"
    assert values[4] == logical_time
    assert values[5].isoformat() == "2026-03-02"


def test_freezes_sql_clock_inside_python_string_without_breaking_quotes():
    source = """
query = "UPDATE records SET updated_at=CURRENT_TIMESTAMP"
sentinels = ['now', 'current_timestamp']
"""
    logical_time = datetime(2025, 12, 31, 9, 5, 1)

    frozen = freeze_temporal_source(source, logical_time)
    namespace = {}
    exec(compile(frozen, "<frozen-sql-string-test>", "exec"), namespace)

    assert namespace["query"] == ("UPDATE records SET updated_at='2025-12-31 09:05:01'")
    assert namespace["sentinels"] == ["now", "current_timestamp"]


def test_source_rewrite_preserves_f_string_escaped_braces():
    source = 'pattern = fr"{3}:?\\d{{0,2}}"\nquery = "SELECT CURRENT_TIMESTAMP"\n'

    frozen = freeze_temporal_source(source, FALLBACK_LOGICAL_TIME)
    namespace = {}
    exec(compile(frozen, "<f-string-preservation-test>", "exec"), namespace)

    assert namespace["pattern"] == r"3:?\d{0,2}"
    assert namespace["query"] == "SELECT '2025-01-15 12:00:00'"


def test_freezes_sqlalchemy_server_clock_expression():
    source = "server_default = func.current_timestamp()\n"
    frozen = freeze_temporal_source(source, FALLBACK_LOGICAL_TIME)
    compile(frozen, "<sqlalchemy-clock-test>", "exec")
    assert frozen.strip() == "server_default = func.datetime('2025-01-15 12:00:00')"


def test_freezes_sqlite_relative_clock_and_schema_defaults_recursively():
    logical_time = datetime(2025, 1, 15, 12, 0, 0)
    payload = {
        "ddl": ("CREATE TABLE events (created_at DATETIME DEFAULT CURRENT_TIMESTAMP, created_date DATE DEFAULT CURRENT_DATE, created_time TIME DEFAULT CURRENT_TIME)"),
        "statements": [
            "INSERT INTO events(created_at) VALUES(datetime('now', '+2 days'))",
            "SELECT strftime('%Y-%m', 'now'), date(\"now\"), julianday('now')",
        ],
    }

    frozen = freeze_temporal_structure(payload, logical_time)

    assert "DEFAULT '2025-01-15 12:00:00'" in frozen["ddl"]
    assert "DEFAULT '2025-01-15'" in frozen["ddl"]
    assert "DEFAULT '12:00:00'" in frozen["ddl"]
    assert "datetime('2025-01-15 12:00:00', '+2 days')" in frozen["statements"][0]
    assert "strftime('%Y-%m', '2025-01-15 12:00:00')" in frozen["statements"][1]
    assert 'date("2025-01-15 12:00:00")' in frozen["statements"][1]
    assert "julianday('2025-01-15 12:00:00')" in frozen["statements"][1]


def test_does_not_rewrite_ordinary_now_text():
    value = "Tell the user: now is a good time; status=CURRENTLY_OPEN"

    assert freeze_temporal_text(value, FALLBACK_LOGICAL_TIME) == value
