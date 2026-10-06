import csv
from pathlib import Path

import pytest

from server.db import DATA_DIR, TABLES, get_connection, init_db, parse_date

CSV_FILES = {
    "employees": "employees.csv",
    "leave_types": "leave_types.csv",
    "leave_entitlements": "leave_entitlements.csv",
    "public_holidays": "public_holidays.csv",
    "leave_requests": "leave_requests.csv",
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _csv_row_count(filename: str) -> int:
    with (DATA_DIR / filename).open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _table_counts(db_path: Path) -> dict[str, int]:
    conn = get_connection(db_path)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()


def test_row_counts_match_csvs(db_path: Path) -> None:
    counts = _table_counts(db_path)
    for table, filename in CSV_FILES.items():
        assert counts[table] == _csv_row_count(filename), table
    assert counts["leave_proposals"] == 0


def test_parse_date_accepts_us_and_iso() -> None:
    assert parse_date("3/4/2019") == "2019-03-04"
    assert parse_date("11/30/2026") == "2026-11-30"
    assert parse_date("2019-03-04") == "2019-03-04"


@pytest.mark.parametrize("value", ["2019/03/04", "04.03.2019", "13/1/2019", "2019-3-4", "x"])
def test_parse_date_rejects_other_formats(value: str) -> None:
    with pytest.raises(ValueError):
        parse_date(value)


def test_init_db_twice_does_not_duplicate(db_path: Path) -> None:
    before = _table_counts(db_path)
    init_db(db_path)
    assert _table_counts(db_path) == before


def test_e1004_probation_end_date(db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT probation_end_date FROM employees WHERE employee_id = 'E1004'"
        ).fetchone()
    finally:
        conn.close()
    assert row["probation_end_date"] == "2026-11-30"


def test_managers_linked_and_foreign_keys_valid(db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        manager = conn.execute(
            "SELECT manager_id FROM employees WHERE employee_id = 'E1001'"
        ).fetchone()["manager_id"]
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()
    assert manager == "E1010"
    assert violations == []


def test_no_crlf_or_bom_in_stored_text(db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        for table in TABLES:
            for row in conn.execute(f"SELECT * FROM {table}"):
                for value in row:
                    if isinstance(value, str):
                        assert "\r" not in value and "﻿" not in value, (table, value)
    finally:
        conn.close()
