import sqlite3

import pytest

from server.balance import get_balance


def test_e1001_annual(conn: sqlite3.Connection) -> None:
    # 25 + 3 carried over - 15 approved (requests 1, 4) - 3 pending (request 5);
    # cancelled request 3 is ignored.
    balance = get_balance(conn, "E1001", 2026, "ANNUAL")
    assert balance.entitled_days == 25
    assert balance.carried_over_days == 3
    assert balance.approved_days == 15
    assert balance.pending_days == 3
    assert balance.available_days == 10


def test_e1001_sick(conn: sqlite3.Connection) -> None:
    balance = get_balance(conn, "E1001", 2026, "SICK")
    assert balance.approved_days == 2
    assert balance.available_days == 8


def test_rejected_requests_are_ignored(conn: sqlite3.Connection) -> None:
    # E1002: 24 - (5 + 15) approved; rejected request 8 (5 days) does not count.
    balance = get_balance(conn, "E1002", 2026, "ANNUAL")
    assert balance.approved_days == 20
    assert balance.pending_days == 0
    assert balance.available_days == 4


def test_no_requests_gives_full_entitlement(conn: sqlite3.Connection) -> None:
    assert get_balance(conn, "E1004", 2026, "ANNUAL").available_days == 8


def test_negative_sick_balance_is_capped_at_zero(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days, status, created_at, created_via)
        VALUES ('E1004', 'SICK', '2026-09-07', '2026-09-18', 12, 'approved',
                '2026-09-07T09:00:00', 'portal')
        """
    )
    balance = get_balance(conn, "E1004", 2026, "SICK")
    assert balance.approved_days == 12
    assert balance.available_days == 0


def test_requests_from_other_years_are_ignored(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days, status, created_at, created_via)
        VALUES ('E1004', 'ANNUAL', '2027-01-04', '2027-01-05', 2, 'pending',
                '2026-12-01T09:00:00', 'portal')
        """
    )
    assert get_balance(conn, "E1004", 2026, "ANNUAL").available_days == 8


@pytest.mark.parametrize("leave_type", ["BEREAVEMENT", "PARENTAL"])
def test_types_without_annual_balance_raise(conn: sqlite3.Connection, leave_type: str) -> None:
    with pytest.raises(LookupError):
        get_balance(conn, "E1001", 2026, leave_type)
