import sqlite3
from datetime import date

import pytest

from server.rules import REDIRECT, REJECT, ValidationResult, validate_request

TODAY = date(2026, 10, 19)  # APP_TODAY, a Monday


def check(
    conn: sqlite3.Connection,
    employee_id: str,
    leave_type: str,
    start: str,
    end: str,
    **kwargs: object,
) -> ValidationResult:
    return validate_request(
        conn,
        employee_id=employee_id,
        leave_type=leave_type,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        today=TODAY,
        **kwargs,  # type: ignore[arg-type]
    )


def only(result: ValidationResult, code: str, action: str) -> None:
    """Assert the result has exactly one violation with this code and action."""
    assert result.codes == [code], result.violations
    assert result.violations[0].action == action


def add_request(
    conn: sqlite3.Connection,
    employee_id: str,
    start: str,
    end: str,
    days: int,
    leave_type: str = "ANNUAL",
    status: str = "pending",
) -> None:
    conn.execute(
        """
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days, status, created_at, created_via)
        VALUES (?, ?, ?, ?, ?, ?, '2026-10-01T09:00:00', 'portal')
        """,
        (employee_id, leave_type, start, end, days, status),
    )


# --- valid request ---------------------------------------------------------


def test_valid_annual_request_passes_and_returns_days(conn: sqlite3.Connection) -> None:
    result = check(conn, "E1003", "ANNUAL", "2026-10-27", "2026-10-29")
    assert result.ok, result.violations
    assert result.days == 3


# --- blocking checks (stop at the first failure) ---------------------------


def test_unknown_employee(conn: sqlite3.Connection) -> None:
    only(check(conn, "E9999", "ANNUAL", "2026-11-02", "2026-11-03"), "UNKNOWN_EMPLOYEE", REJECT)


def test_unknown_leave_type(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1001", "VACATION", "2026-11-02", "2026-11-03"), "UNKNOWN_LEAVE_TYPE", REJECT)


@pytest.mark.parametrize(
    ("leave_type", "article"), [("BEREAVEMENT", "8.3"), ("STUDY", "9.3"), ("PARENTAL", "10.2")]
)
def test_unsupported_types_are_redirected(
    conn: sqlite3.Connection, leave_type: str, article: str
) -> None:
    result = check(conn, "E1001", leave_type, "2026-11-02", "2026-11-03")
    only(result, "LEAVE_TYPE_NOT_SUPPORTED", REDIRECT)
    assert result.violations[0].article == article


def test_start_after_end(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1003", "ANNUAL", "2026-11-05", "2026-11-02"), "DATES_INVALID", REJECT)


def test_dates_in_next_year(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1003", "ANNUAL", "2027-02-01", "2027-02-03"), "DATES_NEXT_YEAR", REJECT)


def test_dates_crossing_into_next_year(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1003", "ANNUAL", "2026-12-28", "2027-01-08"), "DATES_CROSS_YEAR", REJECT)


def test_dates_in_past_year(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1003", "SICK", "2025-12-01", "2025-12-02"), "DATES_PAST_YEAR", REJECT)


def test_weekend_only_annual_is_zero_days(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1003", "ANNUAL", "2026-10-31", "2026-11-01"), "ZERO_DAYS", REJECT)


def test_weekend_unpaid_counts_calendar_days(conn: sqlite3.Connection) -> None:
    result = check(conn, "E1003", "UNPAID", "2026-11-07", "2026-11-08", comment="ოჯახური მიზეზი")
    assert result.ok, result.violations
    assert result.days == 2


# --- probation (4.3) -------------------------------------------------------


def test_annual_during_probation_is_rejected(conn: sqlite3.Connection) -> None:
    # E1004's probation ends 2026-11-30 (inclusive).
    only(check(conn, "E1004", "ANNUAL", "2026-11-02", "2026-11-04"), "PROBATION", REJECT)


def test_annual_ending_on_last_probation_day_is_rejected(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1004", "ANNUAL", "2026-11-30", "2026-11-30"), "PROBATION", REJECT)


def test_annual_after_probation_is_allowed(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1004", "ANNUAL", "2026-12-01", "2026-12-03").ok


def test_probation_does_not_block_sick(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1004", "SICK", "2026-10-19", "2026-10-20").ok


# --- notice period (4.4, 7.2) ----------------------------------------------


def test_three_days_with_three_days_notice_is_rejected(conn: sqlite3.Connection) -> None:
    # Working days strictly between Mon 10-19 and Fri 10-23: 20, 21, 22.
    result = check(conn, "E1003", "ANNUAL", "2026-10-23", "2026-10-27")
    only(result, "NOTICE_PERIOD", REJECT)
    assert "2026-10-27" in (result.violations[0].alternative or "")


def test_policy_example_monday_to_next_tuesday_passes(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1003", "ANNUAL", "2026-10-27", "2026-10-29").ok


def test_six_days_needs_fifteen_days_notice(conn: sqlite3.Connection) -> None:
    # Mon 11-02 .. Mon 11-09 is 6 days; only 9 working days of notice.
    only(check(conn, "E1003", "ANNUAL", "2026-11-02", "2026-11-09"), "NOTICE_PERIOD", REJECT)


def test_unpaid_needs_ten_days_notice(conn: sqlite3.Connection) -> None:
    result = check(conn, "E1003", "UNPAID", "2026-10-28", "2026-10-30", comment="პირადი საქმე")
    only(result, "NOTICE_PERIOD", REJECT)
    assert result.violations[0].article == "7.2"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-10-16", "2026-10-16"),  # Friday before today
        ("2026-10-16", "2026-10-20"),  # spans today
        ("2026-10-19", "2026-10-19"),  # today itself
        ("2026-01-05", "2026-01-06"),  # early in the current year
    ],
)
def test_annual_starting_in_the_past_is_a_clean_notice_error(
    conn: sqlite3.Connection, start: str, end: str
) -> None:
    result = check(conn, "E1001", "ANNUAL", start, end)
    only(result, "NOTICE_PERIOD", REJECT)
    assert "2026-10-27" in (result.violations[0].alternative or "")


def test_unpaid_starting_in_the_past_is_a_clean_notice_error(conn: sqlite3.Connection) -> None:
    result = check(conn, "E1003", "UNPAID", "2026-10-16", "2026-10-16", comment="პირადი საქმე")
    only(result, "NOTICE_PERIOD", REJECT)


def test_notice_does_not_apply_to_sick(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1003", "SICK", "2026-10-20", "2026-10-21", known_in_advance=True).ok


# --- sick timing (6.2) -----------------------------------------------------


def test_sick_two_working_days_late_is_allowed(conn: sqlite3.Connection) -> None:
    # Thu 10-15: counted days are Fri 10-16 and Mon 10-19.
    assert check(conn, "E1002", "SICK", "2026-10-15", "2026-10-16").ok


def test_sick_three_working_days_late_is_redirected(conn: sqlite3.Connection) -> None:
    # Tue 10-13: 10-14 is a holiday, so the counted days are 15, 16, 19.
    only(check(conn, "E1002", "SICK", "2026-10-13", "2026-10-13"), "SICK_LATE", REDIRECT)


def test_future_sick_needs_known_in_advance(conn: sqlite3.Connection) -> None:
    only(
        check(conn, "E1002", "SICK", "2026-10-26", "2026-10-27"),
        "SICK_FUTURE_UNCONFIRMED",
        REJECT,
    )
    # E1002's rejected annual request (10-26 .. 10-30) does not count as an overlap.
    assert check(conn, "E1002", "SICK", "2026-10-26", "2026-10-27", known_in_advance=True).ok


# --- maximum continuous leave (4.5) ----------------------------------------


def test_single_request_over_fifteen_days(conn: sqlite3.Connection) -> None:
    # 11-02 .. 11-24 = 16 working days (11-23 is a holiday). E1012 has 16 left.
    only(check(conn, "E1012", "ANNUAL", "2026-11-02", "2026-11-24"), "MAX_CONTINUOUS", REDIRECT)


def test_chain_of_exactly_fifteen_days_passes(conn: sqlite3.Connection) -> None:
    add_request(conn, "E1012", "2026-11-02", "2026-11-13", 10)
    assert check(conn, "E1012", "ANNUAL", "2026-11-16", "2026-11-20").ok


def test_chain_of_sixteen_days_split_by_weekend_is_rejected(conn: sqlite3.Connection) -> None:
    add_request(conn, "E1012", "2026-11-02", "2026-11-13", 10)
    # Mon 11-16 .. Tue 11-24 = 6 days (11-23 holiday); 10 + 6 = 16.
    only(check(conn, "E1012", "ANNUAL", "2026-11-16", "2026-11-24"), "MAX_CONTINUOUS", REDIRECT)


def test_chain_works_when_new_request_comes_first(conn: sqlite3.Connection) -> None:
    add_request(conn, "E1012", "2026-11-16", "2026-11-27", 9)  # 11-23 holiday
    # Mon 11-02 .. Fri 11-13 = 10 days, then a weekend, then 9 more = 19.
    result = check(conn, "E1012", "ANNUAL", "2026-11-02", "2026-11-13")
    assert "MAX_CONTINUOUS" in result.codes


def test_requests_with_a_working_day_between_are_not_chained(conn: sqlite3.Connection) -> None:
    add_request(conn, "E1012", "2026-11-02", "2026-11-13", 10)
    # Mon 11-16 is a working day off between the two requests.
    assert check(conn, "E1012", "ANNUAL", "2026-11-17", "2026-11-24").ok


def test_cancelled_requests_do_not_chain(conn: sqlite3.Connection) -> None:
    add_request(conn, "E1012", "2026-11-02", "2026-11-13", 10, status="cancelled")
    assert check(conn, "E1012", "ANNUAL", "2026-11-16", "2026-11-24").ok


# --- restricted periods (4.6) ----------------------------------------------


def test_aud_employee_in_december_restricted_period(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1001", "ANNUAL", "2026-12-01", "2026-12-03"), "RESTRICTED_PERIOD", REDIRECT)


def test_aud_restricted_period_end_is_inclusive(conn: sqlite3.Connection) -> None:
    only(check(conn, "E1001", "ANNUAL", "2026-12-18", "2026-12-18"), "RESTRICTED_PERIOD", REDIRECT)
    assert check(conn, "E1001", "ANNUAL", "2026-12-21", "2026-12-23").ok


def test_restricted_period_only_applies_to_aud(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1003", "ANNUAL", "2026-12-01", "2026-12-03").ok


# --- balance (5.1, 6.4) ----------------------------------------------------


def test_annual_over_balance(conn: sqlite3.Connection) -> None:
    # E1002 has 4 ANNUAL days left.
    only(check(conn, "E1002", "ANNUAL", "2026-11-02", "2026-11-06"), "INSUFFICIENT_BALANCE", REJECT)


def test_sick_over_paid_limit_is_redirected_to_hr(conn: sqlite3.Connection) -> None:
    # E1001 has 8 paid SICK days left; 10-19 .. 10-29 is 9 working days.
    result = check(conn, "E1001", "SICK", "2026-10-19", "2026-10-29")
    only(result, "SICK_PAID_LIMIT", REDIRECT)
    assert result.violations[0].article == "6.4"


# --- unpaid reason (7.2) ---------------------------------------------------


@pytest.mark.parametrize("comment", [None, "", "   "])
def test_unpaid_without_reason(conn: sqlite3.Connection, comment: str | None) -> None:
    result = check(conn, "E1003", "UNPAID", "2026-11-09", "2026-11-10", comment=comment)
    only(result, "REASON_REQUIRED", REJECT)


def test_unpaid_with_reason_passes(conn: sqlite3.Connection) -> None:
    assert check(conn, "E1003", "UNPAID", "2026-11-09", "2026-11-10", comment="გადასვლა").ok


# --- overlap (12.3) --------------------------------------------------------


def test_overlap_with_pending_request(conn: sqlite3.Connection) -> None:
    # E1001 has pending request 5: 11-09 .. 11-11.
    result = check(conn, "E1001", "ANNUAL", "2026-11-10", "2026-11-12")
    only(result, "OVERLAP", REJECT)
    assert "№5" in result.violations[0].message_ka


def test_overlap_applies_across_leave_types(conn: sqlite3.Connection) -> None:
    result = check(conn, "E1001", "SICK", "2026-11-11", "2026-11-11", known_in_advance=True)
    only(result, "OVERLAP", REJECT)


# --- collecting several failures -------------------------------------------


def test_all_policy_failures_are_reported_together(conn: sqlite3.Connection) -> None:
    # E1004: on probation, too little notice, and overlapping a new pending request.
    add_request(conn, "E1004", "2026-10-22", "2026-10-22", 1, leave_type="SICK")
    result = check(conn, "E1004", "ANNUAL", "2026-10-21", "2026-10-23")
    assert result.codes == ["PROBATION", "NOTICE_PERIOD", "OVERLAP"]
    assert result.days == 3
