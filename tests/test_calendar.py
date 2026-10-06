import sqlite3
from datetime import date

import pytest

from server.calendar import (
    count_calendar_days,
    count_working_days,
    is_working_day,
    load_holidays,
    working_days_between,
)

NO_HOLIDAYS: frozenset[date] = frozenset()


def test_st_georges_day_week_has_four_working_days(conn: sqlite3.Connection) -> None:
    holidays = load_holidays(conn)
    assert date(2026, 11, 23) in holidays
    assert count_working_days(date(2026, 11, 23), date(2026, 11, 27), holidays) == 4


def test_is_working_day() -> None:
    holidays = {date(2026, 11, 23)}
    assert is_working_day(date(2026, 11, 24), holidays)  # Tuesday
    assert not is_working_day(date(2026, 11, 23), holidays)  # Monday holiday
    assert not is_working_day(date(2026, 11, 28), holidays)  # Saturday
    assert not is_working_day(date(2026, 11, 29), holidays)  # Sunday


def test_policy_2_2_example_monday_to_monday_with_tuesday_holiday() -> None:
    # Mon 2026-11-02 .. Mon 2026-11-09 with a holiday on Tue 2026-11-03.
    holidays = {date(2026, 11, 3)}
    assert count_working_days(date(2026, 11, 2), date(2026, 11, 9), holidays) == 5


def test_count_calendar_days_includes_weekends_and_holidays() -> None:
    assert count_calendar_days(date(2026, 11, 23), date(2026, 11, 29)) == 7
    assert count_calendar_days(date(2026, 11, 23), date(2026, 11, 23)) == 1


def test_reversed_range_raises() -> None:
    with pytest.raises(ValueError):
        count_working_days(date(2026, 11, 27), date(2026, 11, 23), NO_HOLIDAYS)
    with pytest.raises(ValueError):
        count_calendar_days(date(2026, 11, 27), date(2026, 11, 23))


def test_policy_4_4_example_notice_from_monday() -> None:
    # Submitted Mon 2026-10-19: a 3-day leave (5 working days notice) can start
    # no earlier than Tuesday of the following week.
    submitted = date(2026, 10, 19)
    assert working_days_between(submitted, date(2026, 10, 26), NO_HOLIDAYS) == 4
    assert working_days_between(submitted, date(2026, 10, 27), NO_HOLIDAYS) == 5


def test_working_days_between_skips_holidays_and_adjacent_days() -> None:
    holidays = {date(2026, 11, 23)}
    # Fri 2026-11-20 -> Wed 2026-11-25: only Tuesday counts (Monday is a holiday).
    assert working_days_between(date(2026, 11, 20), date(2026, 11, 25), holidays) == 1
    assert working_days_between(date(2026, 11, 24), date(2026, 11, 25), holidays) == 0
    assert working_days_between(date(2026, 11, 24), date(2026, 11, 24), holidays) == 0


def test_days_in_csv_requests_match_counting_rules(conn: sqlite3.Connection) -> None:
    holidays = load_holidays(conn)
    rows = conn.execute(
        """
        SELECT r.request_id, r.start_date, r.end_date, r.days, t.day_unit
        FROM leave_requests r JOIN leave_types t ON t.code = r.leave_type
        """
    ).fetchall()
    assert rows
    for row in rows:
        start, end = date.fromisoformat(row["start_date"]), date.fromisoformat(row["end_date"])
        if row["day_unit"] == "calendar":
            counted = count_calendar_days(start, end)
        else:
            counted = count_working_days(start, end, holidays)
        assert counted == row["days"], row["request_id"]
