"""Day counting under Leave Policy Articles 1.3, 2.2 and 4.4.

Holidays come only from the public_holidays table (Article 2.3); pass them in
so the functions stay pure and easy to test.
"""

import os
import sqlite3
from collections.abc import Collection
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def today() -> date:
    """The app's "today": APP_TODAY from .env if set, otherwise the real date.

    Call this only at the edges (MCP tools, CLI); business logic takes `today`
    as a parameter so tests stay deterministic.
    """
    load_dotenv(_ENV_FILE)
    raw = os.getenv("APP_TODAY")
    if not raw:
        return datetime.now().astimezone().date()
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"APP_TODAY must be YYYY-MM-DD, got {raw!r}") from exc


def load_holidays(conn: sqlite3.Connection) -> frozenset[date]:
    return frozenset(
        date.fromisoformat(row[0]) for row in conn.execute("SELECT date FROM public_holidays")
    )


def is_working_day(day: date, holidays: Collection[date]) -> bool:
    """Monday to Friday and not a public holiday (Article 1.3)."""
    return day.weekday() < 5 and day not in holidays


def _check_range(start: date, end: date) -> None:
    if start > end:
        raise ValueError(f"start {start} is after end {end}")


def count_working_days(start: date, end: date, holidays: Collection[date]) -> int:
    """Working days from start to end, both inclusive (Article 2.2)."""
    _check_range(start, end)
    return sum(
        is_working_day(start + timedelta(days=offset), holidays)
        for offset in range((end - start).days + 1)
    )


def count_calendar_days(start: date, end: date) -> int:
    """Calendar days from start to end, both inclusive (Article 7.1)."""
    _check_range(start, end)
    return (end - start).days + 1


def working_days_between(start: date, end: date, holidays: Collection[date]) -> int:
    """Full working days strictly between start and end (Article 4.4 notice periods).

    Both endpoints are excluded: the submission day and the first day of leave.
    Returns 0 when there is no day in between.
    """
    if end - start <= timedelta(days=1):
        return 0
    return count_working_days(start + timedelta(days=1), end - timedelta(days=1), holidays)
