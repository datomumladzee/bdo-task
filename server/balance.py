"""Leave balance under Leave Policy Article 5.1."""

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Balance:
    employee_id: str
    year: int
    leave_type: str
    entitled_days: int
    carried_over_days: int
    approved_days: int
    pending_days: int
    available_days: int


def get_balance(conn: sqlite3.Connection, employee_id: str, year: int, leave_type: str) -> Balance:
    """available = entitled + carried over - approved - pending.

    Only requests of the same employee, type and leave year count; a request
    belongs to the year of its first day (Article 2.1). Rejected and cancelled
    requests are ignored. A negative SICK result is shown as 0 (Article 5.1).
    Raises LookupError when there is no entitlement row, which is always the
    case for BEREAVEMENT and PARENTAL (no annual balance).
    """
    entitlement = conn.execute(
        """
        SELECT entitled_days, carried_over_days
        FROM leave_entitlements
        WHERE employee_id = ? AND year = ? AND leave_type = ?
        """,
        (employee_id, year, leave_type),
    ).fetchone()
    if entitlement is None:
        raise LookupError(f"No {leave_type} entitlement for {employee_id} in {year}")

    used = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN status = 'approved' THEN days END), 0) AS approved,
            COALESCE(SUM(CASE WHEN status = 'pending' THEN days END), 0) AS pending
        FROM leave_requests
        WHERE employee_id = ? AND leave_type = ?
          AND start_date BETWEEN ? AND ?
        """,
        (employee_id, leave_type, f"{year:04d}-01-01", f"{year:04d}-12-31"),
    ).fetchone()

    available = (
        entitlement["entitled_days"]
        + entitlement["carried_over_days"]
        - used["approved"]
        - used["pending"]
    )
    if leave_type == "SICK":
        available = max(available, 0)

    return Balance(
        employee_id=employee_id,
        year=year,
        leave_type=leave_type,
        entitled_days=entitlement["entitled_days"],
        carried_over_days=entitlement["carried_over_days"],
        approved_days=used["approved"],
        pending_days=used["pending"],
        available_days=available,
    )
