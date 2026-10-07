"""Leave actions behind the MCP tools, with role checks.

Who is calling (employee id and role) is fixed when the server starts and is
passed in as an Identity. It never comes from the AI's tool arguments, so the
assistant cannot act as someone else (Policy Articles 5.2 and 12.3).

Employees create requests in two steps (Article 12.2): propose_leave() checks
the request and stores a proposal; confirm_leave() creates the request only
after the employee has clearly confirmed. Confirming the same proposal again
returns the same request id and creates nothing new.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from server.balance import get_balance
from server.rules import ValidationResult, validate_request

ROLES = ("employee", "hr")
STATUSES = ("pending", "approved", "rejected", "cancelled")
HR_DEPARTMENT = "HRS"


@dataclass(frozen=True)
class Identity:
    employee_id: str
    role: str


class ServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class PermissionDenied(ServiceError):
    pass


class NotFound(ServiceError):
    pass


class InvalidRequest(ServiceError):
    pass


@contextmanager
def _write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """BEGIN IMMEDIATE takes the write lock up front, so two confirmations of the
    same proposal cannot both pass the "already confirmed?" check."""
    if conn.in_transaction:  # the caller already holds a transaction; join it
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def load_identity(conn: sqlite3.Connection, employee_id: str, role: str) -> Identity:
    """Check the startup identity. The hr role is only for Human Resources staff."""
    if role not in ROLES:
        raise InvalidRequest("INVALID_ROLE", f"role must be one of {ROLES}, got {role!r}")
    _require_employee(conn, employee_id)
    if role == "hr":
        department = conn.execute(
            "SELECT department_code FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()["department_code"]
        if department != HR_DEPARTMENT:
            raise PermissionDenied(
                "HR_ROLE_NOT_ALLOWED",
                f"Employee {employee_id} is in department {department}, not {HR_DEPARTMENT}; "
                "only Human Resources staff can use the hr role.",
            )
    return Identity(employee_id=employee_id, role=role)


def require_hr(identity: Identity) -> None:
    if identity.role != "hr":
        raise PermissionDenied(
            "HR_ONLY",
            "This action is for HR only. The employee assistant cannot approve, reject, "
            "cancel or view other employees' data (Policy Article 12.3).",
        )


def _require_employee(conn: sqlite3.Connection, employee_id: str) -> None:
    found = conn.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    if found is None:
        raise NotFound("UNKNOWN_EMPLOYEE", f"Employee {employee_id} not found")


def _scope_to_caller(identity: Identity, employee_id: str | None) -> str | None:
    """Employees only ever see their own data (Article 5.2); HR may see anyone's."""
    if identity.role == "hr":
        return employee_id
    if employee_id not in (None, identity.employee_id):
        raise PermissionDenied(
            "OWN_DATA_ONLY",
            "Employees can only see their own data (Policy Articles 5.2 and 12.3).",
        )
    return identity.employee_id


# --- reading ----------------------------------------------------------------


def list_leave_types(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT code, name, day_unit, annual_limit_days, self_service,
               assistant_supported, policy_reference
        FROM leave_types ORDER BY rowid
        """
    ).fetchall()
    return [
        {
            "code": r["code"],
            "name": r["name"],
            "day_unit": r["day_unit"],
            "annual_limit_days": r["annual_limit_days"],
            "self_service": bool(r["self_service"]),
            "assistant_can_create": bool(r["assistant_supported"]),
            "policy_reference": r["policy_reference"],
        }
        for r in rows
    ]


def get_balances(
    conn: sqlite3.Connection,
    identity: Identity,
    *,
    year: int,
    employee_id: str | None = None,
    leave_type: str | None = None,
) -> list[dict[str, Any]]:
    """One leave type, or every type the employee has an entitlement for that year."""
    target = _scope_to_caller(identity, employee_id)
    if target is None:
        raise InvalidRequest("EMPLOYEE_REQUIRED", "employee_id is required")
    _require_employee(conn, target)
    if leave_type:
        types = [leave_type.strip().upper()]
    else:
        types = [
            r["leave_type"]
            for r in conn.execute(
                "SELECT leave_type FROM leave_entitlements WHERE employee_id = ? AND year = ? "
                "ORDER BY leave_type",
                (target, year),
            )
        ]
    balances = []
    for code in types:
        try:
            balances.append(asdict(get_balance(conn, target, year, code)))
        except LookupError as exc:
            raise NotFound(
                "NO_BALANCE",
                f"{exc}. BEREAVEMENT and PARENTAL have no annual balance (Policy Article 5.1).",
            ) from exc
    return balances


def list_requests(
    conn: sqlite3.Connection,
    identity: Identity,
    *,
    employee_id: str | None = None,
    status: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict[str, Any]]:
    """Requests filtered by employee, status and a date range (any overlap counts)."""
    target = _scope_to_caller(identity, employee_id)
    if status is not None and status not in STATUSES:
        raise InvalidRequest("INVALID_STATUS", f"status must be one of {STATUSES}")
    if date_from and date_to and date_from > date_to:
        raise InvalidRequest("INVALID_DATES", "date_from is after date_to")
    if target is not None:
        _require_employee(conn, target)

    clauses: list[str] = []
    params: list[str] = []
    if target is not None:
        clauses.append("employee_id = ?")
        params.append(target)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if date_from is not None:
        clauses.append("end_date >= ?")
        params.append(date_from.isoformat())
    if date_to is not None:
        clauses.append("start_date <= ?")
        params.append(date_to.isoformat())
    where = " AND ".join(clauses) if clauses else "1 = 1"
    rows = conn.execute(
        f"SELECT * FROM leave_requests WHERE {where} ORDER BY start_date, request_id", params
    ).fetchall()
    return [dict(r) for r in rows]


# --- propose and confirm (employee) ----------------------------------------


def _violations(result: ValidationResult) -> list[dict[str, Any]]:
    return [asdict(v) for v in result.violations]


def propose_leave(
    conn: sqlite3.Connection,
    identity: Identity,
    *,
    conversation_id: str,
    leave_type: str,
    start: date,
    end: date,
    today: date,
    now: datetime,
    comment: str | None = None,
    known_in_advance: bool = False,
) -> dict[str, Any]:
    """Check a request and, if it passes, store a proposal for the employee to confirm.

    Nothing is written to leave_requests here. Older unconfirmed proposals in the
    same conversation expire, so only the latest one can be confirmed.
    """
    leave_type = leave_type.strip().upper()
    comment = comment.strip() if comment and comment.strip() else None
    result = validate_request(
        conn,
        employee_id=identity.employee_id,
        leave_type=leave_type,
        start=start,
        end=end,
        today=today,
        comment=comment,
        known_in_advance=known_in_advance,
    )
    if not result.ok:
        return {"ok": False, "violations": _violations(result)}
    assert result.days is not None

    balance = get_balance(conn, identity.employee_id, start.year, leave_type)
    proposal_id = uuid.uuid4().hex
    payload = {
        "leave_type": leave_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": result.days,
        "comment": comment,
        "known_in_advance": known_in_advance,
    }
    with _write_transaction(conn):
        conn.execute(
            """
            UPDATE leave_proposals SET status = 'expired'
            WHERE employee_id = ? AND conversation_id = ? AND status = 'proposed'
            """,
            (identity.employee_id, conversation_id),
        )
        conn.execute(
            """
            INSERT INTO leave_proposals
                (proposal_id, employee_id, conversation_id, payload, status, created_at)
            VALUES (?, ?, ?, ?, 'proposed', ?)
            """,
            (
                proposal_id,
                identity.employee_id,
                conversation_id,
                json.dumps(payload, ensure_ascii=False),
                now.isoformat(),
            ),
        )
    return {
        "ok": True,
        "proposal_id": proposal_id,
        **payload,
        "available_before": balance.available_days,
        "available_after": balance.available_days - result.days,
    }


def confirm_leave(
    conn: sqlite3.Connection,
    identity: Identity,
    *,
    conversation_id: str,
    proposal_id: str,
    today: date,
    now: datetime,
) -> dict[str, Any]:
    """Create the pending request for a confirmed proposal (idempotent)."""
    with _write_transaction(conn):
        proposal = conn.execute(
            """
            SELECT status, payload, request_id FROM leave_proposals
            WHERE proposal_id = ? AND employee_id = ? AND conversation_id = ?
            """,
            (proposal_id, identity.employee_id, conversation_id),
        ).fetchone()
        if proposal is None:
            raise NotFound(
                "UNKNOWN_PROPOSAL", "No such proposal for this employee and conversation"
            )
        if proposal["status"] == "confirmed":
            return {
                "ok": True,
                "request_id": proposal["request_id"],
                "status": "pending",
                "already_confirmed": True,
            }
        if proposal["status"] == "expired":
            raise InvalidRequest(
                "PROPOSAL_EXPIRED",
                "This proposal was replaced or is no longer valid. Propose the request again.",
            )

        payload = json.loads(proposal["payload"])
        # Re-check: the situation may have changed since the proposal (Article 12.3).
        result = validate_request(
            conn,
            employee_id=identity.employee_id,
            leave_type=payload["leave_type"],
            start=date.fromisoformat(payload["start_date"]),
            end=date.fromisoformat(payload["end_date"]),
            today=today,
            comment=payload["comment"],
            known_in_advance=payload["known_in_advance"],
        )
        if not result.ok:
            conn.execute(
                "UPDATE leave_proposals SET status = 'expired' WHERE proposal_id = ?",
                (proposal_id,),
            )
            return {"ok": False, "violations": _violations(result)}

        assert result.days is not None
        request_id = _insert_request(
            conn,
            employee_id=identity.employee_id,
            leave_type=payload["leave_type"],
            start=payload["start_date"],
            end=payload["end_date"],
            days=result.days,
            comment=payload["comment"],
            now=now,
        )
        conn.execute(
            "UPDATE leave_proposals SET status = 'confirmed', request_id = ? WHERE proposal_id = ?",
            (request_id, proposal_id),
        )
    return {"ok": True, "request_id": request_id, "status": "pending", "already_confirmed": False}


def _insert_request(
    conn: sqlite3.Connection,
    *,
    employee_id: str,
    leave_type: str,
    start: str,
    end: str,
    days: int,
    comment: str | None,
    now: datetime,
) -> int:
    """Every request created through the MCP server is pending and created via the
    assistant (data dictionary)."""
    cursor = conn.execute(
        """
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days, status,
             created_at, created_via, comment)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, 'assistant', ?)
        """,
        (employee_id, leave_type, start, end, days, now.isoformat(), comment),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


# --- create on behalf of an employee (HR) ------------------------------------


def create_request(
    conn: sqlite3.Connection,
    identity: Identity,
    *,
    employee_id: str,
    leave_type: str,
    start: date,
    end: date,
    today: date,
    now: datetime,
    comment: str | None = None,
    known_in_advance: bool = False,
) -> dict[str, Any]:
    """HR creates a request for an employee in one step, under the same rules as
    the employee assistant. Calling it twice is caught by the overlap rule."""
    require_hr(identity)
    leave_type = leave_type.strip().upper()
    comment = comment.strip() if comment and comment.strip() else None
    with _write_transaction(conn):
        result = validate_request(
            conn,
            employee_id=employee_id,
            leave_type=leave_type,
            start=start,
            end=end,
            today=today,
            comment=comment,
            known_in_advance=known_in_advance,
        )
        if not result.ok:
            return {"ok": False, "violations": _violations(result)}
        assert result.days is not None
        request_id = _insert_request(
            conn,
            employee_id=employee_id,
            leave_type=leave_type,
            start=start.isoformat(),
            end=end.isoformat(),
            days=result.days,
            comment=comment,
            now=now,
        )
    return {
        "ok": True,
        "request_id": request_id,
        "employee_id": employee_id,
        "leave_type": leave_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": result.days,
        "status": "pending",
    }


# --- decisions (HR) ---------------------------------------------------------


def _get_request(conn: sqlite3.Connection, request_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)
    ).fetchone()
    if row is None:
        raise NotFound("UNKNOWN_REQUEST", f"Request {request_id} not found")
    return row


def _set_status(
    conn: sqlite3.Connection,
    identity: Identity,
    request_id: int,
    *,
    new_status: str,
    allowed_from: tuple[str, ...],
    comment: str | None,
    now: datetime,
) -> dict[str, Any]:
    with _write_transaction(conn):
        current = _get_request(conn, request_id)["status"]
        if current not in allowed_from:
            raise InvalidRequest(
                "INVALID_STATE",
                f"Request {request_id} is {current}; only {' or '.join(allowed_from)} "
                f"requests can become {new_status}.",
            )
        conn.execute(
            """
            UPDATE leave_requests
            SET status = ?, decided_by = ?, decided_at = ?, decision_comment = ?
            WHERE request_id = ?
            """,
            (new_status, identity.employee_id, now.isoformat(), comment, request_id),
        )
    return dict(_get_request(conn, request_id))


def approve_request(
    conn: sqlite3.Connection,
    identity: Identity,
    request_id: int,
    *,
    now: datetime,
    comment: str | None = None,
) -> dict[str, Any]:
    require_hr(identity)
    return _set_status(
        conn,
        identity,
        request_id,
        new_status="approved",
        allowed_from=("pending",),
        comment=comment,
        now=now,
    )


def reject_request(
    conn: sqlite3.Connection,
    identity: Identity,
    request_id: int,
    *,
    reason: str,
    now: datetime,
) -> dict[str, Any]:
    """A rejection must state its reason (Article 13.2)."""
    require_hr(identity)
    if not reason or not reason.strip():
        raise InvalidRequest("REASON_REQUIRED", "A rejection needs a written reason (Article 13.2)")
    return _set_status(
        conn,
        identity,
        request_id,
        new_status="rejected",
        allowed_from=("pending",),
        comment=reason.strip(),
        now=now,
    )


def cancel_request(
    conn: sqlite3.Connection,
    identity: Identity,
    request_id: int,
    *,
    now: datetime,
    reason: str | None = None,
) -> dict[str, Any]:
    require_hr(identity)
    return _set_status(
        conn,
        identity,
        request_id,
        new_status="cancelled",
        allowed_from=("pending", "approved"),
        comment=reason.strip() if reason and reason.strip() else None,
        now=now,
    )
