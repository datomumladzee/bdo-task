import sqlite3
from datetime import date, datetime

import pytest

from server import service
from server.service import Identity

TODAY = date(2026, 10, 19)
NOW = datetime.fromisoformat("2026-10-19T10:00:00")  # naive, like the CSV timestamps
EMPLOYEE = Identity("E1003", "employee")
HR = Identity("E1007", "hr")


def propose(conn: sqlite3.Connection, conversation: str = "conv-1", **overrides: object) -> dict:
    args: dict = {
        "conversation_id": conversation,
        "leave_type": "ANNUAL",
        "start": date(2026, 10, 27),
        "end": date(2026, 10, 29),
        "today": TODAY,
        "now": NOW,
    }
    args.update(overrides)
    return service.propose_leave(conn, EMPLOYEE, **args)


def confirm(conn: sqlite3.Connection, proposal_id: str, conversation: str = "conv-1") -> dict:
    return service.confirm_leave(
        conn, EMPLOYEE, conversation_id=conversation, proposal_id=proposal_id, today=TODAY, now=NOW
    )


def request_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]


# --- identity -----------------------------------------------------------------


def test_load_identity_rejects_unknown_employee_and_role(conn: sqlite3.Connection) -> None:
    assert service.load_identity(conn, "E1001", "employee") == Identity("E1001", "employee")
    with pytest.raises(service.NotFound):
        service.load_identity(conn, "E9999", "employee")
    with pytest.raises(service.InvalidRequest):
        service.load_identity(conn, "E1001", "admin")


# --- propose / confirm ----------------------------------------------------------


def test_propose_stores_a_proposal_but_no_request(conn: sqlite3.Connection) -> None:
    before = request_count(conn)
    result = propose(conn)
    assert result["ok"] is True
    assert result["days"] == 3
    assert result["available_before"] == 15
    assert result["available_after"] == 12
    assert request_count(conn) == before
    row = conn.execute(
        "SELECT employee_id, conversation_id, status FROM leave_proposals WHERE proposal_id = ?",
        (result["proposal_id"],),
    ).fetchone()
    assert tuple(row) == ("E1003", "conv-1", "proposed")


def test_propose_returns_violations_and_stores_nothing(conn: sqlite3.Connection) -> None:
    result = propose(conn, start=date(2026, 10, 23), end=date(2026, 10, 27))
    assert result["ok"] is False
    assert [v["code"] for v in result["violations"]] == ["NOTICE_PERIOD"]
    assert conn.execute("SELECT COUNT(*) FROM leave_proposals").fetchone()[0] == 0


def test_confirm_creates_pending_assistant_request(conn: sqlite3.Connection) -> None:
    proposal = propose(conn)
    result = confirm(conn, proposal["proposal_id"])
    assert result == {
        "ok": True,
        "request_id": 28,
        "status": "pending",
        "already_confirmed": False,
    }
    row = conn.execute("SELECT * FROM leave_requests WHERE request_id = 28").fetchone()
    assert row["employee_id"] == "E1003"
    assert row["status"] == "pending"
    assert row["created_via"] == "assistant"
    assert row["days"] == 3
    assert row["created_at"] == "2026-10-19T10:00:00"


def test_confirm_twice_returns_same_request_and_creates_one_row(conn: sqlite3.Connection) -> None:
    proposal = propose(conn)
    first = confirm(conn, proposal["proposal_id"])
    count = request_count(conn)
    second = confirm(conn, proposal["proposal_id"])
    assert second["request_id"] == first["request_id"]
    assert second["already_confirmed"] is True
    assert request_count(conn) == count


def test_pending_request_reduces_balance(conn: sqlite3.Connection) -> None:
    confirm(conn, propose(conn)["proposal_id"])
    [balance] = service.get_balances(conn, EMPLOYEE, year=2026, leave_type="ANNUAL")
    assert balance["pending_days"] == 3
    assert balance["available_days"] == 12


def test_confirm_needs_same_employee_and_conversation(conn: sqlite3.Connection) -> None:
    proposal = propose(conn)
    with pytest.raises(service.NotFound):
        confirm(conn, proposal["proposal_id"], conversation="other-conversation")
    with pytest.raises(service.NotFound):
        service.confirm_leave(
            conn,
            Identity("E1001", "employee"),
            conversation_id="conv-1",
            proposal_id=proposal["proposal_id"],
            today=TODAY,
            now=NOW,
        )


def test_new_proposal_expires_the_previous_one(conn: sqlite3.Connection) -> None:
    old = propose(conn)
    new = propose(conn, start=date(2026, 11, 2), end=date(2026, 11, 3))
    with pytest.raises(service.InvalidRequest) as error:
        confirm(conn, old["proposal_id"])
    assert error.value.code == "PROPOSAL_EXPIRED"
    assert confirm(conn, new["proposal_id"])["ok"] is True


def test_confirm_rechecks_the_rules(conn: sqlite3.Connection) -> None:
    proposal = propose(conn)
    # Meanwhile a portal request for the same dates appears.
    conn.execute(
        """
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days, status, created_at, created_via)
        VALUES ('E1003', 'ANNUAL', '2026-10-28', '2026-10-28', 1, 'pending',
                '2026-10-19T09:00:00', 'portal')
        """
    )
    result = confirm(conn, proposal["proposal_id"])
    assert result["ok"] is False
    assert [v["code"] for v in result["violations"]] == ["OVERLAP"]
    status = conn.execute(
        "SELECT status FROM leave_proposals WHERE proposal_id = ?", (proposal["proposal_id"],)
    ).fetchone()[0]
    assert status == "expired"


# --- reading and permissions ----------------------------------------------------


def test_employee_sees_only_own_requests(conn: sqlite3.Connection) -> None:
    requests = service.list_requests(conn, EMPLOYEE)
    assert {r["employee_id"] for r in requests} == {"E1003"}
    with pytest.raises(service.PermissionDenied):
        service.list_requests(conn, EMPLOYEE, employee_id="E1001")
    with pytest.raises(service.PermissionDenied):
        service.get_balances(conn, EMPLOYEE, year=2026, employee_id="E1001")


def test_list_requests_filters(conn: sqlite3.Connection) -> None:
    pending = service.list_requests(conn, HR, status="pending")
    assert [r["request_id"] for r in pending] == [5, 15]
    november = service.list_requests(
        conn, HR, date_from=date(2026, 11, 1), date_to=date(2026, 11, 30)
    )
    assert [r["request_id"] for r in november] == [5, 15]
    e1001_july = service.list_requests(
        conn, HR, employee_id="E1001", date_from=date(2026, 7, 20), date_to=date(2026, 7, 20)
    )
    assert [r["request_id"] for r in e1001_july] == [4]  # 07-13 .. 07-24 overlaps


def test_list_requests_rejects_bad_filters(conn: sqlite3.Connection) -> None:
    with pytest.raises(service.InvalidRequest):
        service.list_requests(conn, HR, status="done")
    with pytest.raises(service.InvalidRequest):
        service.list_requests(conn, HR, date_from=date(2026, 12, 1), date_to=date(2026, 11, 1))


def test_get_balances_all_types(conn: sqlite3.Connection) -> None:
    balances = service.get_balances(conn, HR, year=2026, employee_id="E1001")
    by_type = {b["leave_type"]: b["available_days"] for b in balances}
    assert by_type == {"ANNUAL": 10, "SICK": 8, "STUDY": 5, "UNPAID": 30}


# --- HR decisions ---------------------------------------------------------------


@pytest.mark.parametrize("action", ["approve", "reject", "cancel"])
def test_hr_actions_are_denied_for_employees(conn: sqlite3.Connection, action: str) -> None:
    with pytest.raises(service.PermissionDenied):
        if action == "approve":
            service.approve_request(conn, EMPLOYEE, 5, now=NOW)
        elif action == "reject":
            service.reject_request(conn, EMPLOYEE, 5, reason="x", now=NOW)
        else:
            service.cancel_request(conn, EMPLOYEE, 5, now=NOW)


def test_approve_pending_request(conn: sqlite3.Connection) -> None:
    result = service.approve_request(conn, HR, 5, now=NOW)
    assert result["status"] == "approved"
    assert result["decided_by"] == "E1007"
    assert result["decided_at"] == "2026-10-19T10:00:00"


def test_reject_needs_reason(conn: sqlite3.Connection) -> None:
    with pytest.raises(service.InvalidRequest):
        service.reject_request(conn, HR, 5, reason="  ", now=NOW)
    result = service.reject_request(conn, HR, 5, reason="კლიენტის პროექტი", now=NOW)
    assert result["status"] == "rejected"
    assert result["decision_comment"] == "კლიენტის პროექტი"


def test_only_pending_requests_can_be_decided(conn: sqlite3.Connection) -> None:
    with pytest.raises(service.InvalidRequest) as error:
        service.approve_request(conn, HR, 1, now=NOW)  # already approved
    assert error.value.code == "INVALID_STATE"


def test_cancel_pending_or_approved_but_not_twice(conn: sqlite3.Connection) -> None:
    assert service.cancel_request(conn, HR, 4, now=NOW)["status"] == "cancelled"  # approved
    with pytest.raises(service.InvalidRequest):
        service.cancel_request(conn, HR, 4, now=NOW)


def test_unknown_request(conn: sqlite3.Connection) -> None:
    with pytest.raises(service.NotFound):
        service.approve_request(conn, HR, 999, now=NOW)
