"""MCP server for Northstar Services leave management.

Who is calling is fixed at startup, never taken from tool arguments:

    uv run python -m server.mcp_server --employee E1001 --role employee
    uv run python -m server.mcp_server --employee E1007 --role hr

Employee tools act only on the caller's own data. HR tools check the role and
are refused for an employee (Policy Articles 5.2 and 12.3).
"""

import argparse
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from server import calendar, service
from server.db import get_connection, get_db_path, init_db

INSTRUCTIONS = """\
Northstar Services HR leave tools. The caller's identity is fixed by the server.

To create a leave request:
1. Call propose_leave_request. Nothing is saved yet.
2. Show the employee the leave type, dates and number of days, and ask for a
   clear confirmation (Policy Article 12.2).
3. Only after a clear "yes" in the employee's latest message, call
   confirm_leave_request with the proposal_id. Never confirm on their behalf.

If a tool returns violations, explain each message_ka with its policy article
and offer the alternative. A created request is pending, not approved.
"""


def _parse_date(value: str | None, name: str) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"INVALID_DATE: {name} must be YYYY-MM-DD, got {value!r}") from exc


def _required_date(value: str, name: str) -> date:
    parsed = _parse_date(value, name)
    if parsed is None:
        raise ToolError(f"INVALID_DATE: {name} is required (YYYY-MM-DD)")
    return parsed


def build_server(
    identity: service.Identity,
    *,
    db_path: str | Path,
    conversation_id: str,
    today_fn: Callable[[], date] = calendar.today,
    now_fn: Callable[[], datetime] = calendar.now,
) -> MCPServer:
    """Create the server for one caller. Tests inject today_fn/now_fn."""
    mcp = MCPServer("northstar-leave", instructions=INSTRUCTIONS)

    @contextmanager
    def connect() -> Iterator[sqlite3.Connection]:
        # One connection per tool call: sqlite3 connections must not be shared
        # across the threads the server may run tools on.
        conn = get_connection(db_path)
        try:
            yield conn
        except service.ServiceError as exc:
            raise ToolError(f"{exc.code}: {exc.message}") from exc
        finally:
            conn.close()

    def hr_only() -> None:
        try:
            service.require_hr(identity)
        except service.ServiceError as exc:
            raise ToolError(f"{exc.code}: {exc.message}") from exc

    # --- employee tools ----------------------------------------------------

    @mcp.tool()
    def list_leave_types() -> dict[str, Any]:
        """List all six leave types, their counting unit, annual limit, policy article,
        and whether the assistant can create requests of that type."""
        with connect() as conn:
            return {"leave_types": service.list_leave_types(conn)}

    @mcp.tool()
    def get_my_balance(year: int | None = None, leave_type: str | None = None) -> dict[str, Any]:
        """Show the caller's own leave balance: entitled, carried over, approved,
        pending and available days. Defaults to the current year and all types."""
        with connect() as conn:
            year = year or today_fn().year
            balances = service.get_balances(conn, identity, year=year, leave_type=leave_type)
            return {"employee_id": identity.employee_id, "year": year, "balances": balances}

    @mcp.tool()
    def list_my_requests(
        status: str | None = None, date_from: str | None = None, date_to: str | None = None
    ) -> dict[str, Any]:
        """List the caller's own leave requests. Optional filters: status (pending,
        approved, rejected, cancelled) and a date range (YYYY-MM-DD, any overlap)."""
        start = _parse_date(date_from, "date_from")
        end = _parse_date(date_to, "date_to")
        with connect() as conn:
            requests = service.list_requests(
                conn, identity, status=status, date_from=start, date_to=end
            )
            return {"count": len(requests), "requests": requests}

    @mcp.tool()
    def propose_leave_request(
        leave_type: str,
        start_date: str,
        end_date: str,
        comment: str | None = None,
        known_in_advance: bool = False,
    ) -> dict[str, Any]:
        """Check a leave request for the caller and prepare it for confirmation.
        Nothing is saved as a request yet. leave_type is ANNUAL, SICK or UNPAID
        (others are explained and redirected). Dates are YYYY-MM-DD, end inclusive.
        UNPAID needs a short reason in comment. For SICK leave on future dates, set
        known_in_advance only if the employee confirmed the period is already known.
        On success, show the details and ask the employee to confirm."""
        start = _required_date(start_date, "start_date")
        end = _required_date(end_date, "end_date")
        with connect() as conn:
            return service.propose_leave(
                conn,
                identity,
                conversation_id=conversation_id,
                leave_type=leave_type,
                start=start,
                end=end,
                today=today_fn(),
                now=now_fn(),
                comment=comment,
                known_in_advance=known_in_advance,
            )

    @mcp.tool()
    def confirm_leave_request(proposal_id: str) -> dict[str, Any]:
        """Create the pending leave request for a proposal the employee has clearly
        confirmed. Calling it again for the same proposal returns the same request_id."""
        with connect() as conn:
            return service.confirm_leave(
                conn,
                identity,
                conversation_id=conversation_id,
                proposal_id=proposal_id,
                today=today_fn(),
                now=now_fn(),
            )

    # --- HR tools ------------------------------------------------------------

    @mcp.tool()
    def list_requests(
        employee_id: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """HR only. List leave requests filtered by employee, status (pending,
        approved, rejected, cancelled) and a date range (YYYY-MM-DD, any overlap)."""
        hr_only()
        start = _parse_date(date_from, "date_from")
        end = _parse_date(date_to, "date_to")
        with connect() as conn:
            requests = service.list_requests(
                conn, identity, employee_id=employee_id, status=status, date_from=start, date_to=end
            )
            return {"count": len(requests), "requests": requests}

    @mcp.tool()
    def get_balance(
        employee_id: str, year: int | None = None, leave_type: str | None = None
    ) -> dict[str, Any]:
        """HR only. Show any employee's leave balance for a year (default: current)
        and leave type (default: all types)."""
        hr_only()
        with connect() as conn:
            year = year or today_fn().year
            balances = service.get_balances(
                conn, identity, year=year, employee_id=employee_id, leave_type=leave_type
            )
            return {"employee_id": employee_id, "year": year, "balances": balances}

    @mcp.tool()
    def create_request(
        employee_id: str,
        leave_type: str,
        start_date: str,
        end_date: str,
        comment: str | None = None,
        known_in_advance: bool = False,
    ) -> dict[str, Any]:
        """HR only. Create a pending leave request for an employee in one step, under
        the same policy rules as the employee assistant (ANNUAL, SICK and UNPAID only;
        dates YYYY-MM-DD, end inclusive; UNPAID needs a reason in comment). Returns
        violations instead of creating the request if a rule is broken."""
        hr_only()
        start = _required_date(start_date, "start_date")
        end = _required_date(end_date, "end_date")
        with connect() as conn:
            return service.create_request(
                conn,
                identity,
                employee_id=employee_id,
                leave_type=leave_type,
                start=start,
                end=end,
                today=today_fn(),
                now=now_fn(),
                comment=comment,
                known_in_advance=known_in_advance,
            )

    @mcp.tool()
    def approve_request(request_id: int, comment: str | None = None) -> dict[str, Any]:
        """HR only. Approve a pending leave request."""
        hr_only()
        with connect() as conn:
            return service.approve_request(
                conn, identity, request_id, now=now_fn(), comment=comment
            )

    @mcp.tool()
    def reject_request(request_id: int, reason: str) -> dict[str, Any]:
        """HR only. Reject a pending leave request. A written reason is required."""
        hr_only()
        with connect() as conn:
            return service.reject_request(conn, identity, request_id, reason=reason, now=now_fn())

    @mcp.tool()
    def cancel_request(request_id: int, reason: str | None = None) -> dict[str, Any]:
        """HR only. Cancel a pending or approved leave request."""
        hr_only()
        with connect() as conn:
            return service.cancel_request(conn, identity, request_id, now=now_fn(), reason=reason)

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Northstar Services leave MCP server (stdio)")
    parser.add_argument("--employee", required=True, help="caller's employee id, e.g. E1001")
    parser.add_argument("--role", choices=service.ROLES, default="employee")
    parser.add_argument(
        "--conversation", help="conversation id for proposals (default: a new random id)"
    )
    args = parser.parse_args(argv)

    db_path = get_db_path()
    init_db(db_path)
    with closing(get_connection(db_path)) as conn:
        try:
            identity = service.load_identity(conn, args.employee, args.role)
        except service.ServiceError as exc:
            parser.error(exc.message)

    server = build_server(
        identity, db_path=db_path, conversation_id=args.conversation or uuid.uuid4().hex
    )
    server.run()


if __name__ == "__main__":
    main()
