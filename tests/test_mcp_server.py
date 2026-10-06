import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from server.mcp_server import build_server
from server.service import Identity

TODAY = date(2026, 10, 19)
NOW = datetime.fromisoformat("2026-10-19T10:00:00")  # naive, like the CSV timestamps
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMPLOYEE_TOOLS = {
    "list_leave_types",
    "get_my_balance",
    "list_my_requests",
    "propose_leave_request",
    "confirm_leave_request",
}
HR_TOOLS = {"list_requests", "get_balance", "approve_request", "reject_request", "cancel_request"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def client(db_path: Path, employee_id: str, role: str) -> Client:
    server = build_server(
        Identity(employee_id, role),
        db_path=db_path,
        conversation_id="conv-1",
        today_fn=lambda: TODAY,
        now_fn=lambda: NOW,
    )
    return Client(server)


async def call(c: Client, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await c.call_tool(tool, args or {})
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


async def call_error(c: Client, tool: str, args: dict[str, Any] | None = None) -> str:
    result = await c.call_tool(tool, args or {})
    assert result.is_error
    return result.content[0].text


@pytest.mark.anyio
async def test_all_tools_are_listed(db_path: Path) -> None:
    async with client(db_path, "E1001", "employee") as c:
        tools = await c.list_tools()
    assert {t.name for t in tools.tools} == EMPLOYEE_TOOLS | HR_TOOLS


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("list_requests", {}),
        ("get_balance", {"employee_id": "E1002"}),
        ("approve_request", {"request_id": 5}),
        ("reject_request", {"request_id": 5, "reason": "x"}),
        ("cancel_request", {"request_id": 5}),
    ],
)
async def test_hr_tools_are_denied_for_employee_role(
    db_path: Path, tool: str, args: dict[str, Any]
) -> None:
    async with client(db_path, "E1001", "employee") as c:
        assert "HR_ONLY" in await call_error(c, tool, args)


@pytest.mark.anyio
async def test_list_leave_types(db_path: Path) -> None:
    async with client(db_path, "E1001", "employee") as c:
        data = await call(c, "list_leave_types")
    creatable = {t["code"] for t in data["leave_types"] if t["assistant_can_create"]}
    assert len(data["leave_types"]) == 6
    assert creatable == {"ANNUAL", "SICK", "UNPAID"}


@pytest.mark.anyio
async def test_get_my_balance_uses_server_identity(db_path: Path) -> None:
    async with client(db_path, "E1001", "employee") as c:
        data = await call(c, "get_my_balance", {"leave_type": "ANNUAL"})
    assert data["employee_id"] == "E1001"
    assert data["year"] == 2026
    [balance] = data["balances"]
    assert (balance["approved_days"], balance["pending_days"], balance["available_days"]) == (
        15,
        3,
        10,
    )


@pytest.mark.anyio
async def test_propose_confirm_flow_end_to_end(db_path: Path) -> None:
    async with client(db_path, "E1003", "employee") as c:
        proposal = await call(
            c,
            "propose_leave_request",
            {"leave_type": "annual", "start_date": "2026-10-27", "end_date": "2026-10-29"},
        )
        assert proposal["ok"] is True
        assert (proposal["leave_type"], proposal["days"]) == ("ANNUAL", 3)

        first = await call(c, "confirm_leave_request", {"proposal_id": proposal["proposal_id"]})
        again = await call(c, "confirm_leave_request", {"proposal_id": proposal["proposal_id"]})
        assert first["request_id"] == again["request_id"] == 28
        assert again["already_confirmed"] is True

        mine = await call(c, "list_my_requests", {"status": "pending"})
    assert [(r["request_id"], r["created_via"]) for r in mine["requests"]] == [(28, "assistant")]

    async with client(db_path, "E1007", "hr") as hr:
        approved = await call(hr, "approve_request", {"request_id": 28})
    assert approved["status"] == "approved"


@pytest.mark.anyio
async def test_propose_returns_georgian_violations(db_path: Path) -> None:
    async with client(db_path, "E1001", "employee") as c:
        data = await call(
            c,
            "propose_leave_request",
            {"leave_type": "BEREAVEMENT", "start_date": "2026-11-02", "end_date": "2026-11-03"},
        )
    assert data["ok"] is False
    [violation] = data["violations"]
    assert (violation["code"], violation["action"], violation["article"]) == (
        "LEAVE_TYPE_NOT_SUPPORTED",
        "redirect",
        "8.3",
    )
    assert "გლოვის" in violation["message_ka"]


@pytest.mark.anyio
async def test_bad_date_is_a_clean_tool_error(db_path: Path) -> None:
    async with client(db_path, "E1001", "employee") as c:
        message = await call_error(
            c,
            "propose_leave_request",
            {"leave_type": "ANNUAL", "start_date": "23/11/2026", "end_date": "2026-11-27"},
        )
    assert "INVALID_DATE" in message


@pytest.mark.anyio
async def test_hr_can_list_and_see_any_balance(db_path: Path) -> None:
    async with client(db_path, "E1007", "hr") as hr:
        pending = await call(hr, "list_requests", {"status": "pending"})
        balance = await call(hr, "get_balance", {"employee_id": "E1002", "leave_type": "ANNUAL"})
        missing = await call_error(hr, "get_balance", {"employee_id": "E9999"})
    assert [r["request_id"] for r in pending["requests"]] == [5, 15]
    assert balance["balances"][0]["available_days"] == 4
    assert "UNKNOWN_EMPLOYEE" in missing


@pytest.mark.anyio
async def test_real_server_over_stdio(db_path: Path) -> None:
    """Start the actual entry point as a subprocess, like an MCP client would."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.mcp_server", "--employee", "E1001", "--role", "employee"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "DB_PATH": str(db_path), "APP_TODAY": "2026-10-19"},
    )
    async with Client(params) as c:
        data = await call(c, "get_my_balance", {"leave_type": "SICK"})
    assert data["balances"][0]["available_days"] == 8
