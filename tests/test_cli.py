from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import pytest
from mcp import Client

from assistant.cli import format_balances, repl
from server.mcp_server import build_server
from server.service import Identity

TODAY = date(2026, 10, 19)
NOW = datetime.fromisoformat("2026-10-19T10:00:00")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def run_session(db_path: Path, lines: list[str]) -> str:
    """Run the CLI loop against an in-process server with scripted input."""
    server = build_server(
        Identity("E1001", "employee"),
        db_path=db_path,
        conversation_id="conv-1",
        today_fn=lambda: TODAY,
        now_fn=lambda: NOW,
    )
    inputs: Iterator[str] = iter(lines)
    output: list[str] = []

    async def read_line(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    async with Client(server) as client:
        await repl(client, read_line, output.append)
    return "\n".join(output)


@pytest.mark.anyio
async def test_balance_command_shows_own_balance(db_path: Path) -> None:
    out = await run_session(db_path, ["ბალანსი", "გასვლა"])
    assert "ყოველწლიური ანაზღაურებადი შვებულება: ხელმისაწვდომია 10 დღე" in out
    assert "დამტკიცებული 15, განხილვის პროცესში 3" in out
    assert "ავადმყოფობის შვებულება: ხელმისაწვდომია 8 დღე" in out
    assert out.endswith("ნახვამდის!")


@pytest.mark.anyio
async def test_english_alias_and_unknown_command(db_path: Path) -> None:
    out = await run_session(db_path, ["BALANCE", "რამე", "exit"])
    assert "ხელმისაწვდომია 10 დღე" in out
    assert "ბრძანება ვერ ვიცანი" in out


@pytest.mark.anyio
async def test_end_of_input_exits_cleanly(db_path: Path) -> None:
    out = await run_session(db_path, ["დახმარება"])
    assert "ბრძანებები:" in out
    assert out.endswith("ნახვამდის!")


def test_format_balances_without_rows() -> None:
    assert format_balances([], {}) == "ბალანსი ვერ მოიძებნა."
