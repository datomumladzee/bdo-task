"""Minimal command-line client for the leave MCP server (no LLM yet).

It starts the MCP server as a subprocess and talks to it over stdio, like any
MCP client. The CLI always runs with the employee role:

    uv run python -m assistant.cli --employee E1001
"""

import argparse
import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BALANCE_COMMANDS = {"ბალანსი", "balance"}
HELP_COMMANDS = {"დახმარება", "help", "?"}
EXIT_COMMANDS = {"გასვლა", "exit", "quit"}

HELP_TEXT = (
    "ბრძანებები:\n"
    "  ბალანსი     შვებულების ბალანსის ნახვა\n"
    "  დახმარება   ეს ტექსტი\n"
    "  გასვლა      პროგრამიდან გასვლა"
)

ReadLine = Callable[[str], Awaitable[str]]
Write = Callable[[str], None]


class ToolCallError(Exception):
    pass


async def call_tool(client: Client, name: str, args: dict[str, Any] | None = None) -> Any:
    """Call an MCP tool and return its result as Python data."""
    result = await client.call_tool(name, args or {})
    text = result.content[0].text if result.content else ""
    if result.is_error:
        raise ToolCallError(text)
    return json.loads(text)


def format_balances(balances: list[dict[str, Any]], type_names: dict[str, str]) -> str:
    """One line per leave type, with approved and pending days (Policy Article 5.2)."""
    if not balances:
        return "ბალანსი ვერ მოიძებნა."
    lines = []
    for b in balances:
        name = type_names.get(b["leave_type"], b["leave_type"])
        lines.append(
            f"{name}: ხელმისაწვდომია {b['available_days']} დღე "
            f"(დამტკიცებული {b['approved_days']}, განხილვის პროცესში {b['pending_days']})"
        )
    return "\n".join(lines)


async def handle(client: Client, line: str, type_names: dict[str, str]) -> str | None:
    """Return the reply to one line of input, or None to exit."""
    command = line.strip().lower()
    if not command:
        return ""
    if command in EXIT_COMMANDS:
        return None
    if command in HELP_COMMANDS:
        return HELP_TEXT
    if command in BALANCE_COMMANDS:
        data = await call_tool(client, "get_my_balance")
        return f"თქვენი ბალანსი, {data['year']}:\n" + format_balances(data["balances"], type_names)
    return "ბრძანება ვერ ვიცანი. აკრიფეთ „დახმარება“."


async def repl(client: Client, read_line: ReadLine, write: Write) -> None:
    """Read-eval-print loop: read a line, answer it, repeat until exit."""
    types = await call_tool(client, "list_leave_types")
    type_names = {t["code"]: t["name"] for t in types["leave_types"]}
    write("გამარჯობა! Northstar Services HR ასისტენტი. აკრიფეთ „დახმარება“.")
    while True:
        try:
            line = await read_line("> ")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            reply = await handle(client, line, type_names)
        except ToolCallError as exc:
            reply = f"შეცდომა: {exc}"
        if reply is None:
            break
        if reply:
            write(reply)
    write("ნახვამდის!")


async def _read_stdin(prompt: str) -> str:
    # input() blocks, so run it in a worker thread to keep the event loop free.
    return await anyio.to_thread.run_sync(input, prompt)


def server_params(employee_id: str, conversation_id: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "server.mcp_server",
            "--employee",
            employee_id,
            "--role",
            "employee",
            "--conversation",
            conversation_id,
        ],
        cwd=str(PROJECT_ROOT),
        env=dict(os.environ),
    )


async def run(employee_id: str) -> None:
    params = server_params(employee_id, conversation_id=uuid.uuid4().hex)
    async with Client(params) as client:
        await repl(client, _read_stdin, print)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Northstar Services HR assistant (CLI)")
    parser.add_argument("--employee", required=True, help="your employee id, e.g. E1001")
    args = parser.parse_args(argv)
    try:
        anyio.run(run, args.employee)
    except* KeyboardInterrupt:
        print("\nნახვამდის!")
    except* MCPError:
        # The server prints the reason (e.g. unknown employee) to stderr before exiting.
        print("MCP სერვერთან კავშირი ვერ დამყარდა. შეამოწმეთ თანამშრომლის ID.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
