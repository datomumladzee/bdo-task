"""Georgian command-line HR assistant.

It starts the MCP server as a subprocess (employee role, one conversation id
per session) and talks to it over stdio. Free text goes to the LLM agent, which
answers policy questions from the documents, shows the balance and creates
leave requests after a clear confirmation.

    uv run python -m assistant.cli --employee E1001
    uv run python -m assistant.cli --employee E1001 --debug   # show tool calls
"""

import argparse
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from openai import OpenAIError

from assistant.agent import Agent, OpenAIChatModel
from rag.index import load_index
from server.calendar import today

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXIT_COMMANDS = {"გასვლა", "exit", "quit"}
HELP_COMMANDS = {"დახმარება", "help", "?"}

GREETING = (
    "გამარჯობა! მე ვარ Northstar Services-ის HR ასისტენტი. შემიძლია ვუპასუხო კითხვებს "
    "კომპანიის წესებზე, გაჩვენოთ შვებულების ბალანსი და შევქმნა შვებულების მოთხოვნა.\n"
    "გასასვლელად აკრიფეთ „გასვლა“."
)
HELP_TEXT = (
    "მაგალითები:\n"
    "  რამდენი დღე შვებულება დამრჩა?\n"
    "  რამდენი დღე გადადის მომდევნო წელზე?\n"
    "  მინდა შვებულება 23-დან 27 ნოემბრამდე\n"
    "გასასვლელად: გასვლა"
)

ReadLine = Callable[[str], Awaitable[str]]
Write = Callable[[str], None]


async def repl(agent: Agent, read_line: ReadLine, write: Write) -> None:
    """Read a line, answer it, repeat until the employee leaves."""
    write(GREETING)
    while True:
        try:
            line = (await read_line("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in EXIT_COMMANDS:
            break
        if line.lower() in HELP_COMMANDS:
            write(HELP_TEXT)
            continue
        try:
            write(await agent.ask(line))
        except (OpenAIError, MCPError) as exc:  # keep the session alive on API errors
            write(f"შეცდომა: {exc}")
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


async def run(employee_id: str, debug: bool) -> None:
    index = load_index()
    params = server_params(employee_id, conversation_id=uuid.uuid4().hex)
    async with Client(params) as client:
        agent = Agent(
            client=client,
            model=OpenAIChatModel(),
            search=index.search,
            employee_id=employee_id,
            today=today(),
            log=(lambda line: print(line, file=sys.stderr)) if debug else None,
        )
        await agent.start()
        await repl(agent, _read_stdin, print)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Northstar Services HR assistant (CLI)")
    parser.add_argument("--employee", required=True, help="your employee id, e.g. E1001")
    parser.add_argument("--debug", action="store_true", help="print tool calls to stderr")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        parser.exit(1, "error: OPENAI_API_KEY is not set; add it to .env (see .env.example)\n")
    try:
        anyio.run(run, args.employee, args.debug)
    except* KeyboardInterrupt:
        print("\nნახვამდის!")
    except* MCPError:
        # The server prints the reason (e.g. unknown employee) to stderr before exiting.
        print("MCP სერვერთან კავშირი ვერ დამყარდა. შეამოწმეთ თანამშრომლის ID.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
