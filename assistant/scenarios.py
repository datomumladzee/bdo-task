"""End-to-end scenarios with the real LLM, the real index and the MCP server.

Each scenario runs in a fresh temporary database (APP_TODAY 2026-10-19), prints
the conversation with the tool calls, and checks the database and a few key
phrases. The LLM's wording varies, so read the transcripts too.

    uv run python -m assistant.scenarios            # all scenarios
    uv run python -m assistant.scenarios carry_over # one scenario
"""

import argparse
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import anyio
from dotenv import load_dotenv
from mcp import Client

from assistant.agent import Agent, OpenAIChatModel
from rag.index import load_index
from server.db import get_connection, init_db
from server.mcp_server import build_server
from server.service import Identity

TODAY = date(2026, 10, 19)
NOW = datetime.fromisoformat("2026-10-19T10:00:00")
CSV_REQUESTS = 27

Check = Callable[[list[str], sqlite3.Connection], str | None]  # None = passed


def reply_contains(turn: int, *options: str) -> Check:
    def check(replies: list[str], conn: sqlite3.Connection) -> str | None:
        if any(option in replies[turn] for option in options):
            return None
        return f"reply {turn + 1} mentions none of {options}"

    return check


def reply_lacks(turn: int, text: str) -> Check:
    def check(replies: list[str], conn: sqlite3.Connection) -> str | None:
        return f"reply {turn + 1} contains {text!r}" if text in replies[turn] else None

    return check


def new_requests(count: int) -> Check:
    def check(replies: list[str], conn: sqlite3.Connection) -> str | None:
        created = conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] - CSV_REQUESTS
        return None if created == count else f"{created} new requests, expected {count}"

    return check


def request_is(leave_type: str, start: str, end: str, days: int) -> Check:
    def check(replies: list[str], conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT * FROM leave_requests WHERE request_id > ? ORDER BY request_id",
            (CSV_REQUESTS,),
        ).fetchone()
        if row is None:
            return "no request was created"
        actual = (row["leave_type"], row["start_date"], row["end_date"], row["days"])
        expected = (leave_type, start, end, days)
        if actual != expected or (row["status"], row["created_via"]) != ("pending", "assistant"):
            return f"created {actual} {row['status']}/{row['created_via']}, expected {expected}"
        return None

    return check


@dataclass
class Scenario:
    name: str
    employee: str
    turns: list[str]
    checks: list[Check] = field(default_factory=list)


SCENARIOS = [
    Scenario(
        "balance",
        "E1001",
        ["რამდენი დღე ყოველწლიური შვებულება დამრჩა?"],
        [reply_contains(0, "10"), new_requests(0)],
    ),
    Scenario(
        "carry_over",
        "E1001",
        ["რამდენი დღე გადადის მომდევნო წელზე გამოუყენებელი შვებულებიდან?"],
        [reply_contains(0, "5"), reply_contains(0, "4.7"), reply_contains(0, "მოძველებ")],
    ),
    Scenario(
        "leave_request",
        "E1001",
        ["მინდა ყოველწლიური შვებულება 23-დან 27 ნოემბრამდე", "კი"],
        [reply_contains(0, "4"), request_is("ANNUAL", "2026-11-23", "2026-11-27", 4)],
    ),
    Scenario(
        "bereavement",
        "E1001",
        ["ბებია გარდამეცვალა, შვებულება მჭირდება"],
        [
            reply_contains(0, "1"),
            reply_contains(0, "პორტალ", "HR", "ადამიანური რესურსების"),
            new_requests(0),
        ],
    ),
    Scenario(
        "other_employee",
        "E1001",
        ["რამდენი დღე შვებულება დარჩა E1004-ს?"],
        [reply_lacks(0, "8 დღე"), new_requests(0)],
    ),
    Scenario(
        "not_found",
        "E1001",
        ["შემიძლია ოფისში ძაღლის მოყვანა?"],
        [reply_contains(0, "ვერ ვიპოვე", "ვერ მოიძებნა", "არ არის მოცემული", "არ მოიპოვება")],
    ),
    Scenario(
        "fitness",
        "E1001",
        ["ფიტნესის აბონემენტს აფინანსებს კომპანია?"],
        [reply_contains(0, "4.3")],
    ),
    Scenario(
        "probation",
        "E1004",
        ["მინდა ყოველწლიური შვებულება 9-დან 11 ნოემბრამდე"],
        [reply_contains(0, "გამოსაცდელ"), new_requests(0)],
    ),
    Scenario(
        "sick_over_limit",
        "E1001",
        ["ავად ვარ. ავადმყოფობის შვებულება მჭირდება დღეიდან 29 ოქტომბრის ჩათვლით"],
        [
            reply_contains(0, "HR", "ადამიანური რესურსების"),
            reply_lacks(0, "აღარ შეიძლება"),
            new_requests(0),
        ],
    ),
    Scenario(
        "unpaid_reason",
        "E1003",
        ["მინდა უხელფასო შვებულება 2-დან 6 ნოემბრამდე"],
        [reply_contains(0, "მიზეზ"), new_requests(0)],
    ),
    Scenario(
        "study_acca",
        "E1001",
        ["ACCA ჩემს სწავლის გეგმაშია, გამოცდისთვის ერთი დღე მჭირდება 20 ნოემბერს"],
        [reply_contains(0, "HR", "პორტალ"), new_requests(0)],
    ),
    Scenario(
        "yes_twice",
        "E1001",
        ["მინდა ყოველწლიური შვებულება 23-დან 27 ნოემბრამდე", "კი", "კი"],
        [new_requests(1)],
    ),
]


async def run_scenario(scenario: Scenario, model: OpenAIChatModel, verbose: bool) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "scenario.db"
        init_db(db_path)
        server = build_server(
            Identity(scenario.employee, "employee"),
            db_path=db_path,
            conversation_id=f"scenario-{scenario.name}",
            today_fn=lambda: TODAY,
            now_fn=lambda: NOW,
        )
        index = load_index()
        log: list[str] = []
        async with Client(server) as client:
            agent = Agent(client, model, index.search, scenario.employee, TODAY, log=log.append)
            await agent.start()
            replies = []
            for turn in scenario.turns:
                log.append(f"👤 {turn}")
                reply = await agent.ask(turn)
                log.append(f"🤖 {reply}")
                replies.append(reply)
        conn = get_connection(db_path)
        try:
            failures = [f for check in scenario.checks if (f := check(replies, conn))]
        finally:
            conn.close()
    if verbose:
        print("\n".join(log))
    return failures


async def main_async(names: list[str], verbose: bool) -> int:
    model = OpenAIChatModel()
    chosen = [s for s in SCENARIOS if not names or s.name in names]
    failed = 0
    for scenario in chosen:
        print(f"\n===== {scenario.name} ({scenario.employee}) =====")
        failures = await run_scenario(scenario, model, verbose)
        failed += bool(failures)
        print("PASS" if not failures else "FAIL: " + "; ".join(failures))
    print(f"\n{len(chosen) - failed}/{len(chosen)} scenarios passed (model {model.model})")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the assistant scenarios with the real LLM")
    parser.add_argument("names", nargs="*", help="scenario names (default: all)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print pass/fail")
    args = parser.parse_args(argv)
    load_dotenv()
    raise SystemExit(anyio.run(main_async, args.names, not args.quiet))


if __name__ == "__main__":
    main()
