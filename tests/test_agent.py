"""Agent loop and confirmation guard, with a scripted fake LLM (no API calls)."""

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from assistant.agent import (
    EMPLOYEE_TOOLS,
    MAX_TOOL_ROUNDS,
    Agent,
    ChatTurn,
    ToolCall,
    is_clear_yes,
)
from rag.index import PolicySearch, SearchResult
from rag.ingest import Chunk
from server.db import get_connection
from server.mcp_server import build_server
from server.service import Identity

TODAY = date(2026, 10, 19)
NOW = datetime.fromisoformat("2026-10-19T10:00:00")
PROPOSE_23_27 = {"leave_type": "ANNUAL", "start_date": "2026-11-23", "end_date": "2026-11-27"}

Step = ChatTurn | Callable[[list[dict[str, Any]]], ChatTurn]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeModel:
    """Returns scripted turns in order; a step may be a function of the messages."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.seen_tools: list[str] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        self.seen_tools = [t["function"]["name"] for t in tools]
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def say(text: str) -> ChatTurn:
    return ChatTurn(text, [])


def call(name: str, **arguments: Any) -> ChatTurn:
    return ChatTurn(None, [ToolCall(f"call-{name}", name, arguments)])


def last_tool_result(messages: list[dict[str, Any]]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "tool")


def confirm_last_proposal(messages: list[dict[str, Any]]) -> ChatTurn:
    """Call confirm with the proposal_id from the most recent propose result."""
    import json

    for m in reversed(messages):
        if m["role"] == "tool" and "proposal_id" in m["content"]:
            proposal_id = json.loads(m["content"])["proposal_id"]
            return call("confirm_leave_request", proposal_id=proposal_id)
    raise AssertionError("no proposal in the conversation")


def fake_search(query: str) -> PolicySearch:
    chunk = Chunk(
        chunk_id="HR-POL-02:4.7:1",
        file="Leave.docx",
        doc_code="HR-POL-02",
        doc_title="შვებულებისა და გაცდენის პოლიტიკა",
        version="4.0",
        effective="2026",
        status="მოქმედი",
        doc_type="policy",
        article="4.7",
        section="4. ყოველწლიური ანაზღაურებადი შვებულება",
        heading="4.7 მომდევნო წელზე გადატანა",
        kind="text",
        page=None,
        superseded_note=None,
        body="არაუმეტეს 5 სამუშაო დღე",
    )
    found = "გადადის" in query
    faq = replace(
        chunk,
        chunk_id="HR-FAQ-01:ა.3:4",
        doc_code="HR-FAQ-01",
        article="ა.3",
        superseded_note="მოძველებულია",
        body="არაუმეტეს 10 დღე",
    )
    results = [SearchResult(chunk, 0.9, 0.5, 0.4)]
    outdated = [SearchResult(faq, 0.5, 0.4, 0.3)]
    return PolicySearch(query, results, 0.5 if found else 0.1, 0.22, outdated)


def make_agent(db_path: Path, model: FakeModel) -> Agent:
    server = build_server(
        Identity("E1001", "employee"),
        db_path=db_path,
        conversation_id="conv-1",
        today_fn=lambda: TODAY,
        now_fn=lambda: NOW,
    )
    return Agent(
        client=Client(server),
        model=model,
        search=fake_search,
        employee_id="E1001",
        today=TODAY,
    )


def request_count(db_path: Path) -> int:
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("text", ["კი", "დიახ", "კი, დაადასტურე", "yes", "Ok"])
def test_clear_yes(text: str) -> None:
    assert is_clear_yes(text)


@pytest.mark.parametrize("text", ["არა", "კი, მაგრამ 24-დან", "კიდევ ერთი კითხვა", "კი?", "no", ""])
def test_not_a_clear_yes(text: str) -> None:
    assert not is_clear_yes(text)


@pytest.mark.anyio
async def test_only_search_and_employee_tools_are_offered(db_path: Path) -> None:
    model = FakeModel([say("გამარჯობა")])
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("გამარჯობა")
    assert set(model.seen_tools) == {"search_policies", *EMPLOYEE_TOOLS}
    assert "approve_request" not in model.seen_tools


@pytest.mark.anyio
async def test_system_prompt_has_today_and_employee(db_path: Path) -> None:
    agent = make_agent(db_path, FakeModel([]))
    async with agent.client:
        await agent.start()
    assert "2026-10-19 (ორშაბათი)" in agent.messages[0]["content"]
    assert "E1001" in agent.messages[0]["content"]


@pytest.mark.anyio
async def test_policy_search_results_reach_the_model(db_path: Path) -> None:
    seen: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        seen.append(last_tool_result(messages))
        return say("5 დღე (მუხლი 4.7)")

    model = FakeModel([call("search_policies", query="რამდენი დღე გადადის?"), answer])
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        reply = await agent.ask("რამდენი დღე გადადის მომდევნო წელზე?")
    assert reply == "5 დღე (მუხლი 4.7)"
    assert '"found": true' in seen[0]
    assert "HR-POL-02, ვერსია 4.0), მუხლი 4.7" in seen[0]
    assert '"outdated_passages"' in seen[0]
    assert "HR-FAQ-01" in seen[0]


@pytest.mark.anyio
async def test_not_found_search_hides_passages(db_path: Path) -> None:
    seen: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        seen.append(last_tool_result(messages))
        return say("ვერ ვიპოვე")

    model = FakeModel([call("search_policies", query="ძაღლი ოფისში"), answer])
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("შემიძლია ძაღლის მოყვანა?")
    assert '"found": false' in seen[0]
    assert "passages" not in seen[0]


@pytest.mark.anyio
async def test_balance_comes_from_mcp(db_path: Path) -> None:
    seen: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        seen.append(last_tool_result(messages))
        return say("10 დღე")

    model = FakeModel([call("get_my_balance", leave_type="ANNUAL"), answer])
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("რამდენი დღე დამრჩა?")
    assert '"available_days": 10' in seen[0]


@pytest.mark.anyio
async def test_confirm_in_the_same_turn_as_propose_is_refused(db_path: Path) -> None:
    seen: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        seen.append(last_tool_result(messages))
        return say("...")

    model = FakeModel(
        [call("propose_leave_request", **PROPOSE_23_27), confirm_last_proposal, answer]
    )
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("კი, მინდა შვებულება 23-დან 27 ნოემბრამდე")
    assert "CONFIRMATION_REQUIRED" in seen[0]
    assert request_count(db_path) == 27


@pytest.mark.anyio
async def test_confirm_needs_a_clear_yes(db_path: Path) -> None:
    seen: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        seen.append(last_tool_result(messages))
        return say("...")

    model = FakeModel(
        [
            call("propose_leave_request", **PROPOSE_23_27),
            say("4 დღე. ადასტურებთ?"),
            confirm_last_proposal,
            answer,
        ]
    )
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("მინდა შვებულება 23-დან 27 ნოემბრამდე")
        await agent.ask("არ ვიცი, მაგრამ შეიძლება")
    assert "CONFIRMATION_REQUIRED" in seen[0]
    assert request_count(db_path) == 27


@pytest.mark.anyio
async def test_propose_then_yes_twice_creates_one_request(db_path: Path) -> None:
    confirmations: list[str] = []

    def answer(messages: list[dict[str, Any]]) -> ChatTurn:
        confirmations.append(last_tool_result(messages))
        return say("მოთხოვნა შექმნილია")

    model = FakeModel(
        [
            call("propose_leave_request", **PROPOSE_23_27),
            say("ANNUAL, 2026-11-23 – 2026-11-27, 4 დღე. ადასტურებთ?"),
            confirm_last_proposal,
            answer,
            confirm_last_proposal,
            answer,
        ]
    )
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        await agent.ask("მინდა შვებულება 23-დან 27 ნოემბრამდე")
        await agent.ask("კი")
        await agent.ask("კი")
    assert '"request_id": 28' in confirmations[0]
    assert '"request_id": 28' in confirmations[1]
    assert '"already_confirmed": true' in confirmations[1]
    assert request_count(db_path) == 28
    conn: sqlite3.Connection = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM leave_requests WHERE request_id = 28").fetchone()
    finally:
        conn.close()
    assert (row["status"], row["created_via"], row["days"]) == ("pending", "assistant", 4)


@pytest.mark.anyio
async def test_tool_loop_stops_after_max_rounds(db_path: Path) -> None:
    model = FakeModel([call("list_leave_types")] * MAX_TOOL_ROUNDS)
    agent = make_agent(db_path, model)
    async with agent.client:
        await agent.start()
        reply = await agent.ask("?")
    assert "ვერ მოხერხდა" in reply
