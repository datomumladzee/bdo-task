"""The Georgian HR assistant: an LLM with function calling.

The LLM gets two kinds of tools:
- search_policies: local RAG search over the policy documents (rag/index.py);
- the employee tools of the MCP server (balance, requests, propose/confirm).

The LLM's choice of tool is the intent detection: a policy question goes to
search_policies, a balance question to get_my_balance, a leave request to
propose_leave_request with the leave type it recognised.

Creating a request needs a clear "yes" from the employee (Policy Article 12.2;
the data dictionary says conversation history alone is not permission). This is
enforced in code, not only in the prompt: confirm_leave_request goes through
only if the proposal was shown to the employee in an earlier turn and the
employee's latest message is an explicit yes.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import anyio
from mcp import Client

from assistant.prompts import SYSTEM_PROMPT, WEEKDAYS_KA
from rag.index import PolicySearch, SearchResult

EMPLOYEE_TOOLS = (
    "list_leave_types",
    "get_my_balance",
    "list_my_requests",
    "propose_leave_request",
    "confirm_leave_request",
)
DEFAULT_CHAT_MODEL = "gpt-5.4"
MAX_TOOL_ROUNDS = 8

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_policies",
        "description": (
            "Search the company's policy documents (leave, remote work, travel and "
            "expenses, information security, learning, handbook, FAQ). Use it for every "
            "question about rules. Returns passages with citations, or found=false when "
            "the documents do not cover the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "2-6 Georgian key words for the topic, not a full sentence, e.g. "
                        "'გლოვის შვებულება ბებია' or 'შვებულების გადატანა მომდევნო წელზე'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

# A clear yes: starts with a yes-word and has no "no" or "but" in it.
_YES = re.compile(
    r"^\s*(კი|დიახ|ჰო|ki|diax|diakh|yes|ok|okay|დავადასტურე|ვადასტურებ|ვეთანხმები)(?![ა-ჰa-z])",
    re.IGNORECASE,
)
_NOT_CLEAR = re.compile(r"(არა|მაგრამ|ოღონდ|\bno\b|\bbut\b|\?)", re.IGNORECASE)


def is_clear_yes(message: str) -> bool:
    return bool(_YES.match(message)) and not _NOT_CLEAR.search(message)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCall]


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn: ...


class OpenAIChatModel:
    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
        self._client = OpenAI()

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, tools=tools
        )
        message = response.choices[0].message
        calls = [
            ToolCall(call.id, call.function.name, json.loads(call.function.arguments or "{}"))
            for call in message.tool_calls or []
        ]
        return ChatTurn(message.content, calls)


SearchFn = Callable[[str], PolicySearch]
LogFn = Callable[[str], None]


def _search_result(search: PolicySearch) -> dict[str, Any]:
    """What the LLM sees from a policy search. Nothing at all when not found."""
    if not search.found:
        return {
            "found": False,
            "message": (
                "Nothing relevant found. If the query was a long sentence, search once more "
                "with 2-4 key words. If that also finds nothing, tell the employee the "
                "documents do not cover this."
            ),
        }

    def passage(r: SearchResult) -> dict[str, Any]:
        return {
            "citation": r.chunk.citation(),
            "status": r.chunk.status,
            "superseded_note": r.chunk.superseded_note,
            "text": r.chunk.text,
        }

    result: dict[str, Any] = {
        "found": True,
        "passages": [passage(r) for r in search.results],
    }
    if search.outdated:
        # Older FAQ/Handbook statements on the same topic. Never answer from these;
        # use them only to point out what is outdated.
        result["outdated_passages"] = [passage(r) for r in search.outdated]
    return result


@dataclass
class Agent:
    client: Client
    model: ChatModel
    search: SearchFn
    employee_id: str
    today: date
    log: LogFn | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    _shown_proposals: set[str] = field(default_factory=set)  # shown before the latest message
    _new_proposals: set[str] = field(default_factory=set)  # proposed in the current turn
    _last_user_message: str = ""

    async def start(self) -> None:
        listed = await self.client.list_tools()
        self.tools = [SEARCH_TOOL] + [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in listed.tools
            if tool.name in EMPLOYEE_TOOLS
        ]
        prompt = SYSTEM_PROMPT.format(
            today=self.today.isoformat(),
            weekday=WEEKDAYS_KA[self.today.weekday()],
            employee_id=self.employee_id,
        )
        self.messages = [{"role": "system", "content": prompt}]

    async def ask(self, text: str) -> str:
        """One employee message in, one assistant reply out (with tool calls in between)."""
        self._shown_proposals |= self._new_proposals
        self._new_proposals = set()
        self._last_user_message = text
        self.messages.append({"role": "user", "content": text})
        for _ in range(MAX_TOOL_ROUNDS):
            turn = await anyio.to_thread.run_sync(self.model.complete, self.messages, self.tools)
            if not turn.tool_calls:
                reply = turn.content or ""
                self.messages.append({"role": "assistant", "content": reply})
                return reply
            self.messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                result = await self._run_tool(call)
                if self.log:
                    self.log(
                        f"[tool] {call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                    )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        reply = "ბოდიში, მოთხოვნის დამუშავება ვერ მოხერხდა. სცადეთ თავიდან ან მიმართეთ HR-ს."
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    async def _run_tool(self, call: ToolCall) -> dict[str, Any]:
        if call.name == "search_policies":
            query = str(call.arguments.get("query", ""))
            return _search_result(await anyio.to_thread.run_sync(self.search, query))
        if call.name not in EMPLOYEE_TOOLS:
            return {"ok": False, "error": f"Unknown tool {call.name}"}
        if call.name == "confirm_leave_request":
            refusal = self._confirmation_refusal(str(call.arguments.get("proposal_id", "")))
            if refusal:
                return refusal
        result = await self.client.call_tool(call.name, call.arguments)
        text = result.content[0].text if result.content else ""
        if result.is_error:
            return {"ok": False, "error": text}
        data = json.loads(text)
        if call.name == "propose_leave_request" and data.get("ok"):
            self._new_proposals.add(data["proposal_id"])
        return data

    def _confirmation_refusal(self, proposal_id: str) -> dict[str, Any] | None:
        """Why confirm is not allowed right now, or None if it is."""
        if proposal_id not in self._shown_proposals:
            return {
                "ok": False,
                "error": "CONFIRMATION_REQUIRED",
                "message": (
                    "Show the employee the leave type, dates and number of days and wait "
                    "for their explicit confirmation in their next message."
                ),
            }
        if not is_clear_yes(self._last_user_message):
            return {
                "ok": False,
                "error": "CONFIRMATION_REQUIRED",
                "message": (
                    "The employee's latest message is not a clear yes. Ask them to confirm "
                    "with 'კი' or to change the request."
                ),
            }
        return None
