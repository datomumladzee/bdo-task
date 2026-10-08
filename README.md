# Northstar Services HR Leave Assistant

An MCP server for employee leave management (Task 1) and a Georgian command-line
assistant that uses it and answers policy questions from the company documents
with RAG (Task 2).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An **OpenAI API key** (chat model and embeddings) for the assistant and the
  document index. Task 1 (database, MCP server, tests) runs without it.

## Setup

```bash
uv sync
cp .env.example .env
```

`.env` settings:

| Variable | Purpose |
|---|---|
| `APP_TODAY` | The date the app treats as today. Use `2026-10-19`, the date of the data snapshot. |
| `DB_PATH` | SQLite database file. Default: `leave.db` in the project root. |
| `OPENAI_API_KEY` | OpenAI API key. Required for the assistant and the document index. |
| `OPENAI_MODEL` | Chat model. Default: `gpt-5.4`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model. Default: `text-embedding-3-large`. |
| `RAG_MIN_SCORE` | "Not found" threshold for document search. Default: `0.22`. |

## Build the database

```bash
uv run python scripts/setup_db.py
```

This creates the tables and loads `data/*.csv`. Running it again does not
duplicate rows. The MCP server also builds the database on startup if it is missing.

### Reset the database

```bash
rm leave.db
uv run python scripts/setup_db.py
```

This rebuilds the database from the CSVs and removes any requests created while
testing.

## Run the MCP server

The caller's identity and role are fixed when the server starts, not passed
as tool arguments, so the assistant cannot act for another employee.

```bash
# employee assistant
uv run python -m server.mcp_server --employee E1001 --role employee

# HR
uv run python -m server.mcp_server --employee E1007 --role hr
```

The server uses the stdio transport: after starting it prints nothing and waits
for an MCP client on standard input. That is expected. To use it, connect a
client: the assistant CLI below, or MCP Inspector in a browser.

### Try the tools in a browser (MCP Inspector)

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the
official testing UI for MCP servers. It needs Node.js 18+ (for `npx`). From the
project root:

```bash
npx @modelcontextprotocol/inspector --config mcp-inspector.json
```

It opens a page at `http://127.0.0.1:6274`. Pick a server from
`mcp-inspector.json`, connect, open **Tools**, and call any tool with its
parameters. The file defines three identities:

| Server | Who | Try |
|---|---|---|
| `employee-E1001` | Employee (Audit) | `get_my_balance`; `propose_leave_request` for 2026-11-23 to 2026-11-27 (4 days), then `confirm_leave_request`; `approve_request` returns `HR_ONLY` |
| `employee-E1004-probation` | Employee on probation | `propose_leave_request` ANNUAL in November is refused (Article 4.3) |
| `hr-E1007` | HR | `list_requests` with `status=pending`; `get_balance` for `E1002`; `approve_request` / `reject_request` |

> **Note:** tool calls really change the database (`leave.db`). To undo your
> test requests, see [Reset the database](#reset-the-database).

Without a browser, the Inspector's `--cli` mode calls a tool directly:

```bash
npx @modelcontextprotocol/inspector --cli --config mcp-inspector.json \
  --server hr-E1007 --method tools/call --tool-name get_balance \
  --tool-arg employee_id=E1002 --tool-arg leave_type=ANNUAL
```

The tools:

| Tool | Role | Purpose |
|---|---|---|
| `list_leave_types` | any | The six leave types and which ones the assistant can create |
| `get_my_balance` | any | The caller's balance: entitled, carried over, approved, pending, available |
| `list_my_requests` | any | The caller's requests, filtered by status and date range |
| `propose_leave_request` | any | Validate a request and prepare it for confirmation (nothing is saved yet) |
| `confirm_leave_request` | any | Create the confirmed request as `pending`; repeating it returns the same id |
| `list_requests` | HR | All requests, filtered by employee, status and date range |
| `get_balance` | HR | Any employee's balance |
| `create_request` | HR | Create a pending request for an employee, under the same rules |
| `approve_request` | HR | Approve a pending request |
| `reject_request` | HR | Reject a pending request (reason required) |
| `cancel_request` | HR | Cancel a pending or approved request |

HR tools return an `HR_ONLY` error when the server runs with the employee role.
Only employees of the Human Resources department (`HRS`, e.g. E1007) can start
the server with `--role hr`; anyone else gets an error at startup.

## Build the document index

```bash
uv run python -m rag.index --build
```

This splits the 7 documents into 273 chunks (one per article; tables separately)
and embeds them once. Embeddings are cached in `.index/` and only recomputed
when a chunk's text or the model changes. The assistant builds the index on
first use if it is missing.

## Run the assistant

```bash
uv run python -m assistant.cli --employee E1001
uv run python -m assistant.cli --employee E1001 --debug   # also print tool calls
```

The CLI starts the MCP server itself over stdio, with the employee role and a
new conversation id per session. Example questions:

Example questions:

```
> რამდენი დღე შვებულება დამრჩა?
> რამდენი დღე გადადის მომდევნო წელზე?
> მინდა ყოველწლიური შვებულება 23-დან 27 ნოემბრამდე
> კი
> ბებია გარდამეცვალა, რა ვქნა?
> შემიძლია ოფისში ძაღლის მოყვანა?
> ნიკას ბალანსი მაჩვენე
```

Type `გასვლა` to exit.

### How it works

- **Intent.** The LLM gets `search_policies` (local RAG) and the five employee
  MCP tools. Its choice of tool is the intent: a policy question goes to
  `search_policies`, a balance question to `get_my_balance`, a leave request
  to `propose_leave_request` with the leave type it recognised. BEREAVEMENT,
  STUDY and PARENTAL are explained and redirected, never created.
- **RAG.** Documents are split per article (Word heading styles for DOCX; bold
  numbered headings for PDF, with page headers and footers removed). Search is
  hybrid: BM25 over character 3-grams (Georgian words change their endings)
  plus embedding similarity. Outdated FAQ and Handbook sections rank lower and
  are returned separately, so answers cite the current policy and point out
  what is outdated (Leave Policy 1.4, Remote Work Policy 1.3). Below the
  `RAG_MIN_SCORE` threshold the assistant says the documents do not cover it.
- **Confirmation.** A request is only created after the employee's clear "yes"
  (Policy Article 12.2). This is enforced in code, not only in the prompt:
  `confirm_leave_request` is refused unless the proposal was shown in an
  earlier turn and the employee's latest message is an explicit yes. Repeated
  confirmation returns the same request id.

## Evaluate

```bash
uv run python -m rag.evaluate         # retrieval: 17 questions + 3 unanswerable
uv run python -m assistant.scenarios  # 12 end-to-end conversations with the LLM
```

Retrieval on the eval set: hit@1 12/17, hit@5 17/17, MRR 0.81, "not found"
3/3 (embeddings alone: hit@5 1/17). The scenarios run against a fresh
temporary database each and check the database and key phrases; they call the
OpenAI API.

## Run the tests

```bash
uv run pytest
uv run ruff check .
```

## Project layout

```
server/db.py          schema and CSV import
server/calendar.py    working-day counting, APP_TODAY
server/balance.py     balance formula (Policy Article 5.1)
server/rules.py       request validation (Policy Article 12.3)
server/service.py     leave actions and role checks
server/mcp_server.py  MCP server and tools
rag/ingest.py         document parsing and chunking
rag/index.py          hybrid search (BM25 + embeddings) with cache
rag/evaluate.py       retrieval evaluation (rag/eval_set.json)
assistant/agent.py    LLM function-calling loop and confirmation guard
assistant/prompts.py  Georgian system prompt
assistant/cli.py      command-line assistant
assistant/scenarios.py end-to-end scenarios with the real LLM
scripts/setup_db.py   build the database
tests/                pytest tests
```

## Notes

- The data dictionary says CSV dates are ISO (`YYYY-MM-DD`), but the provided
  CSVs use `M/D/YYYY`. The CSVs are not modified; the importer accepts both
  formats and stores ISO dates.
- For employees, "create leave request" is split into `propose_leave_request` and
  `confirm_leave_request`, because Policy Article 12.2 requires showing the
  details and getting a clear confirmation first. The employee comes from the
  server's identity. HR uses `create_request` with an explicit `employee_id`.

## Simplifications

- **One `hr` role makes all decisions.** In the policy, the direct manager
  approves annual leave (Article 4.4) and unpaid leave needs both the manager
  and HR (Article 7.3). Here a single HR role approves, rejects and cancels.
- **The Article 4.8 cancellation limit is not enforced for HR.** The policy lets
  employees cancel approved leave up to 2 working days before it starts; later
  needs the manager's consent. HR can cancel any pending or approved request.
- **The HR role comes from the department.** Anyone in `HRS` may run the server
  as HR; there is no separate permission list.
