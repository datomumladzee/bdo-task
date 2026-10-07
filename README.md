# Northstar Services HR Leave Assistant

An MCP server for employee leave management (Task 1) and a Georgian command-line
assistant that uses it (Task 2, in progress).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

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
| `OPENAI_API_KEY` | OpenAI API key. Not needed yet; the LLM assistant will use it. |

## Build the database

```bash
uv run python scripts/setup_db.py
```

This creates the tables and loads `data/*.csv`. Running it again does not
duplicate rows. The MCP server also builds the database on startup if it is missing.

## Run the MCP server

The caller's identity and role are fixed when the server starts, not passed
as tool arguments, so the assistant cannot act for another employee.

```bash
# employee assistant
uv run python -m server.mcp_server --employee E1001 --role employee

# HR
uv run python -m server.mcp_server --employee E1007 --role hr
```

The server uses the stdio transport. Tools:

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

## Run the CLI

```bash
uv run python -m assistant.cli --employee E1001
```

The CLI starts the MCP server itself over stdio. For now it supports
`ბალანსი` (balance), `დახმარება` (help) and `გასვლა` (exit).

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
assistant/cli.py      command-line client
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
