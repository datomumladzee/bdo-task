"""SQLite schema and CSV importer for the HR leave database."""

import csv          # reads CSV files
import os           # reads environment variables (.env)
import re           # pattern matching (to recognise dates)
import sqlite3      # talks to the SQLite database
from datetime import date, datetime   # works with dates
from pathlib import Path              # works with file paths

from dotenv import load_dotenv        # loads the .env file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = PROJECT_ROOT / "leave.db"

TABLES = (
    "employees",
    "leave_types",
    "leave_entitlements",
    "public_holidays",
    "leave_requests",
    "leave_proposals",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id        TEXT PRIMARY KEY,
    full_name          TEXT NOT NULL,
    email              TEXT NOT NULL UNIQUE,
    department_code    TEXT NOT NULL,
    department_name    TEXT NOT NULL,
    job_title          TEXT NOT NULL,
    employment_type    TEXT NOT NULL,
    start_date         TEXT NOT NULL,
    probation_end_date TEXT NOT NULL,
    manager_id         TEXT REFERENCES employees (employee_id),
    status             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_types (
    code                TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    day_unit            TEXT CHECK (day_unit IN ('working', 'calendar')),
    annual_limit_days   INTEGER,
    self_service        INTEGER NOT NULL CHECK (self_service IN (0, 1)),
    assistant_supported INTEGER NOT NULL CHECK (assistant_supported IN (0, 1)),
    policy_reference    TEXT
);

CREATE TABLE IF NOT EXISTS leave_entitlements (
    employee_id       TEXT NOT NULL REFERENCES employees (employee_id),
    year              INTEGER NOT NULL,
    leave_type        TEXT NOT NULL REFERENCES leave_types (code),
    entitled_days     INTEGER NOT NULL CHECK (entitled_days >= 0),
    carried_over_days INTEGER NOT NULL DEFAULT 0 CHECK (carried_over_days >= 0),
    PRIMARY KEY (employee_id, year, leave_type)
);

CREATE TABLE IF NOT EXISTS public_holidays (
    date TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_requests (
    request_id       INTEGER PRIMARY KEY,
    employee_id      TEXT NOT NULL REFERENCES employees (employee_id),
    leave_type       TEXT NOT NULL REFERENCES leave_types (code),
    start_date       TEXT NOT NULL,
    end_date         TEXT NOT NULL,
    days             INTEGER NOT NULL CHECK (days > 0),
    status           TEXT NOT NULL
                     CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    created_at       TEXT NOT NULL,
    created_via      TEXT NOT NULL CHECK (created_via IN ('portal', 'assistant')),
    comment          TEXT,
    decided_by       TEXT,
    decided_at       TEXT,
    decision_comment TEXT,
    CHECK (start_date <= end_date)
);

CREATE TABLE IF NOT EXISTS leave_proposals (
    proposal_id     TEXT PRIMARY KEY,
    employee_id     TEXT NOT NULL REFERENCES employees (employee_id),
    conversation_id TEXT NOT NULL,
    payload         TEXT NOT NULL CHECK (json_valid(payload)),
    status          TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed', 'confirmed', 'expired')),
    request_id      INTEGER UNIQUE REFERENCES leave_requests (request_id),
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_employee_status
    ON leave_requests (employee_id, status);
CREATE INDEX IF NOT EXISTS idx_requests_employee_dates
    ON leave_requests (employee_id, start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_proposals_employee_conversation
    ON leave_proposals (employee_id, conversation_id);
"""

Value = str | int | None
Converter = Callable[[str], str | int]

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# The shipped CSVs use US month/day/year, despite the data dictionary saying ISO.
_US_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def get_db_path() -> Path:
    """Return DB_PATH from .env (relative paths resolve against the project root)."""
    load_dotenv(PROJECT_ROOT / ".env")
    raw = os.getenv("DB_PATH")
    if not raw:
        return DEFAULT_DB_PATH
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def parse_date(value: str) -> str:
    """Return an ISO date for YYYY-MM-DD or M/D/YYYY input; reject anything else."""
    value = value.strip()
    try:
        if _ISO_DATE.fullmatch(value):
            return date.fromisoformat(value).isoformat()
        if match := _US_DATE.fullmatch(value):
            month, day, year = (int(part) for part in match.groups())
            return date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}: {exc}") from exc
    raise ValueError(f"Unrecognised date {value!r}: expected YYYY-MM-DD or M/D/YYYY")


def parse_timestamp(value: str) -> str:
    """Validate an ISO 8601 timestamp and return it in canonical form."""
    try:
        return datetime.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 timestamp {value!r}") from exc


def _read_csv(path: Path, converters: dict[str, Converter]) -> list[dict[str, Value]]:
    """Read a CSV, turning empty strings into None and applying per-column converters."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        if headers != set(converters):
            raise ValueError(
                f"{path.name}: unexpected columns "
                f"(missing {sorted(set(converters) - headers)}, "
                f"extra {sorted(headers - set(converters))})"
            )
        rows: list[dict[str, Value]] = []
        for line_no, raw in enumerate(reader, start=2):
            row: dict[str, Value] = {}
            for column, convert in converters.items():
                text = raw[column].strip()
                try:
                    row[column] = convert(text) if text else None
                except ValueError as exc:
                    raise ValueError(f"{path.name} line {line_no}, {column}: {exc}") from exc
            rows.append(row)
        return rows


def _insert(conn: sqlite3.Connection, table: str, rows: list[dict[str, Value]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [tuple(row[c] for c in columns) for row in rows],
    )


def import_csvs(conn: sqlite3.Connection, data_dir: Path = DATA_DIR) -> None:
    """Load all CSVs into an empty schema. The caller owns the transaction."""
    employees = _read_csv(
        data_dir / "employees.csv",
        {
            "employee_id": str,
            "full_name": str,
            "email": str,
            "department_code": str,
            "department_name": str,
            "job_title": str,
            "employment_type": str,
            "start_date": parse_date,
            "probation_end_date": parse_date,
            "manager_id": str,
            "status": str,
        },
    )
    # Managers can appear after their reports, and SQLite checks FKs per row,
    # so insert without managers first and link them once everyone exists.
    _insert(conn, "employees", [{**e, "manager_id": None} for e in employees])
    conn.executemany(
        "UPDATE employees SET manager_id = ? WHERE employee_id = ?",
        [(e["manager_id"], e["employee_id"]) for e in employees if e["manager_id"]],
    )

    _insert(
        conn,
        "leave_types",
        _read_csv(
            data_dir / "leave_types.csv",
            {
                "code": str,
                "name": str,
                "day_unit": str,
                "annual_limit_days": int,
                "self_service": int,
                "assistant_supported": int,
                "policy_reference": str,
            },
        ),
    )
    _insert(
        conn,
        "leave_entitlements",
        _read_csv(
            data_dir / "leave_entitlements.csv",
            {
                "employee_id": str,
                "year": int,
                "leave_type": str,
                "entitled_days": int,
                "carried_over_days": int,
            },
        ),
    )
    _insert(
        conn,
        "public_holidays",
        _read_csv(data_dir / "public_holidays.csv", {"date": parse_date, "name": str}),
    )
    _insert(
        conn,
        "leave_requests",
        _read_csv(
            data_dir / "leave_requests.csv",
            {
                "request_id": int,
                "employee_id": str,
                "leave_type": str,
                "start_date": parse_date,
                "end_date": parse_date,
                "days": int,
                "status": str,
                "created_at": parse_timestamp,
                "created_via": str,
                "comment": str,
            },
        ),
    )


def init_db(db_path: str | Path | None = None, data_dir: Path = DATA_DIR) -> None:
    """Create the schema and import the CSVs if the database is empty."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
            with conn:
                import_csvs(conn, data_dir)
    finally:
        conn.close()
