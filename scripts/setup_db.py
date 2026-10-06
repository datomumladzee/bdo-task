"""Create the SQLite database, import the CSVs and print row counts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.db import TABLES, get_connection, get_db_path, init_db


def main() -> None:
    db_path = get_db_path()
    init_db(db_path)
    print(f"Database: {db_path}")
    conn = get_connection(db_path)
    try:
        for table in TABLES:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<20} {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
