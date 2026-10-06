import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from server.db import get_connection, init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = get_connection(db_path)
    yield connection
    connection.close()
