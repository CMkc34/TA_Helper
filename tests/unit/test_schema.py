"""Migrationsrunner."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from trading_helfer.data.schema import (
    MIGRATIONS_DIR,
    MigrationError,
    applied_versions,
    discover_migrations,
    run_migrations,
)


def write(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql, encoding="utf-8")


def test_shipped_migrations_start_with_init() -> None:
    migrations = discover_migrations(MIGRATIONS_DIR)
    assert migrations[0].version == 1
    assert migrations[0].name == "init"
    assert [m.version for m in migrations] == sorted(m.version for m in migrations)


def test_runs_in_order_and_only_once(tmp_path: Path) -> None:
    write(tmp_path, "002_second.sql", "INSERT INTO t VALUES (2);")
    write(tmp_path, "001_first.sql", "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1);")
    write(tmp_path, "README.txt", "wird ignoriert")
    conn = duckdb.connect()

    assert run_migrations(conn, tmp_path) == [1, 2]
    assert run_migrations(conn, tmp_path) == []
    assert conn.execute("SELECT x FROM t ORDER BY x").fetchall() == [(1,), (2,)]
    assert applied_versions(conn) == {1, 2}

    write(tmp_path, "003_third.sql", "INSERT INTO t VALUES (3);")
    assert run_migrations(conn, tmp_path) == [3]
    names = conn.execute("SELECT version, name FROM schema_version ORDER BY version").fetchall()
    assert names == [(1, "first"), (2, "second"), (3, "third")]


def test_failing_migration_rolls_back(tmp_path: Path) -> None:
    write(tmp_path, "001_ok.sql", "CREATE TABLE a (x INTEGER);")
    write(tmp_path, "002_bad.sql", "CREATE TABLE b (x INTEGER); SELECT * FROM missing_table;")
    conn = duckdb.connect()

    with pytest.raises(MigrationError, match=r"002_bad\.sql"):
        run_migrations(conn, tmp_path)

    assert applied_versions(conn) == {1}
    tables = {row[0] for row in conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "a" in tables
    assert "b" not in tables


@pytest.mark.parametrize("name", ["1_init.sql", "001-init.sql", "001_Init.sql", "init.sql"])
def test_rejects_bad_file_names(tmp_path: Path, name: str) -> None:
    write(tmp_path, name, "SELECT 1;")
    with pytest.raises(MigrationError, match="Ungültiger Dateiname"):
        discover_migrations(tmp_path)


def test_rejects_duplicate_versions(tmp_path: Path) -> None:
    write(tmp_path, "001_a.sql", "SELECT 1;")
    write(tmp_path, "001_b.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="doppelt"):
        discover_migrations(tmp_path)
