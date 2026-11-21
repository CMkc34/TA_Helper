"""Migrationsrunner für das DuckDB-Schema.

Schemaänderungen liegen als nummerierte SQL-Dateien ``NNN_name.sql`` im Ordner
``data/migrations/``. Der Runner führt alle noch nicht angewendeten Dateien in
aufsteigender Reihenfolge aus, jede in einer eigenen Transaktion, und vermerkt sie in
der Tabelle ``schema_version``. Eine bereits angewendete Migration läuft nie ein
zweites Mal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MIGRATIONS_DIR",
    "Migration",
    "MigrationError",
    "applied_versions",
    "discover_migrations",
    "run_migrations",
]

MIGRATIONS_DIR: Final = Path(__file__).with_name("migrations")
"""Ordner mit den mitgelieferten Migrationen."""

_FILE_PATTERN: Final = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")

_CREATE_VERSION_TABLE: Final = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    name        VARCHAR NOT NULL,
    applied_at  TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""


class MigrationError(RuntimeError):
    """Eine Migration ist ungültig oder ließ sich nicht ausführen."""


@dataclass(frozen=True, slots=True)
class Migration:
    """Eine Migrationsdatei."""

    version: int
    """Laufende Nummer aus dem Dateinamen (``001`` → 1)."""
    name: str
    """Name ohne Nummer und Endung (``001_init.sql`` → ``init``)."""
    path: Path
    """Pfad der SQL-Datei."""

    def read_sql(self) -> str:
        """Liest den SQL-Text der Migration (UTF-8)."""
        return self.path.read_text(encoding="utf-8")


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Findet alle Migrationsdateien in ``directory``, sortiert nach Version.

    Dateien, die nicht auf ``.sql`` enden, werden ignoriert.

    Raises:
        MigrationError: Ein Dateiname passt nicht zum Muster ``NNN_name.sql`` oder eine
            Versionsnummer kommt doppelt vor.
    """
    migrations: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _FILE_PATTERN.match(path.name)
        if match is None:
            raise MigrationError(
                f"Ungültiger Dateiname der Migration: {path.name} (erwartet NNN_name.sql)."
            )
        version = int(match.group(1))
        if version in migrations:
            raise MigrationError(
                f"Migrationsnummer {version:03d} ist doppelt vergeben: "
                f"{migrations[version].path.name} und {path.name}."
            )
        migrations[version] = Migration(version, match.group(2), path)
    return [migrations[v] for v in sorted(migrations)]


def applied_versions(conn: Any) -> set[int]:
    """Gibt die Versionen zurück, die in ``schema_version`` vermerkt sind.

    Legt die Tabelle ``schema_version`` an, falls sie fehlt.

    Args:
        conn: Offene DuckDB-Verbindung.
    """
    conn.execute(_CREATE_VERSION_TABLE)
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    return {int(row[0]) for row in rows}


def run_migrations(conn: Any, directory: Path = MIGRATIONS_DIR) -> list[int]:
    """Wendet alle noch fehlenden Migrationen in aufsteigender Reihenfolge an.

    Jede Migration läuft samt Eintrag in ``schema_version`` in einer Transaktion;
    schlägt sie fehl, bleibt die Datenbank auf dem Stand davor.

    Args:
        conn: Offene DuckDB-Verbindung.
        directory: Ordner mit den Migrationsdateien.

    Returns:
        Die in diesem Aufruf angewendeten Versionen (leer, wenn alles aktuell ist).

    Raises:
        MigrationError: Eine Datei ist ungültig oder ihr SQL schlägt fehl.
    """
    done = applied_versions(conn)
    applied: list[int] = []
    for migration in discover_migrations(directory):
        if migration.version in done:
            continue
        sql = migration.read_sql()
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_version (version, name) VALUES (?, ?)",
                [migration.version, migration.name],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            raise MigrationError(f"Migration {migration.path.name} fehlgeschlagen: {exc}") from exc
        applied.append(migration.version)
    return applied
