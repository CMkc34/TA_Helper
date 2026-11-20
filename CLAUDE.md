# Trading Helfer – Hinweise für Agenten

Maßgeblich ist die Spezifikation in [`docs/SPEC.md`](docs/SPEC.md). Jedes Issue verweist auf ihre Abschnitte; Signaturen dort sind verbindlich, Interna frei.

## Befehle

```bash
uv sync                      # Abhängigkeiten installieren
uv run trading-helfer        # App starten
uv run ruff check            # Lint
uv run ruff format --check   # Format prüfen (uv run ruff format zum Formatieren)
uv run mypy src              # Typen prüfen
uv run pytest                # Tests (inkl. Abdeckung); -m "not slow" ohne Benchmarks
```

## Architekturregeln

- `core`, `data`, `indicators`, `fundamentals`, `backtest`, `strategies`, `scanner` importieren niemals Qt. Nur `gui` (und `app.py` zum Start der GUI) nutzt PySide6/pyqtgraph.
- Die GUI ruft Logik nur über `services` auf. Lange Aufgaben laufen im `QThreadPool` und melden Fortschritt.
- Netzwerkzugriffe gibt es nur in `data/providers`. Tests laufen komplett offline (`tests/conftest.py` blockiert Verbindungen); externe Antworten liegen als Fixtures in `tests/fixtures/`.
- Testdaten sind synthetisch (Trend, Sägezahn, Gap, Split 4:1 …). Echte Kursdaten werden nicht ins Repo committet.
- Datenbank-Schemaänderungen nur über nummerierte Migrationen in `src/trading_helfer/data/migrations/NNN_name.sql`.
- Determinismus: gleiche Daten und Parameter ergeben gleiche Ergebnisse; Zufall nur mit Seed.

## Definition of Done (für jedes Issue)

1. Alle Akzeptanzkriterien des Issues erfüllt und durch Tests belegt.
2. `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src` und `uv run pytest` grün, lokal und in CI.
3. Testabdeckung der geänderten Module mindestens 85 % (GUI-Module 60 %).
4. Kein Netzwerkzugriff in Tests; externe Antworten als Fixtures in `tests/fixtures/`.
5. Öffentliche Funktionen mit Typen und Docstring; Nutzertexte auf Deutsch, Code und Bezeichner auf Englisch.
6. Ein Pull Request je Issue, Titel `[<Issue-ID>] <Titel>`, verknüpft mit `Closes #<Nr>`.

## Konventionen

- Python 3.12, Paket `trading_helfer` unter `src/`, Tests unter `tests/unit`, `tests/integration`, `tests/gui`.
- Ein Issue darf erst starten, wenn alle Abhängigkeiten gemergt sind (Tabelle in `docs/SPEC.md`).
- Neue Abhängigkeiten über `uv add <paket>` (bzw. `uv add --dev`), damit `uv.lock` aktuell bleibt.
- Langsame Tests (Benchmarks) mit `@pytest.mark.slow` markieren.
