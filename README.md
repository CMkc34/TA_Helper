# Trading Helfer

Lokale Desktop-App für technische Analyse, Kennzahlen und Backtesting von US-Aktien (S&P 500, Nasdaq-100) auf den Zeitrahmen 1D, 3D, 1W und 1M. Die App erzeugt Signale, platziert aber keine Orders und ist keine Anlageberatung.

Details: [docs/SPEC.md](docs/SPEC.md).

## Installation

Voraussetzungen: Python 3.12 und [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CMkc34/TA_Helper.git
cd TA_Helper
uv sync
```

## Start

```bash
uv run trading-helfer            # App starten
uv run trading-helfer --version  # Version anzeigen
```

## Options-Schnappschüsse

Kostenlose historische Optionsdaten gibt es nicht. Der Befehl `options-snapshot` speichert
deshalb die aktuellen Optionsketten (yfinance) als Schnappschuss des Tages in der Tabelle
`option_snapshots`; so entsteht ab dem ersten Lauf eine eigene Historie.

```bash
uv run trading-helfer options-snapshot                       # Watchlist „Eigene“
uv run trading-helfer options-snapshot --watchlist "Nasdaq-100" --max-expiries 6
uv run trading-helfer options-snapshot --symbols SPY QQQ AAPL
```

Ein zweiter Lauf am selben Tag ersetzt den Schnappschuss, am Wochenende gilt der Freitag
als Stichtag. Täglich nach Börsenschluss über die Windows-Aufgabenplanung, z. B. um
22:30 Uhr:

```bat
schtasks /Create /SC WEEKLY /D MO,TUE,WED,THU,FRI /ST 22:30 /TN "Trading Helfer Optionen" ^
  /TR "cmd /c cd /d C:\Pfad\zum\Repo && uv run trading-helfer options-snapshot"
```

## Tests und Prüfungen

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest                 # alle Tests mit Abdeckung
uv run pytest -m "not slow"   # ohne Benchmarks
```

## Lizenz der Daten

Kursdaten von Yahoo (yfinance) und Tiingo sind nur für die private Nutzung bestimmt. Das Repo enthält deshalb ausschließlich synthetische Testdaten.
