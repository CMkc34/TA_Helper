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
