# Trading Helfer – Spezifikation

Stand: 20.11.2026. Dieses Issue ist die Referenz für alle Umsetzungs-Issues. Mit I-01 wird der Inhalt als `docs/SPEC.md` ins Repo übernommen; ab dann gilt die Datei.

## Ziel und Umfang

Trading Helfer ist eine lokale Desktop-App für technische Analyse, Kennzahlen und Backtesting von US-Aktien (S&P 500, Nasdaq-100) auf den Zeitrahmen 1D, 3D, 1W, 1M. Sie erzeugt Signale, platziert aber keine Orders. Diese Spezifikation ist die Grundlage der Issues in diesem Repo, die ein Agent ohne Rückfragen umsetzen kann.

**Im Umfang (Version 1.0, Phasen 1 bis 5)**

- Daten: Tageskurse (OHLCV, bereinigt um Splits und Dividenden) mit Historie so weit wie verfügbar (Ziel: vor 2000), Indexmitglieder, Fundamentaldaten
- Zeitrahmen: 1D als Basis, 3D, 1W, 1M werden daraus berechnet
- Charts: Kerzen, Volumen, Indikatoren, Signalmarker
- Kennzahlen: Bewertung, Wachstum, Profitabilität, Verschuldung, alle gängigen Kennzahlen&#32;
- Backtesting: Einzelwert und Portfolio, Gebühren, Slippage, Kennzahlen, Equity-Kurve, Walk-Forward
- Strategien: Plugin-Schnittstelle, damit mit KI erarbeitete Strategien als Datei hinzugefügt werden
- Scanner: alle Werte eines Universums auf aktuelle Signale prüfen

**Später (Phase 6 und danach)**

- Optionen: Ketten, Griechen, Payoff-Diagramme, Backtesting von Optionsstrategien
- Anbindung an einen Broker für automatisches Handeln

**Nicht-Ziele**

- Kein Intraday-Handel, keine Echtzeitkurse
- Keine Orderausführung in Version 1.0
- Keine Mehrbenutzer- oder Cloud-Funktion
- Keine Anlageberatung: die App ist ein Analysewerkzeug

## Entscheidungen

Alle acht Punkte sind entschieden; maßgeblich ist die Spalte „Entscheidung“.

| Nr | Frage | Optionen | Empfehlung | Entscheidung |
| --- | --- | --- | --- | --- |
| E1 | Betriebssystem | Windows, macOS, Linux, alle | Alle drei (Python läuft überall), getestet auf deinem System | ich habe nur Windows, die anderen nur wenn es kein Mehraufwand ist |
| E2 | Tech-Stack | Python + PySide6; Web-Technik in Electron/Tauri | Python 3.12 + PySide6 + pyqtgraph: beste Bibliotheken für Daten und Backtesting, KI schreibt Strategien am zuverlässigsten in Python | Wie empfohlen |
| E3 | Datenquelle Kurse | yfinance, Tiingo (kostenlos), Kaggle-CSV | Austauschbare Schnittstelle. Start mit yfinance, Tiingo als zweite Quelle, Kaggle-CSV als Import | Wie empfohlen und ideal wäre es Daten von vor 2000 zu finden. |
| E4 | Historische Indexmitglieder (gegen Survivorship Bias) | nur heutige Mitglieder; inklusive früherer Mitglieder | Inklusive früherer Mitglieder, sonst wirken Backtests zu gut | Wie empfohlen |
| E5 | Datenbank | SQLite; DuckDB | DuckDB (eine Datei, sehr schnell für Zeitreihen-Abfragen) | Wie empfohlen |
| E6 | Format der KI-Strategien | Python-Klasse; YAML-Regeln ohne Code | Python-Klasse in einem Plugin-Ordner, geprüft durch einen Validator | Wie empfohlen |
| E7 | Long und Short | nur Long; beide | Engine kann beides, Strategien standardmäßig Long | Wie empfohlen |
| E8 | KI in der App | KI nur außerhalb (Chat erstellt die Datei); Claude-API direkt in der App | Zuerst außerhalb; In-App-Assistent als optionales Issue in Phase 4 | Wie empfohlen |

Weitere Festlegungen: Währung USD, Startkapital 1.000 USD, Gebühr 1 USD je Order plus 0,05 % Slippage (alles einstellbar), Historie so weit wie verfügbar (Standard ab 1990-01-01), Bruchstücke von Aktien erlaubt (bei 1.000 USD Kapital sonst kaum Portfolio möglich).

## Datenquellen

Keine kostenlose Quelle deckt alles ab, deshalb bekommt die App eine Provider-Schnittstelle mit mehreren Quellen. Alle drei Kandidaten sind nur für private Nutzung erlaubt, was für dein Vorhaben passt.

| Quelle | Kosten | Inhalt | Grenzen | Rolle |
| --- | --- | --- | --- | --- |
| [yfinance](https://raw.githubusercontent.com/ranaroussi/yfinance/main/README.md) | 0 USD | Tageskurse bereinigt, Fundamentaldaten, Optionsketten (nur aktuell) | Inoffiziell, Yahoo-API „intended for personal use only“, kann ohne Vorwarnung brechen | Startquelle für Kurse, Kennzahlen, später Optionen |
| [Tiingo Starter](https://www.tiingo.com/about/pricing) | 0 USD | Tageskurse roh und bereinigt (Splits und Dividenden), 30+ Jahre | 500 Symbole/Monat, 50 Anfragen/Stunde, 1.000/Tag, keine Fundamentaldaten, „Internal Use Only“ | Zweite, stabile Quelle für Kurse; Power-Plan 30 USD/Monat bei Bedarf |
| [Kaggle S&P 500, 25 Jahre](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily) | 0 USD | OHLCV aller rund 500 Mitglieder ab 2000, täglich aktualisiert | Lizenz, Ursprung, Bereinigung und frühere Mitglieder auf der Seite nicht erkennbar | Schneller Offline-Import für den Erststart |

**Was die App zusätzlich braucht**

- Indexmitglieder mit Ein- und Austrittsdatum für S&P 500 und Nasdaq-100. Quelle: Änderungslisten der Wikipedia-Artikel, als CSV im Repo gepflegt (Issue D-05).
- Kurse früherer Mitglieder: yfinance liefert für viele delistete Ticker keine Daten mehr. Fehlende Werte werden im Datenqualitätsbericht ausgewiesen, nicht stillschweigend ignoriert.
- Optionsdaten mit Historie gibt es kostenlos nicht. Für Phase 6 wird das gesondert entschieden (z. B. eigene tägliche Snapshots ab Start oder ein Bezahlanbieter).

Die Kaggle-Seite ließ sich nur als Kurzbeschreibung lesen. Ich prüfe Lizenz und Spalten, sobald du den Datensatz herunterlädst oder mir die Datei gibst.

**Daten vor 2000**: Tiingo nennt „30+ Jahre“ Historie, yfinance liefert für viele Werte ebenfalls Kurse vor 2000. Der Standard-Startpunkt ist deshalb 1990-01-01, das Update lädt jeweils ab dem frühesten verfügbaren Datum. Die Indexmitglieder vor 2000 sind unsicherer, Backtests in diesem Zeitraum werden als Survivorship-Risiko markiert (D-05).

## Tech-Stack und Architektur

Die Logik (Daten, Indikatoren, Backtest, Strategien) ist reines Python ohne GUI-Abhängigkeit; die GUI ruft sie nur über Services auf. So lässt sich alles ohne Bildschirm testen, und eine spätere Automatisierung nutzt dieselben Module.

Schichten: GUI → Services → Logik-Module (Daten, Indikatoren/Kennzahlen, Backtest/Scanner, Strategien) → DuckDB. Quellen (yfinance, Tiingo, CSV) speisen nur das Daten-Modul; Strategien aus `plugins/` laden über die Strategie-Registry.

Strategien aus `plugins/` laufen durch denselben Validator und dieselbe Engine wie die eingebauten.

| Bereich | Wahl | Grund |
| --- | --- | --- |
| Sprache | Python 3.12 | Datenbibliotheken, KI schreibt Python zuverlässig |
| Paketverwaltung | uv, pyproject.toml | schnell, reproduzierbar mit Lockfile |
| GUI | PySide6 (Qt 6) | native Desktop-App, LGPL-Lizenz |
| Charts | pyqtgraph | schnell bei 25 Jahren Daten, läuft in Qt |
| Datenverarbeitung | pandas 2, numpy | Standard für Zeitreihen |
| Speicher | DuckDB (eine Datei `data/market.duckdb`) | schnelle Abfragen, kein Server |
| Konfiguration | pydantic-settings, TOML-Datei | geprüfte Einstellungen |
| Quellen | yfinance, requests (Tiingo) | siehe Datenquellen |
| Tests | pytest, pytest-qt, hypothesis | Logik, GUI und Eigenschaften testen |
| Qualität | ruff (Lint und Format), mypy (strict für `core`) | einheitlicher Code |
| CI | GitHub Actions (Windows; Linux für schnelle Tests) | jedes Issue läuft gegen dieselben Prüfungen |
| Paketierung | PyInstaller | startbare App ohne Python-Installation |

Indikatoren werden selbst mit numpy/pandas implementiert und mit festen Referenzwerten getestet. So hängt die App nicht an schlecht gepflegten TA-Bibliotheken.

**Ordnerstruktur**

```text
trading-helfer/
  pyproject.toml
  src/trading_helfer/
    core/          # Datenklassen, Konfiguration, Logging, Zeitrahmen
    data/          # providers/, storage.py, resample.py, universe.py, quality.py, update.py
    indicators/    # trend.py, momentum.py, volatility.py, volume.py, registry.py
    fundamentals/  # Kennzahlen berechnen und abfragen
    backtest/      # engine.py, broker_sim.py, costs.py, metrics.py, walkforward.py, optimize.py
    strategies/    # base.py, registry.py, validator.py, builtin/
    scanner/       # Signale über ein Universum
    services/      # Fassade, die die GUI aufruft
    gui/           # main_window.py, views/, widgets/
    app.py         # Einstiegspunkt
  plugins/         # eigene und KI-erarbeitete Strategien
  tests/           # unit/, integration/, gui/, fixtures/
  docs/            # Architektur, Strategie-Leitfaden für KI
```

**Regeln**

- `core`, `data`, `indicators`, `backtest`, `strategies`, `scanner` importieren niemals Qt.
- Lange Aufgaben (Datenupdate, Backtest, Scan) laufen in einem Hintergrund-Thread (`QThreadPool`) und melden Fortschritt; die GUI friert nie ein.
- Netzwerkzugriffe gibt es nur in `data/providers`; Tests laufen komplett offline mit Fixtures.

## Datenmodell

Gespeichert werden nur Tageskurse; 3D, 1W und 1M werden beim Laden berechnet und zwischengespeichert. Alle Tabellen liegen in einer DuckDB-Datei, Schema-Änderungen laufen über nummerierte Migrationen (`data/migrations/001_init.sql` …).

| Tabelle | Schlüssel | Spalten | Zweck |
| --- | --- | --- | --- |
| `instruments` | `symbol` | name, exchange, sector, industry, currency, is\_active, first\_date, last\_date, source | Stammdaten je Wertpapier |
| `index_membership` | `index_code, symbol, start_date` | end\_date (NULL = aktuell) | Wer wann im S&P 500 oder Nasdaq-100 war |
| `bars_daily` | `symbol, date` | open, high, low, close, volume, adj\_close, adj\_factor, source | Tageskurse roh plus Bereinigungsfaktor |
| `corporate_actions` | `symbol, date, type` | value (Split-Verhältnis oder Dividende je Aktie) | Splits und Dividenden |
| `fundamentals` | `symbol, period_end, period_type, metric` | value, reported\_date, source | Langformat, period\_type = Q oder FY |
| `update_log` | `id` | started\_at, finished\_at, provider, symbols\_ok, symbols\_failed, message | Protokoll jedes Datenupdates |
| `backtest_runs` | `run_id` (UUID) | strategy\_id, strategy\_version, params\_json, universe, timeframe, start, end, settings\_json, metrics\_json, created\_at | Reproduzierbare Ergebnisse |
| `backtest_trades` | `run_id, trade_id` | symbol, side, entry\_date, entry\_price, exit\_date, exit\_price, qty, pnl, pnl\_pct, fees, exit\_reason | Trade-Liste |
| `backtest_equity` | `run_id, date` | equity, cash, exposure, drawdown | Equity-Kurve |
| `signals` | `scan_id, symbol, date` | strategy\_id, signal (BUY, SELL, EXIT), strength, note | Ergebnisse des Scanners |

**Bereinigte Kurse**: `adj_factor` = adj\_close / close. Indikatoren und Backtests rechnen mit bereinigten OHLC-Werten (open, high, low, close jeweils × adj\_factor). Der Chart zeigt wahlweise roh oder bereinigt, Standard ist bereinigt.

**Zeitrahmen-Regeln** (in `data/resample.py`)

- 1W: Woche Montag bis Freitag, Kerze mit Datum des letzten Handelstags der Woche. Open = erstes Open, High = Maximum, Low = Minimum, Close = letztes Close, Volume = Summe.
- 3D: je drei aufeinanderfolgende Handelstage, gezählt ab dem festen Anker 2000-01-03 über den US-Börsenkalender; Handelstage vor dem Anker werden rückwärts in Dreiergruppen gezählt. Dadurch sind die Kerzen für alle Symbole und bei jedem Laden gleich.
- Die letzte, noch unvollständige Kerze wird als `is_partial = true` markiert. Backtests und Scanner nutzen nur abgeschlossene Kerzen.
- Börsenkalender: Paket `exchange_calendars` (Kalender XNYS).
- 1M: Kerze mit Datum des letzten Handelstags des Monats. Open = erstes Open, High = Maximum, Low = Minimum, Close = letztes Close, Volume = Summe.

## Module im Detail

Jedes Modul hat eine feste Schnittstelle, auf die sich die Issues beziehen. Signaturen sind verbindlich, Interna frei.

### Daten-Provider

```python
class PriceProvider(Protocol):
    name: str

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        ...
        # Spalten: date, open, high, low, close, volume, adj_close; leer, wenn nichts da

    def fetch_actions(self, symbol: str) -> pd.DataFrame: ...  # date, type, value


class FundamentalsProvider(Protocol):
    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        ...
        # period_end, period_type, metric, value, reported_date
```

- Implementierungen: `YFinanceProvider`, `TiingoProvider` (API-Key aus Einstellungen), `CsvImportProvider` (Kaggle und eigene CSV mit Spalten-Mapping).
- Rate-Limit und Wiederholungen: höchstens 2 Anfragen pro Sekunde je Provider, 3 Versuche mit 2, 4, 8 Sekunden Pause.
- Inkrementelles Update: je Symbol ab `last_date − 5 Tage` laden, Überschneidungen überschreiben. Bei geändertem adj\_factor (neuer Split oder Dividende) die ganze Historie des Symbols neu laden.
- Datenqualität: Prüfungen auf Lücken gegen Börsenkalender, High < Low, Kurse ≤ 0, Sprünge > 50 % ohne Corporate Action. Ergebnis als Bericht in der GUI.

### Indikatoren

Reine Funktionen `f(df, **params) -> pd.Series | pd.DataFrame`, registriert mit Name, Parametern, Standardwerten und Anzeigeart (im Chart oder eigenes Panel).

- Trend: SMA, EMA, WMA, MACD, ADX/DMI, Ichimoku, Parabolic SAR, Supertrend
- Momentum: RSI, Stochastik, ROC, Williams %R, CCI
- Volatilität: ATR, Bollinger Bänder, Keltner-Kanäle, Donchian-Kanäle, historische Volatilität
- Volumen: OBV, VWAP (gleitend), Chaikin Money Flow, Volumen-SMA
- Relative Stärke: Kurs geteilt durch Benchmark (SPY oder QQQ), RS-Rang im Universum

### Kennzahlen (Fundamentaldaten)

KGV (trailing), KUV, KBV, EV/EBITDA, Dividendenrendite, Umsatz- und Gewinnwachstum (YoY, 3 Jahre CAGR), Brutto-, operative und Nettomarge, ROE, ROIC, Verschuldungsgrad (Debt/Equity), Current Ratio, Free Cashflow, Marktkapitalisierung; außerdem EPS und EPS-Wachstum, PEG-Ratio, KCV (Kurs/operativer Cashflow), Free-Cashflow-Rendite, Ausschüttungsquote, Dividendenwachstum (5 Jahre), ROA, Asset Turnover, Enterprise Value, Netto-Verschuldung/EBITDA, Zinsdeckungsgrad, Quick Ratio, Veränderung der Aktienanzahl (Rückkäufe), Piotroski-F-Score und Altman-Z-Score. Werte sind „point in time“: im Backtest gilt eine Zahl erst ab `reported_date`.

### Backtest-Engine

Ereignisgesteuert je Kerze, damit Signale exakt wie später im Live-Betrieb entstehen.

- Ablauf je Kerze t: Strategie sieht nur Daten bis einschließlich Close von t. Orders werden zum Open von t+1 ausgeführt. Stop- und Limit-Orders fallen intraday aus High/Low von t+1; bei Gap über den Stop wird zum Open ausgeführt.
- Order-Typen: Market, Stop, Limit, Trailing Stop (in % oder ATR-Vielfachen).
- Positionsgröße: fester Betrag, % des Kapitals, Risiko je Trade (Abstand zum Stop), Volatilitäts-Ziel.
- Portfolio: mehrere Werte gleichzeitig, maximale Anzahl Positionen, Rangfolge bei mehr Signalen als freien Plätzen (Strategie liefert `strength`).
- Kosten: fixe Gebühr je Order, prozentuale Gebühr, Slippage in %. Short: Leihgebühr p. a. in %.
- Universum point in time: nur Werte, die am Tag t Indexmitglied waren, dürfen neu gekauft werden.
- Kennzahlen: Gesamtrendite, CAGR, Volatilität, Sharpe, Sortino, Calmar, Max Drawdown und Dauer, Trefferquote, Profit Factor, durchschnittlicher Gewinn/Verlust, Erwartungswert je Trade, Anzahl Trades, Haltedauer, Exposure, Vergleich mit Buy and Hold des Benchmarks.
- Robustheit: Parameter-Raster (Grid), Walk-Forward (z. B. 5 Jahre Training, 1 Jahr Test, rollierend), Monte-Carlo-Umordnung der Trades.
- Determinismus: gleiche Daten und Parameter ergeben byte-gleiche Ergebnisse (Zufall nur mit Seed).

### Strategie-Schnittstelle (Plugins)

Eine Strategie ist eine Python-Datei in `plugins/`, die eine Klasse von `Strategy` ableitet.

```python
class Strategy(ABC):
    id: ClassVar[str]  # eindeutig, z. B. "trend_donchian_breakout"
    name: ClassVar[str]
    version: ClassVar[str]  # SemVer
    description: ClassVar[str]
    timeframes: ClassVar[list[Timeframe]]  # z. B. [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]]  # Param(default, min, max, step, beschreibung)

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Vektorisiert alle Indikatoren vorab berechnen."""

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Je Kerze: Kontext mit Kursen bis t, Indikatoren, Position, Kapital."""

    def plot(self) -> list[PlotSpec]:
        """Optional: welche Linien und Marker im Chart erscheinen."""
```

- Validator vor dem Laden: Syntax, Pflichtfelder, keine Netzwerk- oder Dateizugriffe (AST-Prüfung auf verbotene Importe wie `os`, `socket`, `requests`, `subprocess`), Lookahead-Test (Ergebnis bis t darf sich nicht ändern, wenn Daten nach t verändert werden), Lauf auf Testdaten ohne Fehler.
- GUI-Funktion „Strategie hinzufügen“: Datei wählen oder Code einfügen, Validator-Bericht ansehen, dann in `plugins/` speichern.
- `docs/strategy_guide.md` erklärt die Schnittstelle so, dass eine KI daraus eine gültige Strategie schreibt (inkl. Vorlage und zwei Beispielen).
- Mitgelieferte Beispiele: SMA-Kreuzung, Donchian-Ausbruch, RSI-Mean-Reversion im Aufwärtstrend.

### Scanner

Wendet eine Strategie auf ein Universum an und listet Werte mit Signal auf der letzten abgeschlossenen Kerze. Filter: Sektor, Marktkapitalisierung, Kennzahlen. Export als CSV.

### GUI

Hauptfenster mit linker Watchlist und Tabs.

- **Chart**: Kerzen und Volumen, Zeitrahmen-Umschalter 1D/3D/1W/1M, Indikatoren hinzufügen und konfigurieren, Zoom und Verschieben, Fadenkreuz mit Werten, Signal- und Trade-Marker, log/linear.
- **Kennzahlen**: Tabelle und Verlauf der Fundamentaldaten je Wert, Vergleich mit Sektor-Median.
- **Backtest**: Strategie, Parameter, Universum, Zeitraum und Kosten wählen; Ergebnis mit Kennzahlen, Equity- und Drawdown-Kurve, Trade-Liste (Klick springt in den Chart), Monatsrenditen-Heatmap; Läufe vergleichen.
- **Optimierung**: Parameter-Raster, Walk-Forward, Ergebnis-Heatmap.
- **Scanner**: Signal-Liste mit Filtern.
- **Strategien**: Liste, Details, hinzufügen, Validator-Bericht.
- **Daten**: Update starten, Fortschritt, Qualitätsbericht, Provider und API-Keys.
- Hell- und Dunkelmodus, Sprache Deutsch (Texte zentral in einer Datei, später übersetzbar).

## Konventionen für die Umsetzung

Diese Regeln gelten für jedes Issue und stehen später in `CLAUDE.md` im Repo, damit ein Agent sie ohne Nachfrage befolgt.

**Definition of Done (für jedes Issue)**

1. Alle Akzeptanzkriterien des Issues erfüllt und durch Tests belegt.
2. `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src` und `uv run pytest` grün, lokal und in CI.
3. Testabdeckung der geänderten Module mindestens 85 % (GUI-Module 60 %).
4. Kein Netzwerkzugriff in Tests; externe Antworten als Fixtures in `tests/fixtures/`.
5. Öffentliche Funktionen mit Typen und Docstring; Nutzertexte auf Deutsch, Code und Bezeichner auf Englisch.
6. Ein Pull Request je Issue, Titel `[<Issue-ID>] <Titel>`, verknüpft mit `Closes #<Nr>`.

**Aufbau jedes Issues**

- Ziel (ein Satz), Kontext (Verweis auf den Abschnitt dieser Spezifikation)
- Abhängigkeiten (Issue-IDs)
- Umfang: Dateien und Schnittstellen, die entstehen oder sich ändern
- Akzeptanzkriterien als prüfbare Aussagen
- Tests: konkrete Testfälle mit erwarteten Werten
- Nicht im Umfang

**Labels**: `phase-1` bis `phase-6`, `data`, `indicators`, `backtest`, `strategy`, `gui`, `infra`, `docs`.

**Testdaten**: `tests/fixtures/bars/` enthält fünf echte Symbole (AAPL, MSFT, JPM, XOM, ein delisteter Wert) als CSV für 2018 bis 2023 sowie synthetische Reihen mit bekannten Eigenschaften (konstanter Trend, Sägezahn, Gap, Split 4:1).

## Issues

45 Issues in sechs Phasen; Phasen 1 bis 5 sind voll spezifiziert, Phase 6 (Optionen) ist grob, bis die Datenfrage geklärt ist. Ein Issue darf erst starten, wenn alle Abhängigkeiten gemergt sind.

Innerhalb einer Phase können Issues ohne gegenseitige Abhängigkeit parallel laufen, z. B. D-02, D-07 und X-01.

| ID | Titel | Phase | Abhängig von |
| --- | --- | --- | --- |
| I-01 | Repo-Grundgerüst, Werkzeuge und CI | 1 | – |
| I-02 | Konfiguration und Logging | 1 | I-01 |
| I-03 | Kern-Datentypen und Zeitrahmen | 1 | I-01 |
| D-01 | DuckDB-Speicher und Migrationen | 1 | I-02, I-03 |
| D-02 | Provider-Schnittstelle und yfinance | 1 | I-03 |
| D-03 | Tiingo-Provider | 1 | D-02 |
| D-04 | CSV-Import (Kaggle) | 1 | D-01 |
| D-05 | Indexmitglieder mit Historie | 1 | D-01 |
| D-06 | Inkrementelles Datenupdate und CLI | 1 | D-01, D-02, D-05 |
| D-07 | Zeitrahmen 3D, 1W und 1M berechnen | 1 | I-03 |
| D-08 | Datenqualitätsprüfung | 1 | D-01 |
| D-09 | Fundamentaldaten laden | 1 | D-01, D-02 |
| G-01 | GUI-Grundgerüst | 2 | I-02 |
| G-02 | Daten-Tab | 2 | G-01, D-06, D-08 |
| X-01 | Indikator-Registry und Trend-Indikatoren | 2 | I-03 |
| X-02 | Momentum- und Volatilitäts-Indikatoren | 2 | X-01 |
| X-03 | Volumen und relative Stärke | 2 | X-01, D-05 |
| G-03 | Chart mit Kerzen, Volumen, Zeitrahmen | 2 | G-01, D-07 |
| G-04 | Indikatoren im Chart | 2 | G-03, X-01 |
| F-01 | Kennzahlen berechnen (point in time) | 2 | D-09 |
| G-05 | Kennzahlen-Tab | 2 | G-01, F-01 |
| B-01 | Strategie-Basisklasse und Order-Modell | 3 | I-03, X-01 |
| B-02 | Simulierter Broker und Kosten | 3 | B-01 |
| B-03 | Backtest-Engine für einen Wert | 3 | B-02, D-07 |
| B-04 | Portfolio-Backtest | 3 | B-03, D-05 |
| B-05 | Performance-Kennzahlen | 3 | B-03 |
| B-06 | Ergebnisse speichern und laden | 3 | B-05, D-01 |
| B-07 | Drei Beispielstrategien | 3 | B-03 |
| G-06 | Backtest-Tab | 3 | B-06, G-04 |
| B-08 | Parameter-Optimierung und Walk-Forward | 3 | B-04, B-05 |
| B-09 | Monte-Carlo-Analyse | 3 | B-05 |
| G-07 | Optimierungs-Tab | 3 | B-08, G-06 |
| S-01 | Plugin-Loader und Registry | 4 | B-01 |
| S-02 | Strategie-Validator | 4 | S-01, B-03 |
| S-03 | Strategie-Leitfaden für KI | 4 | S-02, B-07 |
| G-08 | Strategien-Tab mit „Hinzufügen“ | 4 | S-02, G-01 |
| S-04 | Scanner | 4 | S-01, D-05, D-07 |
| G-09 | Scanner-Tab | 4 | S-04, G-01 |
| P-01 | Paketierung als startbare App | 4 | G-09 |
| R-01 | Forschungsrahmen und Baseline | 5 | B-08, B-09 |
| R-02 | Kandidat Trendfolge 1W | 5 | R-01 |
| R-03 | Kandidat Momentum-Rotation 1W | 5 | R-01, X-03 |
| R-04 | Kandidat Pullback im Trend 3D | 5 | R-01 |
| O-01 | Optionen: Datenquelle und Snapshots | 6 | D-06 |
| O-02 | Optionen: Bewertung und Griechen | 6 | I-03 |

Weitere Phase-6-Issues (Optionsketten-Ansicht, Payoff-Diagramm, Options-Backtest) werden nach O-01 spezifiziert. Der In-App-KI-Assistent (E8) kommt als S-05 dazu, falls du ihn willst.

---
_Generated by [Claude Code](https://claude.ai/code)_
