# Forschungsrahmen (R-01)

Ziel ist eine robuste 3D/1W-Strategie, die über Marktphasen hinweg den Benchmark risikobereinigt schlägt. Damit kein Ergebnis durch Überanpassung zustande kommt, wird jeder Kandidat (R-02 bis R-04) mit demselben Skript und demselben festen Ablauf geprüft.

## Aufruf

```bash
# 1. Entwickeln: nur In-Sample (Kurse nach 2015 werden vor jeder Berechnung abgeschnitten)
uv run python research/run_study.py --strategy trend_sma_cross --tf W1 --index SP500 \
    --param fast=10 --param slow=40 --in-sample-only

# 2. Erst wenn die Strategie feststeht: vollständige Studie mit Out-of-Sample
uv run python research/run_study.py --strategy trend_sma_cross --tf W1 --index SP500 \
    --param fast=10 --param slow=40
```

Weitere Optionen: `--symbol AAPL` oder `--symbols AAPL MSFT …` statt `--index`, `--max-positions 10`, `--grid fast=8,10,12` (eigenes Walk-Forward-Raster), `--end 2024-12-31`, `--workers 4`, `--out ORDNER`, `--db DATEI`, `--capital USD` (Standard 100000). Die Strategie wird wie in der App aus den eingebauten Strategien und dem Plugin-Ordner geladen.

Der Bericht landet in `research/reports/<strategie>_<zeitrahmen>_<universum>[_is]/`:

| Datei | Inhalt |
| --- | --- |
| `report.md` | Bericht mit Tabellen in fester Reihenfolge |
| `equity.svg` | Equity der Strategie und der Baselines (logarithmisch), Grenze zum Out-of-Sample |
| `drawdown.svg` | Drawdown der Strategie und der Baselines |
| `monte_carlo.svg` | Verteilung des Max Drawdowns aus der Umordnung der Trades |
| `summary.json` | Kriterien und Kennzahlen zum Vergleich der Kandidaten |

Gleiche Daten und Parameter ergeben byte-gleiche Berichte (Monte-Carlo mit festem Seed 42).

## Ablauf

| Schritt | Zeitraum | Inhalt |
| --- | --- | --- |
| In-Sample | 2000-01-01 bis 2015-12-31 | Backtest mit den gewählten Parametern |
| Robustheit | In-Sample | Optimierung über die Parameter-Nachbarschaft: jeder Zahlenparameter einen `step` darunter, gleich, einen darüber |
| Walk-Forward | 2000 bis Datenende | 5 Jahre Training, 1 Jahr Test, rollierend, Ziel Sharpe; Raster = Nachbarschaft oder `--grid` |
| Gesamtzeitraum | 2000 bis Datenende | Trades, Max Drawdown, Equity- und Drawdown-Grafik |
| Monte-Carlo | Gesamtzeitraum | 1.000 Umordnungen der Trades, Seed 42 |
| Kosten-Stresstest | Gesamtzeitraum | Gebühr und Slippage verdoppelt |
| Marktphasen | 2000–2002, 2008–2009, 2020, 2022 | Rendite und Max Drawdown je Phase |
| Out-of-Sample | 2016-01-01 bis Datenende | Erst am Ende ansehen |

Studien rechnen mit **100.000 USD** Startkapital (`--capital`, unabhängig vom App-Standard 1.000 USD). Bei 1.000 USD und 10 Positionen kostet die feste Gebühr je Kauf und Verkauf rund 2 % und überdeckt die Regel (Befund R-04, Entscheidung vom 23.11.2026). Kosten kommen aus den Einstellungen (Standard 1 USD je Order, 0,05 % Slippage, Bruchstücke erlaubt). Vor 2000 liegen nur Kerzen für den Anlauf der Indikatoren.

**Baselines** (ohne Kosten, gleicher Zeitrahmen): Buy and Hold SPY, Buy and Hold QQQ und das gleichgewichtete Universum. Das gleichgewichtete Universum nimmt je Kerze den Durchschnitt der Renditen aller Werte, die zu Beginn der Kerze Indexmitglied waren (bei einem einzelnen Wert dessen Buy and Hold).

## Datenbasis prüfen

Studien sind nur so gut wie die Kurse. yfinance kennt viele ausgeschiedene Werte nicht
mehr und liefert unter neu vergebenen Tickern Kurse der neuen Firma. Vor einer
Bewertung deshalb:

```bash
# Tiingo als Ersatzquelle (kostenloser Key, 50 Anfragen pro Stunde, 1.000 pro Tag)
set TH_TIINGO_API_KEY=...          # PowerShell: $env:TH_TIINGO_API_KEY = "..."
uv run trading-helfer update --index SP500 --recheck --wait   # wartet am Tiingo-Limit, über Nacht laufen lassen
uv run trading-helfer quality --exclude                 # Fremdkurse ausschließen, Bericht
uv run trading-helfer exclude                           # Liste der ausgeschlossenen Werte
```

Werte ohne Kurse bei yfinance fragt das Update bei Tiingo nach; Werte mit Fremdkursen
lädt `--recheck` aus allen Quellen neu. Was danach noch Fremdkurse hat, schließt das
Update automatisch aus (`trading-helfer exclude --allow SYMBOL` gibt einen Wert trotzdem
frei). Eigene Daten, etwa ein Kaggle-Datensatz mit delisteten Werten, kommen per
`uv run trading-helfer import-csv <ordner>`; danach `quality --exclude` erneut ausführen.
Fehlen am Stichtag mehr als 10 % der damaligen Indexmitglieder, steht im Bericht ein
Survivorship-Hinweis mit der Abdeckung je Jahr.

## Bewertungskriterien (verbindlich für R-02 bis R-04)

| Kriterium | Gemessen an |
| --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | Sharpe der beiden getrennten Backtests; In-Sample-Sharpe muss positiv sein |
| Max Drawdown kleiner als der Benchmark | Gesamtzeitraum, verglichen mit Buy and Hold SPY |
| Mindestens 100 Trades | Abgeschlossene Trades im Gesamtzeitraum |
| Robustheitsmaß ≥ 60 % | Anteil der Nachbarparameter mit positivem In-Sample-Sharpe; ohne variierbare Parameter der Anteil der Walk-Forward-Fenster mit positivem Test-Sharpe |
| Positiv nach doppelten Kosten | Gesamtrendite im Kosten-Stresstest > 0 |

Ein Kandidat ist **bestanden**, wenn alle fünf Kriterien erfüllt sind. Mit `--in-sample-only` bleibt das Sharpe-Kriterium **offen**, und Drawdown, Trades und Kosten beziehen sich nur auf 2000–2015.

## Regeln für die Arbeit an Kandidaten

1. Parameter nur mit `--in-sample-only` entwickeln und verändern.
2. Die vollständige Studie mit Out-of-Sample einmal am Ende ausführen. Wer danach Parameter ändert, muss das im Bericht des Kandidaten offenlegen.
3. Berichte enthalten nur abgeleitete Kennzahlen; Kursdaten werden nicht ins Repo committet.
