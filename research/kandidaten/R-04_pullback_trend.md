# Kandidat R-04: Pullback im Trend (3D)

Plugin: [`plugins/pullback_trend.py`](../../plugins/pullback_trend.py), ID `pullback_trend`, Version 1.0.0.
Studienrahmen und verbindliche Kriterien: [`research/README.md`](../README.md).

## Hypothese

Werte in einem langfristigen Aufwärtstrend kehren nach einem kurzen, starken Rücksetzer
überdurchschnittlich oft zum Trend zurück. Auf 3-Tages-Kerzen ist der Rücksetzer groß genug,
dass Gebühr (1 USD je Order) und Slippage weniger ins Gewicht fallen als auf Tageskerzen,
und die Haltedauer bleibt kurz (höchstens 10 Kerzen, also etwa sechs Wochen).

## Regeln

Entscheidung auf dem Close der Kerze t, Ausführung zum Open von t+1.

| Teil | Regel | Parameter (Standard) |
| --- | --- | --- |
| Trendfilter | Close über der SMA 67 (SMA 200 Tage ÷ 3) | `trend_period` = 67 |
| Rücksetzer | RSI(2) unter 10, alternativ mindestens 3 tiefere Closes in Folge | `trigger` = `rsi`, `rsi_entry` = 10, `down_closes` = 3 |
| Universum | Nur Werte, die am Tag t Indexmitglied sind | |
| Größe | 10 % des Kapitals je Position, höchstens 10 Positionen | `--max-positions 10` |
| Rangfolge | Mehr Signale als freie Plätze: tiefster RSI(2) zuerst | |
| Stop | Verkaufs-Stop 2 × ATR(14) unter dem Einstiegskurs, fest | `atr_mult` = 2,0 |
| Erholung | Verkauf, sobald der RSI(2) über 70 steigt | `rsi_exit` = 70 |
| Zeit-Stop | Verkauf nach 10 Kerzen ohne Erholung oder Stop | `max_bars` = 10 |

Alle Standardwerte stammen aus dem Issue und wurden vor dem ersten Lauf festgelegt, nicht an
Daten angepasst.

**Robustheit:** Die Studie variiert die drei Parameter mit Schrittweite: `trend_period`
(57/67/77), `rsi_entry` (5/10/15) und `atr_mult` (1,5/2,0/2,5), also 27 Kombinationen.
`rsi_exit` und `max_bars` sind feste Regeln aus dem Issue (keine Schrittweite), damit das
Raster und die Laufzeit des Walk-Forward klein bleiben; sie lassen sich mit `--param` ändern.

## Studie ausführen

Die Studie braucht die lokale Kursdatenbank (S&P 500 ab 1990 mit früheren Mitgliedern) und
läuft deshalb auf dem eigenen Rechner, nicht in der Cloud.

1. Plugin in den Plugin-Ordner der App kopieren, z. B. unter Windows
   `copy plugins\pullback_trend.py "%LOCALAPPDATA%\trading-helfer\plugins\"`, oder im
   Strategien-Tab mit „Hinzufügen“ die Datei wählen. Der Validator muss grün sein.
2. In-Sample (Parameter dürfen nur hier geändert werden):

   ```bash
   uv run python research/run_study.py --strategy pullback_trend --tf D3 --index SP500 --in-sample-only
   ```

3. Erst danach einmal die vollständige Studie mit Out-of-Sample:

   ```bash
   uv run python research/run_study.py --strategy pullback_trend --tf D3 --index SP500
   ```

   Der Bericht landet in `research/reports/pullback_trend_D3_SP500/`. Variante mit
   tieferen Closes: zusätzlich `--param trigger=closes`.

## Bewertung gegen die Kriterien

Stand 22.11.2026: **offen**. In der Cloud-Umgebung gibt es keine Kursdaten (Datenquellen
gesperrt, echte Kurse werden nicht ins Repo committet). Der Ablauf ist mit synthetischen
Kursen getestet (`tests/unit/test_pullback_plugin.py`); die Zahlen unten werden nach dem
lokalen Lauf aus `summary.json` übernommen.

| Kriterium | Ziel | In-Sample 2000–2015 | Gesamt mit Out-of-Sample |
| --- | --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | erfüllt | offen | – |
| Max Drawdown kleiner als Buy and Hold SPY | erfüllt | – | – |
| Mindestens 100 Trades | erfüllt | – | – |
| Robustheitsmaß ≥ 60 % | erfüllt | – | – |
| Positiv nach doppelten Kosten | erfüllt | – | – |
| **Urteil** | bestanden | – | – |

## Erwartungen und Risiken

- **Trades:** Rücksetzer-Strategien handeln oft. Bei rund 500 Werten sind 100 Trades in
  16 Jahren sicher erreicht; eher ist die Zahl der Signale größer als die 10 Plätze.
- **Drawdown:** Der Trendfilter hält die Strategie in Bärenmärkten (2000–2002, 2008) weitgehend
  draußen, weil wenige Werte über ihrer SMA 67 liegen. Ein Crash innerhalb eines Aufwärtstrends
  (2020) trifft dagegen volle zehn Positionen; der feste ATR-Stop begrenzt den Schaden je Wert,
  aber Gaps unter den Stop werden zum Open ausgeführt.
- **Kosten:** Bei 1.000 USD Startkapital ist eine Position anfangs 100 USD groß; 2 USD Gebühr je
  Runde sind dann 2 %. Der Kosten-Stresstest (doppelte Kosten) ist deshalb das kritischste
  Kriterium. Falls er scheitert, ist das ein Befund über das kleine Konto, nicht unbedingt
  über die Regel; das wäre in der gemeinsamen Entscheidung nach R-02 bis R-04 zu besprechen.
- **Überanpassung:** Keine Parameter wurden an Daten angepasst. Änderungen nach dem
  Out-of-Sample-Lauf werden hier offengelegt.
