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

## Bewertung gegen die Kriterien (In-Sample 2000–2015, 100.000 USD)

Lauf am 23.11.2026 mit `--in-sample-only --max-positions 10` und Standardparametern
(trend_period 67, trigger rsi, rsi_entry 10, rsi_exit 70, max_bars 10, atr_mult 2,0),
Startkapital 100.000 USD (Entscheidung vom 23.11.2026, siehe unten), 1 USD je Order,
0,05 % Slippage. Laufzeit rund 50 Minuten. Out-of-Sample ist **nicht** gerechnet.

Datenbasis: lokale Datenbank aus yfinance plus Tiingo für delistete Werte (1.015 von 1.214
S&P-Mitgliedern seit 1990, 142 Werte mit Fremdkursen ausgeschlossen). Je nach Jahr fehlen
Kurse für 36–45 % der damaligen Mitglieder; die Zahlen sind eher zu gut als zu schlecht.

**Urteil: offen.** Alle vier In-Sample-Kriterien sind erfüllt; es fehlt das
Out-of-Sample-Kriterium.

| Kriterium | Ergebnis | Erfüllt |
| --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | nicht ausgewertet | offen |
| Max Drawdown kleiner als Buy and Hold SPY | −35,5 % gegenüber −54,7 % | ja |
| Mindestens 100 Trades | 2.584 | ja |
| Robustheitsmaß ≥ 60 % | 100 % (alle 27 Nachbarn Sharpe 0,50 bis 0,75) | ja |
| Positiv nach doppelten Kosten | +175,3 % | ja |

| Kennzahl | Strategie | Kosten × 2 | SPY | QQQ | Universum gleichgewichtet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gesamtrendite | 274,6 % | 175,3 % | 90,0 % | 37,7 % | 384,3 % |
| CAGR | 8,6 % | 6,5 % | 4,1 % | 2,0 % | 10,4 % |
| Sharpe | 0,58 | 0,47 | 0,31 | 0,21 | 0,61 |
| Max Drawdown | −35,5 % | −36,3 % | −54,7 % | −82,6 % | −58,5 % |
| Trefferquote | 67,6 % | 67,0 % | – | – | – |
| Profit Factor | 1,24 | 1,18 | – | – | – |
| Exposure | 75,2 % | 75,2 % | – | – | – |

Volatilität 16,5 % (SPY 18,0 %), Sortino 0,85, Calmar 0,24. Die Baselines sind ohne Kosten.

| Marktphase | Strategie (Rendite / Max DD) | SPY | QQQ | Universum |
| --- | --- | --- | --- | --- |
| 2000–2002 | −9,6 % / −28,1 % | −38,1 % / −46,5 % | −72,8 % / −82,6 % | +5,3 % / −30,9 % |
| 2008–2009 | −6,0 % / −35,0 % | −19,4 % / −51,7 % | −9,0 % / −48,5 % | −6,7 % / −55,3 % |

**Walk-Forward** (11 Fenster, 5 Jahre Training, 1 Jahr Test): verkettet CAGR 5,7 %, Sharpe
0,45, Max DD −33,8 %, 1.823 Trades. Test-Sharpe je Jahr 2005–2015: 1,16 · 0,17 · 1,07 ·
−1,82 · 1,73 · 0,68 · 0,86 · 0,21 · 1,14 · −0,27 · 0,30 (9 von 11 positiv). Gewählt wurde
trend_period 57 (8 Fenster) oder 77, nie 67; rsi_entry 15 oder 10; atr_mult 1,5 bis 2,5.

**Robustheit** (In-Sample-Sharpe, Spalten atr_mult 1,5 | 2,0 | 2,5):

| trend_period / rsi_entry | 1,5 | 2,0 | 2,5 |
| --- | ---: | ---: | ---: |
| 57 / 5 | 0,62 | 0,63 | 0,63 |
| 57 / 10 | 0,63 | 0,63 | 0,69 |
| 57 / 15 | 0,62 | 0,75 | 0,75 |
| 67 / 5 | 0,50 | 0,57 | 0,64 |
| 67 / 10 | 0,55 | **0,58** | 0,63 |
| 67 / 15 | 0,60 | 0,58 | 0,70 |
| 77 / 5 | 0,52 | 0,58 | 0,60 |
| 77 / 10 | 0,63 | 0,63 | 0,69 |
| 77 / 15 | 0,58 | 0,61 | 0,74 |

**Monte-Carlo** (1.000 Umordnungen, Seed 42): Max DD 5 % −24,5 %, Median −16,1 %,
95 % −11,7 %; tatsächlich −34,4 %. Die Verluste kommen gehäuft (vor allem 2008), der echte
Drawdown ist schlechter als jede Umordnung.

### Einordnung

- R-04 schlägt SPY und QQQ in Rendite und Sharpe und verliert in beiden Krisen deutlich
  weniger. Gegen das gleichgewichtete Universum (ohne Kosten) liegt die Rendite niedriger
  (CAGR 8,6 % gegenüber 10,4 %) bei fast gleichem Sharpe, aber mit viel kleinerem Drawdown.
- Walk-Forward (Sharpe 0,45) liegt unter In-Sample (0,58); 2008 ist klar negativ.
- Der Standardsatz liegt im Raster eher in der unteren Hälfte. Ein weiterer Stop
  (atr_mult 2,5) war fast überall besser. Nach den Regeln wird der Standardsatz trotzdem
  unverändert in die Out-of-Sample-Prüfung gegeben; eine Variante müsste offengelegt werden.

## Erster Lauf mit 1.000 USD und Ursache

Der erste In-Sample-Lauf (23.11.2026) mit 1.000 USD Startkapital endete bei −99,5 %
(Konto Ende 2002 bei 5,40 USD, Trefferquote 27 %, 430 Trades). Die Analyse der Trades
2000–2003 auf derselben Datenbank ergab:

- Von 994 USD Verlust bis Ende 2002 waren 860 USD Gebühren. Bei 10 Positionen zu je rund
  100 USD kosten Kauf und Verkauf zusammen etwa 2 %; mit sinkendem Konto steigt der Anteil
  (2002 standen 2 USD Gebühr gegen Positionen von wenigen Cent). 2.179 Käufe wurden wegen
  fehlenden Kapitals abgelehnt.
- Dieselben Regeln 2000–2003: ohne Gebühr +38,2 % (Trefferquote 65 %), mit 100.000 USD und
  1 USD Gebühr +36,3 %.
- Entscheidung der Person: Studien rechnen mit 100.000 USD (`run_study.py --capital`,
  Standard seit PR #98); der App-Standard bleibt 1.000 USD. Mit 1.000 USD Konto wäre die
  Strategie in dieser Form wegen der Fixgebühr nicht handelbar.

Nebenbefunde aus der Analyse:

- Der Stop wird auf der Einstiegskerze gesetzt und greift erst ab der folgenden Kerze; die
  erste 3D-Kerze ist ungeschützt (PTC fiel am 03.04.2000 am Einstiegstag um 49 %).
- Die Stops lagen 2000 weit weg (Median 11,8 % unter dem Einstieg, max. 46 %), weil die ATR
  nach den Crash-Kerzen hoch war.
- Unter TWX liegen vor 2001 vermutlich AOL-Kurse (Tiingo).
- Die Trade-Liste zeigt als Datum das Ende der 3D-Kerze; ausgeführt wird zum Open ihres
  ersten Tages.

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
