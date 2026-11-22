# R-02: Trendfolge auf Wochenkerzen mit Marktfilter

Kandidat aus Phase 5, geprüft nach dem festen Ablauf und den Kriterien aus
[`research/README.md`](../README.md) (R-01).

| | |
| --- | --- |
| Plugin | [`plugins/trend_breakout_mf.py`](../../plugins/trend_breakout_mf.py), ID `trend_breakout_mf`, Version 1.0.0 |
| Zeitrahmen | 1W (3D ebenfalls erlaubt) |
| Universum | S&P 500 point in time (`--index SP500`), max. 10 Positionen |
| Stand | Regeln und Parameter festgelegt; Studie mit echten Kursen steht aus (siehe unten) |

## Hypothese

Aktien, die auf Wochenbasis ein neues 20-Wochen-Hoch erreichen und über ihrem
40-Wochen-Durchschnitt liegen, setzen den Trend häufiger fort, als sie umkehren. In
Abwärtsmärkten scheitern Ausbrüche dagegen überdurchschnittlich oft. Ein Marktfilter
(SPY über seiner SMA 40W) soll deshalb vor allem die großen Drawdowns von 2000–2002 und
2008–2009 vermeiden, ohne die Trends in Aufwärtsmärkten zu verpassen. Der ATR-Trailing-Stop
lässt Gewinner laufen und begrenzt Verlierer.

## Regeln

Alle Signale auf der abgeschlossenen Wochenkerze `t`, Ausführung zum Open der nächsten Kerze.

| Teil | Regel |
| --- | --- |
| Marktfilter | Nur neue Käufe, wenn SPY-Wochenclose > SMA 40 der SPY-Wochencloses (`market_sma`) |
| Einstieg | Close > höchstes Hoch der vorherigen 20 Wochen (Donchian, ohne die aktuelle Kerze) **und** Close > eigene SMA 40W; nur Indexmitglieder am Tag `t` |
| Rangfolge | Mehr Signale als freie Plätze: höhere 26-Wochen-Rendite zuerst (`strength`) |
| Positionsgröße | 10 % des aktuellen Kapitals je Position, höchstens 10 Positionen |
| Ausstieg | Nur Trailing Stop 3 × ATR(14) unter dem Close, gesetzt auf der Einstiegskerze, danach nur nachgezogen. Der Marktfilter schließt keine offenen Positionen. |
| Richtung | Nur Long |

Parameter (Standard; Robustheit und Walk-Forward variieren nur die drei Kernparameter mit Schrittweite um ±1 `step`, also 27 Nachbarn):

| Parameter | Standard | Schritt | Bedeutung |
| --- | --- | --- | --- |
| `period` | 20 | 5 | Kerzen für das Ausbruchshoch |
| `trend` | 40 | fest | Trend-SMA des Werts |
| `market_sma` | 40 | 10 | SMA des Markts (SPY) |
| `atr_mult` | 3,0 | 0,5 | Abstand des Trailing Stops in ATR(14) |
| `momentum` | 26 | fest | Kerzen für die Rangfolge |
| `position_pct` | 10 | fest | Anteil am Kapital je Position in % |

Die Standardwerte sind vorab aus der Literatur zur Wochen-Trendfolge gewählt (20/40 Wochen,
3 ATR) und nicht auf Kursdaten optimiert. So bleibt das Out-of-Sample unberührt.

## Ablauf der Studie (lokal mit der eigenen Kursdatenbank)

Die Cloud-Umgebung hat keine echten Kursdaten (sie werden nicht ins Repo committet). Die
Studie läuft deshalb auf dem eigenen Rechner:

1. Plugin in den Plugin-Ordner der App kopieren (Standard `<Datenordner>/plugins`) oder im
   Strategien-Tab über „Hinzufügen“ einlesen. SPY, QQQ und die S&P-500-Werte müssen ab 1990
   geladen sein.
2. In-Sample (2000–2015) prüfen:

   ```bash
   uv run python research/run_study.py --strategy trend_breakout_mf --tf W1 --index SP500 --in-sample-only
   ```

3. Nur wenn die In-Sample-Kriterien (Drawdown, Trades, Robustheit, Kosten) erfüllt sind,
   **einmal** die vollständige Studie mit Out-of-Sample ausführen:

   ```bash
   uv run python research/run_study.py --strategy trend_breakout_mf --tf W1 --index SP500
   ```

4. `research/reports/trend_breakout_mf_W1_SP500/` committen (nur abgeleitete Kennzahlen) und
   die Tabelle unten aus `summary.json` ausfüllen.

Ändert sich nach Schritt 3 ein Parameter, wird das hier unter „Abweichungen“ offengelegt.

## Bewertung gegen die Kriterien

| Kriterium | Schwelle | Ergebnis | Erfüllt |
| --- | --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | IS-Sharpe > 0 | ausstehend | offen |
| Max Drawdown kleiner als Buy and Hold SPY | Gesamtzeitraum | ausstehend | offen |
| Mindestens 100 Trades | Gesamtzeitraum | ausstehend | offen |
| Robustheitsmaß ≥ 60 % | Nachbarparameter mit IS-Sharpe > 0 | ausstehend | offen |
| Positiv nach doppelten Kosten | Gesamtrendite im Kosten-Stresstest | ausstehend | offen |

**Urteil:** offen, bis die Studie mit echten Kursen gelaufen ist.

## Was vorab schon geprüft ist

- Das Plugin besteht alle Stufen des Validators, auch den Lookahead-Test für die
  Marktindikatoren (die Testkurse dienen im Validator als Markt).
- Auf synthetischen Kursen (Tests in `tests/unit/research/test_r02_trend_breakout.py`):
  kein Einstieg, wenn SPY auf der Signalkerze unter seiner SMA 40W lag (ohne Filter gibt es
  solche Einstiege), Ausstiege nur über den ATR-Trailing-Stop, höchstens 10 Positionen, die
  Studie läuft vollständig durch und findet die SPY-Kurse für den Filter.

## Erwartungen und Risiken

- **Kosten:** Mit 1.000 USD Startkapital und 10 % je Position sind Orders etwa 100 USD groß.
  1 USD Gebühr je Order kostet dann rund 2 % je Runde. Das Kriterium „positiv nach doppelten
  Kosten“ ist deshalb die größte Hürde; falls es scheitert, ist `position_pct = 20` (5
  Positionen) die naheliegende, offenzulegende Variante.
- **Trades:** Wochen-Trendfolge über 500 Werte und 10 Plätze ergibt typischerweise einige
  hundert Trades in 24 Jahren; das 100-Trade-Kriterium sollte halten.
- **Seitwärtsmärkte** (z. B. 2015, 2018) erzeugen viele Fehlausbrüche, die der Marktfilter
  nicht abfängt.
- **Survivorship:** Die Indexhistorie ist point in time; Werte ohne Kurse in der Datenbank
  fehlen trotzdem und verzerren nach oben (Hinweis im Protokoll der Studie beachten).

## Abweichungen

Keine.
