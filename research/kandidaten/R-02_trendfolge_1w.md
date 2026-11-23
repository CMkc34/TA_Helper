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

## Bewertung gegen die Kriterien (In-Sample 2000–2015)

Lauf am 23.11.2026 mit `--in-sample-only` und Standardparametern (period 20, trend 40,
market_sma 40, atr_mult 3,0, momentum 26, position_pct 10), Laufzeit rund 41 Minuten.
Out-of-Sample ist **nicht** gerechnet.

Datenbasis (23.11.2026): lokale Datenbank aus yfinance plus Tiingo für delistete Werte
(1.017 Symbole mit Kursen, 245 davon von Tiingo; 1.015 von 1.214 S&P-Mitgliedern seit 1990).
142 Werte mit Fremdkursen (neu vergebene Ticker, u. a. GR, EP, CPWR) sind automatisch
ausgeschlossen. Am Stichtag haben trotzdem nur 55 % (2000) bis 64 % (2005) der damaligen
Mitglieder Kurse; der Survivorship-Hinweis der Studie steht deshalb im Bericht, die Zahlen
sind eher zu gut als zu schlecht.

**Urteil: nicht bestanden.** Drawdown knapp größer als SPY, mit doppelten Kosten negativ.

| Kriterium | Ergebnis | Erfüllt |
| --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | nicht ausgewertet | offen |
| Max Drawdown kleiner als Buy and Hold SPY | −55,7 % gegenüber −54,6 % | nein |
| Mindestens 100 Trades | 232 | ja |
| Robustheitsmaß ≥ 60 % | 100 % (alle 27 Nachbarn Sharpe 0,01 bis 0,36) | ja |
| Positiv nach doppelten Kosten | −15,4 % | nein |

| Kennzahl | Strategie | Kosten × 2 | SPY | QQQ | Universum gleichgewichtet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gesamtrendite | 47,3 % | −15,4 % | 86,8 % | 34,8 % | 380,5 % |
| CAGR | 2,4 % | −1,0 % | 4,0 % | 1,9 % | 10,3 % |
| Volatilität p. a. | 19,4 % | 20,0 % | 18,2 % | 26,6 % | 20,2 % |
| Sharpe | 0,22 | 0,05 | 0,31 | 0,20 | 0,59 |
| Max Drawdown | −55,7 % | −57,4 % | −54,6 % | −82,7 % | −58,5 % |
| Trefferquote | 41,4 % | 38,5 % | – | – | – |
| Profit Factor | 1,15 | 0,91 | – | – | – |
| Exposure | 74,7 % | 74,3 % | – | – | – |

| Marktphase (Rendite / Max Drawdown) | Strategie | SPY | Universum gleichgewichtet |
| --- | ---: | ---: | ---: |
| Dotcom-Crash 2000–2002 | −21,7 % / −26,8 % | −38,1 % / −45,7 % | +5,1 % / −29,3 % |
| Finanzkrise 2008–2009 | −16,7 % / −44,6 % | −20,7 % / −52,1 % | −7,2 % / −55,1 % |

Walk-Forward (11 Fenster, 5 J Training / 1 J Test): CAGR 1,4 %, Sharpe 0,17, Max Drawdown
−41,3 %; Test-Sharpe je Jahr zwischen −1,96 (2008) und 1,83 (2013). Monte-Carlo (Trades
umgeordnet): Max Drawdown Median −29,6 %, 5-%-Quantil −41,8 % (nur abgeschlossene Trades,
deshalb weicht das Endkapital von 1.369 USD vom Backtest ab).

### Einordnung

- Der Marktfilter wirkt: In beiden Crashs verliert die Strategie weniger als SPY. Über den
  ganzen Zeitraum bleibt sie aber bei Rendite und Sharpe unter SPY und weit unter dem
  gleichgewichteten Universum.
- Robust (jeder Nachbar positiv), aber der Vorteil ist klein: Profit Factor 1,15 reicht
  nicht, um die Gebühren von rund 2 % je Runde bei 100-USD-Positionen zu tragen.
- Nächste offenzulegende Variante laut „Erwartungen und Risiken“: `position_pct = 20`
  (5 Positionen), damit die feste Gebühr relativ halb so schwer wiegt. Nur In-Sample prüfen.

## Variante B: 5 Positionen zu je 20 % (offengelegt)

Nach dem ersten Lauf auf Wunsch der Person geprüft: `--param position_pct=20
--max-positions 5`, sonst gleiche Parameter und Daten, nur In-Sample (23.11.2026, 39 min).

**Urteil: nicht bestanden.** Drawdown weiter größer als SPY; Rendite schlechter als die
Grundversion.

| Kriterium | Grundversion (10 × 10 %) | Variante B (5 × 20 %) |
| --- | ---: | ---: |
| Max Drawdown kleiner als SPY (−54,6 %) | −55,7 % (nein) | −56,4 % (nein) |
| Mindestens 100 Trades | 232 (ja) | 126 (ja) |
| Robustheitsmaß ≥ 60 % | 100 % (ja) | 100 % (ja, Nachbarn 0,02 bis 0,41) |
| Positiv nach doppelten Kosten | −15,4 % (nein) | +32,4 % (ja, siehe unten) |

| Kennzahl | Grundversion | Variante B |
| --- | ---: | ---: |
| Gesamtrendite | 47,3 % | 6,0 % |
| CAGR | 2,4 % | 0,4 % |
| Sharpe | 0,22 | 0,14 |
| Profit Factor | 1,15 | 1,00 |
| Dotcom-Crash 2000–2002 | −21,7 % / −26,8 % | −22,2 % / −34,9 % |
| Finanzkrise 2008–2009 | −16,7 % / −44,6 % | −12,3 % / −45,6 % |
| Walk-Forward CAGR / Sharpe / Max DD | 1,4 % / 0,17 / −41,3 % | 3,5 % / 0,28 / −42,5 % |

Mit doppelten Kosten ist Variante B **besser** (+32,4 %) als mit normalen Kosten (+6,0 %).
Das ist Pfadabhängigkeit: Weniger Kapital führt zu anderen Käufen (115 statt 126 Trades).
Mit nur 5 Plätzen hängt das Ergebnis an wenigen Trades; das erfüllte Kostenkriterium ist
daher kein Beleg für Robustheit. Die Gebühr wiegt zwar relativ halb so schwer, dafür fehlt
die Streuung.

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
