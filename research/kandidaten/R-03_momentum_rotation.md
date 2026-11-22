# Kandidat R-03: Momentum-Rotation 1W

Strategie `momentum_rotation` (eingebaut, `src/trading_helfer/strategies/builtin/momentum_rotation.py`).
Geprüft wird nach dem festen Ablauf und den Kriterien aus [`research/README.md`](../README.md).

## Hypothese

Aktien mit der stärksten Rendite der letzten sechs Monate laufen in den folgenden Wochen
im Schnitt weiter besser als der Index (Querschnitts-Momentum, Jegadeesh/Titman 1993).
Die jüngste Woche bleibt außen vor, weil kurzfristig eher eine Gegenbewegung folgt.
Der Marktfilter soll die großen Abstürze (2000–2002, 2008) vermeiden, in denen
Momentum-Portfolios besonders stark verlieren.

## Regeln

| Schritt | Regel |
| --- | --- |
| Rangliste | Jede Woche alle Werte, die an diesem Tag Indexmitglied sind, nach `Close[t−1] / Close[t−27] − 1` (26 Wochen ohne letzte Woche). RS-Rang 0 bis 100 wie in X-03 (`rs_rank`), 100 = stärkster Wert |
| Kauf | Wert steht unter den besten 10, RS-Rang ≥ 30 und SPY schließt über seiner SMA 40W. Größe 10 % des Kapitals, Ausführung am Open der Folgewoche. Sind weniger Plätze frei als Kandidaten, gewinnt die höhere 26-Wochen-Rendite |
| Verkauf | RS-Rang fällt unter 30, oder der Wert ist nicht mehr im Index |
| Marktfilter | Sperrt nur neue Käufe (wie R-02); bestehende Positionen laufen nach ihren eigenen Regeln weiter |

„Verkauf bei Rang unter 30 %“ ist wörtlich als RS-Rang < 30 umgesetzt: ein Wert wird
gehalten, solange er nicht ins schwächste Drittel des Index fällt. Der Puffer zwischen
Kauf (Top 10, etwa Rang ≥ 98) und Verkauf (Rang < 30) hält den Umschlag und damit die
Kosten niedrig. Wer die engere Lesart „Verkauf, sobald der Wert die besten 30 % verlässt“
prüfen will, setzt `--param exit_rank=70`.

## Parameter

| Parameter | Standard | Im Robustheitsraster | Bedeutung |
| --- | --- | --- | --- |
| `lookback` | 26 | 22, 26, 30 | Kerzen für die Momentum-Rendite |
| `skip` | 1 | fest | Jüngste Kerzen ohne Wertung |
| `top_n` | 10 | fest | Kauf nur unter den besten N; zusammen mit `--max-positions 10` |
| `exit_rank` | 30 | 20, 30, 40 | Verkauf unter diesem RS-Rang |
| `market_sma` | 40 | 35, 40, 45 | Periode der SMA auf SPY |
| `market_filter` | ja | fest | Marktfilter an/aus |

`skip` und `top_n` haben keinen `step`, damit das Raster bei 27 Kombinationen bleibt.

## Studie ausführen

Voraussetzung: lokale Datenbank mit den S&P-500-Werten seit 1990 und SPY/QQQ
(Daten-Tab oder `trading-helfer` Datenaktualisierung).

```bash
# 1. Entwickeln, nur In-Sample (2000–2015)
uv run python research/run_study.py --strategy momentum_rotation --tf W1 --index SP500 \
    --max-positions 10 --in-sample-only

# 2. Einmal am Ende: vollständige Studie mit Out-of-Sample (ab 2016)
uv run python research/run_study.py --strategy momentum_rotation --tf W1 --index SP500 \
    --max-positions 10
```

Bericht: `research/reports/momentum_rotation_W1_SP500[_is]/`.

## Bewertung (In-Sample 2000–2015)

Lauf am 22.11.2026 mit `--in-sample-only` und Standardparametern auf einer frisch per
yfinance aufgebauten Datenbank (Laufzeit der Studie gut 9 Minuten). Out-of-Sample ist
**nicht** gerechnet.

**Urteil: nicht bestanden.** Der Max Drawdown ist größer als bei Buy and Hold SPY.

| Kriterium | Ergebnis | Erfüllt |
| --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | nicht ausgewertet | offen |
| Max Drawdown kleiner als Buy and Hold SPY | −61,4 % gegenüber −54,6 % | nein |
| Mindestens 100 Trades | 169 | ja |
| Robustheitsmaß ≥ 60 % | 100 % (alle 27 Nachbarn Sharpe 0,31 bis 0,51) | ja |
| Positiv nach doppelten Kosten | +157,8 % | ja |

| Kennzahl | Strategie | Kosten × 2 | SPY | QQQ | Universum gleichgewichtet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gesamtrendite | 251,1 % | 157,8 % | 86,8 % | 34,8 % | 488,3 % |
| CAGR | 8,1 % | 6,1 % | 4,0 % | 1,9 % | 11,7 % |
| Volatilität p. a. | 26,1 % | 26,2 % | 18,2 % | 26,6 % | 19,9 % |
| Sharpe | 0,43 | 0,36 | 0,31 | 0,20 | 0,66 |
| Max Drawdown | −61,4 % | −63,4 % | −54,6 % | −82,7 % | −56,5 % |
| Trefferquote | 45,0 % | 44,3 % | – | – | – |
| Profit Factor | 1,50 | 1,33 | – | – | – |

| Marktphase (Rendite / Max Drawdown) | Strategie | SPY | Universum gleichgewichtet |
| --- | ---: | ---: | ---: |
| Dotcom-Crash 2000–2002 | −5,7 % / −38,3 % | −38,1 % / −45,7 % | +8,0 % / −28,3 % |
| Finanzkrise 2008–2009 | −41,2 % / −58,1 % | −20,7 % / −52,1 % | −4,3 % / −53,7 % |

Walk-Forward (11 Fenster, 5 J Training / 1 J Test): CAGR 6,6 %, Sharpe 0,39, Max
Drawdown −63,3 %; Test-Sharpe je Jahr 2005–2015 zwischen −2,64 und 2,48. Monte-Carlo
(Trades umgeordnet): Max Drawdown Median −30,6 %, 5-%-Quantil −44,9 %.

### Einordnung

- Die Strategie schlägt SPY und QQQ bei Rendite und Sharpe, aber nicht das
  gleichgewichtete Universum (Sharpe 0,43 gegen 0,66). Momentum bringt hier also keinen
  Mehrwert gegenüber dem einfachen Halten aller Mitglieder.
- Der Drawdown entsteht vor allem 2008–2009 (−58 %). Der Marktfilter sperrt nur neue
  Käufe; gehaltene Positionen fallen weiter, bis ihr RS-Rang unter 30 sinkt.
- 849 Kauforders scheiterten an fehlendem Kapital (10 % des Kapitals > freies Cash).
  Das Portfolio war deshalb selten mit 10 vollen Positionen besetzt.

### Datenqualität (wichtig für alle Zahlen oben)

- Die Datenbank kommt nur aus yfinance. Von den 871 Mitgliedern 2000–2015 haben 523 Kurse;
  am Stichtag hatten 2000 rund 47 %, 2005 rund 53 %, 2010 rund 65 % und 2015 rund 75 %
  der damaligen Mitglieder Kurse. Es fehlen vor allem ausgeschiedene und übernommene
  Werte, das lässt Strategie und gleichgewichtetes Universum zu gut aussehen
  (Survivorship-Bias). Der Survivorship-Hinweis der Engine erscheint nur für Starts
  vor 2000 und fehlte deshalb im Bericht.
- GR, EP und CPWR haben wahrscheinlich Kurse einer anderen Firma unter demselben Ticker
  (viele Tagessprünge über 80 %). Sie wurden 12-mal gehandelt. Ohne diese drei Werte:
  Endwert 4.302 statt 3.511 USD, Max Drawdown −65,8 %; das Drawdown-Kriterium bleibt
  verfehlt.

### Nächste Schritte

Keine Parameter nach diesem Lauf verändert. Naheliegende, als neue Variante
offenzulegende Ideen (nur In-Sample prüfen): Positionen schließen, wenn der Marktfilter
abschaltet; Positionsgröße aus dem freien Platz statt aus 10 % des Kapitals.
Aussagekräftige Zahlen brauchen eine Datenquelle mit delisteten Werten (Tiingo oder
Kaggle-Import).
