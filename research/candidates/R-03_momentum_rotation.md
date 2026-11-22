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

## Bewertung

Stand: **offen**. Die Studie braucht echte Kursdaten; die liegen nur in der lokalen
Datenbank und werden nicht ins Repo committet. Nach dem Lauf werden hier die fünf
Kriterien aus `summary.json` eingetragen.

| Kriterium | Ziel | Ergebnis |
| --- | --- | --- |
| Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe | erfüllt | offen |
| Max Drawdown kleiner als Buy and Hold SPY | erfüllt | offen |
| Mindestens 100 Trades | erfüllt | offen |
| Robustheitsmaß ≥ 60 % | erfüllt | offen |
| Positiv nach doppelten Kosten | erfüllt | offen |

Zu beachten beim Lesen: Mit 1.000 USD Startkapital und 1 USD Gebühr je Order kostet
ein Trade mit 100 USD Positionsgröße rund 2 % für Kauf und Verkauf. Der Kosten-
Stresstest ist für diesen Kandidaten deshalb besonders aussagekräftig.
