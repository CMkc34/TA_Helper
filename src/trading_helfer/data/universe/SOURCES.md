# Quellen der Indexmitglieder

Die CSV-Dateien in diesem Ordner beschreiben die Mitglieder von S&P 500 (`sp500_*`) und
Nasdaq-100 (`ndx100_*`). `*_current.csv` ist die aktuelle Liste, `*_changes.csv` enthält
jede Aufnahme (`added`) und Entfernung (`removed`) mit Datum und Quelle (`quelle`).
Symbole stehen in App-Schreibweise (`BRK.B`) und so, wie sie damals gehandelt wurden.

Gepflegt werden die Dateien mit `scripts/build_universe.py` (Wikipedia, „Selected changes“
und Bestandsliste). Die Erstbefüllung am 21.11.2026 stammt aus diesen Quellen:

| Datei | Zeitraum | Quelle (`quelle`) | Lizenz |
| --- | --- | --- | --- |
| `sp500_changes.csv` | 1996-01-22 bis 2026-08-18 | [fja05680/sp500](https://github.com/fja05680/sp500), Tagesstände „S&P 500 Historical Components & Changes (Updated).csv“ (Basis: Andreas Clenow, *Trading Evolved*, ergänzt aus Wikipedia), hier als Differenz aufeinanderfolgender Stände | MIT, © 2019–2020 Farrell J. Aultman |
| `sp500_changes.csv` | 2026-09-21 bis 2026-10-06 | Wikipedia, Bestandsliste „List of S&P 500 companies“ (Spalte „Date added“) | CC BY-SA 4.0 |
| `ndx100_changes.csv` | 2003-05-01 bis 2026-10-09 | [jmccarrell/n100tickers](https://github.com/jmccarrell/n100tickers), Jahresdateien `n100-ticker-changes-YYYY.yaml` (Nasdaq-Mitteilungen) | MIT, © Jeff McCarrell |

**Bekannte Grenzen**

- Wikipedias „Selected changes“ sind unvollständig; eine reine Rückwärtsrechnung daraus
  ergibt für 2000 nur rund 414 S&P-500-Mitglieder. Deshalb die Basis aus fja05680.
- Auch dort fehlen für 1996 bis 2004 einzelne Werte (491–494 Mitglieder statt rund 500).
  Der Plausibilitätstest erlaubt daher 490–510 bis Ende 2004 und 495–510 ab 2005.
- Nasdaq-100 ist ab 2003 belegt; davor gilt die früheste rekonstruierte Liste
  (Survivorship-Risiko, siehe `reliable_since`).
