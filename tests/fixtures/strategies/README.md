# Strategie-Fixtures

`SYNTH_long_daily.csv`: synthetische, bereits bereinigte Tageskurse (keine echten
Marktdaten) an allen Werktagen vom 1995-01-02 bis 2009-12-31 (3914 Zeilen). Lang genug,
damit auch die SMA 200 auf 3D und die SMA 40 auf 1M anlaufen.

Erzeugt mit `numpy.random.default_rng(7)` als geometrischer Zufallsweg (Tagesvolatilität
1,2 %, Start 50) mit Phasen unterschiedlicher Drift je Handelstag:

| Tage | Drift | Phase |
|------|-------|-------|
| 900 | +0,07 % | Aufwärtstrend |
| 500 | 0 | Seitwärts |
| 250 | −0,15 % | Crash |
| 600 | +0,09 % | Erholung |
| 500 | 0 | Seitwärts |
| Rest | +0,05 % | Aufwärtstrend |

Open = Vortages-Close mal kleiner Gap, High/Low mit zufälliger Spanne, Volumen gleichverteilt.
Die Trade-Anzahlen der Beispielstrategien darauf stehen als Regressionswerte in
`tests/unit/test_builtin_strategies.py`.
