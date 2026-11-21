# Chart-Fixtures

`SYNTH_daily.csv`: synthetische Tageskurse (keine echten Marktdaten) für das Symbol
`SYNTH`, alle XNYS-Handelstage vom 2023-01-03 bis 2024-11-20 (475 Zeilen).

- Zufallsweg mit Seed 19; Split 4:1 am 2024-06-10 (rohe Kurse davor viermal so hoch),
  Dividendenfaktor 0,99 vor dem 2024-03-01. `adj_close` ist durchgehend bereinigt.
- Das Ende mitten in Woche und Monat erzeugt unvollständige letzte Kerzen in 3D, 1W, 1M.
- Erwartete Kerzenzahlen: 1D 475, 3D 159, 1W 99, 1M 23.
