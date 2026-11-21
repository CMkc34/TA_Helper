Fehlerhafte Strategien für die Tests des Strategie-Validators (S-02).

Jede Datei verletzt genau eine Regel; der Test erwartet die Ablehnung in dieser Stufe:

| Datei | Fehler | Stufe |
|---|---|---|
| `network.py` | importiert `socket` | Sicherheit |
| `file_access.py` | ruft `open` auf | Sicherheit |
| `lookahead.py` | Indikator nutzt `shift(-1)` (Kurs von morgen) | Lookahead-Test |
| `endless_loop.py` | Endlosschleife in `on_bar` | Probelauf (Zeitlimit) |
| `unseeded_random.py` | `default_rng()` ohne Seed | Determinismus |
| `wrong_signature.py` | `on_bar(self)` ohne `ctx` | Pflichtfelder und Typen |
