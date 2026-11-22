-- 006_symbol_exclusions: Werte, deren Kurse Backtests nicht verwenden (Datenqualität).
-- source: 'auto' (Prüfung auf Fremdkurse, wird bei jedem Datenupdate neu bewertet),
-- 'manuell' (von Hand ausgeschlossen) oder 'freigegeben' (von Hand erlaubt, die
-- automatische Prüfung schließt den Wert dann nicht mehr aus).
CREATE TABLE symbol_exclusions (
    symbol      VARCHAR   PRIMARY KEY,
    reason      VARCHAR   NOT NULL,
    source      VARCHAR   NOT NULL,
    created_at  TIMESTAMP NOT NULL
);
