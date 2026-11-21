"""Tests für den Selbsttest ohne Oberfläche (services/selftest.py, P-01)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from trading_helfer.core.config import Settings
from trading_helfer.services import selftest
from trading_helfer.services.selftest import SelfTestCheck, format_report, run_core_checks


def test_core_checks_pass(tmp_path: Path) -> None:
    checks = run_core_checks(Settings(data_dir=tmp_path / "data"), user_data_dir=tmp_path / "u")
    assert [c.name for c in checks] == [
        "Datenordner",
        "Datenbank",
        "Indexdaten",
        "Indikatoren",
        "Strategien",
        "Strategie-Prüfung",
    ]
    assert all(c.ok for c in checks), format_report(checks)
    assert (tmp_path / "data" / "market.duckdb").exists()


def test_default_data_dir_is_outside_app_dir(tmp_path: Path) -> None:
    checks = run_core_checks(Settings(data_dir=tmp_path))
    assert checks[0].ok, checks[0].detail


def test_data_dir_inside_app_dir_fails(tmp_path: Path) -> None:
    inside = selftest.app_dir() / "daten"
    check = run_core_checks(Settings(data_dir=tmp_path), user_data_dir=inside)[0]
    assert not check.ok
    assert "liegt im App-Ordner" in check.detail


def test_failing_check_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> str:
        raise ValueError("kaputt")

    monkeypatch.setattr(selftest, "_check_universe", boom)
    checks = run_core_checks(Settings(data_dir=tmp_path), user_data_dir=tmp_path / "u")
    universe = next(c for c in checks if c.name == "Indexdaten")
    assert universe == SelfTestCheck("Indexdaten", False, "ValueError: kaputt")
    assert format_report(checks).endswith("Selbsttest fehlgeschlagen.")


def test_format_report() -> None:
    text = format_report([SelfTestCheck("A", True, "gut"), SelfTestCheck("B", False, "schlecht")])
    lines = text.splitlines()
    assert lines[0].endswith("Selbsttest (Python)")
    assert lines[1] == "OK   A: gut"
    assert lines[2] == "FEHL B: schlecht"
    assert lines[-1] == "Selbsttest fehlgeschlagen."
    assert format_report([]).endswith("fehlgeschlagen.")


def test_app_dir_when_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "TradingHelfer" / "TradingHelfer.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert selftest.is_frozen()
    assert selftest.app_dir() == exe.parent.resolve()
    assert "(Paket)" in format_report([SelfTestCheck("A", True, "")])


def test_app_dir_from_source() -> None:
    assert not selftest.is_frozen()
    assert (selftest.app_dir() / "app.py").exists()


def _broken_registry_load(_self: object) -> object:
    from trading_helfer.strategies.registry import LoadError, LoadErrorKind, LoadReport

    return LoadReport((), (LoadError("x.py", LoadErrorKind.EXECUTION, "kaputt"),))


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ("errors", "kaputt"),
        ("empty", "keine eingebauten Strategien"),
        ("no_source", "Quelltext fehlt"),
    ],
)
def test_strategy_check_failures(monkeypatch: pytest.MonkeyPatch, patch: str, message: str) -> None:
    from trading_helfer.backtest import results
    from trading_helfer.strategies.registry import LoadReport, StrategyRegistry

    if patch == "errors":
        monkeypatch.setattr(StrategyRegistry, "load", _broken_registry_load)
    elif patch == "empty":
        monkeypatch.setattr(StrategyRegistry, "load", lambda _self: LoadReport())
    else:
        monkeypatch.setattr(results, "strategy_source_hash", lambda _s: None)
    with pytest.raises(RuntimeError, match=message):
        selftest._check_strategies()


def test_database_check_detects_wrong_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_helfer.data.storage import MarketStore

    monkeypatch.setattr(MarketStore, "schema_version", lambda _self: 0)
    with pytest.raises(RuntimeError, match="Schema-Version 0"):
        selftest._check_database(Settings(data_dir=tmp_path))


def test_universe_and_indicator_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_helfer.data import universe
    from trading_helfer.indicators import registry

    empty = universe.IndexHistory("SP500", frozenset(), ())
    monkeypatch.setattr(universe, "load_history", lambda _code: empty)
    with pytest.raises(RuntimeError, match="keine Mitglieder"):
        selftest._check_universe()
    monkeypatch.setattr(registry, "list_all", list)
    with pytest.raises(RuntimeError, match="keine Indikatoren"):
        selftest._check_indicators()


def test_validator_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_helfer.strategies import validator

    report = validator.ValidationReport(
        "x.py",
        (
            validator.StageResult(
                validator.ValidationStage.SYNTAX, validator.StageStatus.FAILED, "kaputt"
            ),
        ),
    )
    monkeypatch.setattr(validator, "validate", lambda *_a, **_k: report)
    with pytest.raises(RuntimeError, match="abgelehnt"):
        selftest._check_validator()
