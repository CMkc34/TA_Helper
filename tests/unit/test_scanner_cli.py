"""Kommandozeile ``trading-helfer scan`` (S-04)."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.app import main
from trading_helfer.core import config
from trading_helfer.data.storage import MarketStore
from trading_helfer.scanner.cli import parse_filters, parse_params
from trading_helfer.scanner.scan import load_signals

PLUGIN = textwrap.dedent(
    """
    from trading_helfer.core.types import Timeframe
    from trading_helfer.strategies.base import BarContext, Order, Param, Strategy


    class ShortProbe(Strategy):
        id = "short_probe"
        name = "Short-Probe"
        version = "1.0.0"
        description = "Test"
        timeframes = [Timeframe.W1]
        params = {"qty": Param(1.0, 0.5, 10.0, 0.5, "Stück")}

        def on_bar(self, ctx: BarContext) -> list[Order]:
            return [ctx.sell_short(qty=1)]
    """
)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    folder = tmp_path / "data"
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(folder))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(tmp_path / "cfg"))
    plugins = folder / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "short_probe.py").write_text(PLUGIN, encoding="utf-8")
    dates = pd.bdate_range("2015-01-02", periods=120)
    bars = pd.DataFrame(
        {"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 5}
    )
    with MarketStore(folder / "market.duckdb") as store:
        store.upsert_bars("AAA", bars)
        store.upsert_bars("BBB", bars)
    return folder


def test_cli_scan_prints_saves_and_exports(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "scan.csv"
    code = main(
        [
            "scan",
            "--strategy",
            "short_probe",
            "--symbols",
            "AAA",
            "BBB",
            "--tf",
            "W1",
            "--as-of",
            "2015-03-06",
            "--param",
            "qty=2",
            "--csv",
            str(csv_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "2 Signal(e)." in out
    assert "Scan-ID:" in out
    assert csv_path.exists()
    with MarketStore(data_dir / "market.duckdb", read_only=True) as store:
        assert len(load_signals(store)) == 2


def test_cli_scan_index_without_data(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", "--strategy", "short_probe", "--index", "SP500", "--no-save"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Keine Signale." in out
    assert "übersprungen" in out
    assert "Scan-ID" not in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--strategy", "fehlt", "--symbols", "AAA"], "Unbekannte Strategie"),
        (["--strategy", "short_probe", "--index", "DAX"], "Unbekannter Index"),
        (["--strategy", "short_probe", "--symbols", "AAA", "--tf", "D3"], "Zeitrahmen"),
        (["--strategy", "short_probe", "--symbols", "AAA", "--param", "qty"], "NAME=WERT"),
        (["--strategy", "short_probe", "--symbols", "AAA", "--metric", "roe"], "NAME:MIN:MAX"),
    ],
)
def test_cli_scan_errors(
    data_dir: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    assert main(["scan", *extra]) == 2
    assert message in capsys.readouterr().err


def test_cli_scan_missing_database(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["scan", "--strategy", "x", "--symbols", "A", "--db", str(tmp_path / "nix.duckdb")]
    assert main(args) == 2
    assert "nicht gefunden" in capsys.readouterr().err


def test_parse_params_types() -> None:
    assert parse_params(["a=1", "b=2.5", "c=true", "d=FALSE", "e=text"]) == {
        "a": 1,
        "b": 2.5,
        "c": True,
        "d": False,
        "e": "text",
    }


def test_parse_filters() -> None:
    args = argparse.Namespace(
        sector=["Energy"],
        min_cap=1.0,
        max_cap=None,
        metric=["pe_ratio::25", "roe:0.1:"],
        signal=["BUY"],
    )
    filters = parse_filters(args)
    assert filters.sectors == frozenset({"Energy"})
    assert filters.metrics == {"pe_ratio": (None, 25.0), "roe": (0.1, None)}
    assert {s.value for s in filters.signals or ()} == {"BUY"}
    args.metric = ["roe:x:"]
    with pytest.raises(ValueError, match="keine Zahl"):
        parse_filters(args)
