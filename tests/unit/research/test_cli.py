"""Tests der Kommandozeile ``research/run_study.py`` (R-01)."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.core import config
from trading_helfer.data.storage import MarketStore
from trading_helfer.research.cli import default_report_dir, main, parse_grid
from trading_helfer.research.report import REPORT_FILE, SUMMARY_FILE

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bars: dict[str, pd.DataFrame]
) -> Path:
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    folder = tmp_path / "data"
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(folder))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(tmp_path / "cfg"))
    folder.mkdir(parents=True)
    with MarketStore(folder / "market.duckdb") as store:
        for symbol in ("AAA", "SPY"):
            frame = bars[symbol]
            store.upsert_bars(symbol, frame[frame["date"] >= "2008-01-01"])
    return folder


def test_parse_grid() -> None:
    assert parse_grid(["fast=5, 10", "flag=true", "mult=1.5,2"]) == {
        "fast": [5, 10],
        "flag": [True],
        "mult": [1.5, 2],
    }
    for bad in ("fast", "fast=", "=1"):
        with pytest.raises(ValueError, match="NAME=W1,W2"):
            parse_grid([bad])


def test_default_report_dir() -> None:
    path = default_report_dir("trend_sma_cross", "1W", "SP500", in_sample_only=True)
    assert path == Path("research") / "reports" / "trend_sma_cross_1W_SP500_is"
    assert default_report_dir("x", "3D", "BRK.B", in_sample_only=False).name == "x_3D_BRK-B"
    assert default_report_dir("x", "3D", "…", in_sample_only=False).name == "x_3D_liste"


def test_main_writes_report(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bericht"
    code = main(
        [
            "--strategy", "trend_sma_cross", "--symbol", "AAA", "--tf", "W1",
            "--param", "fast=5", "--param", "slow=20", "--grid", "fast=5,10",
            "--in-sample-only", "--simulations", "20", "--workers", "1",
            "--out", str(out),
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (out / REPORT_FILE).is_file()
    assert (out / SUMMARY_FILE).is_file()
    assert "… In-Sample-Backtest" in captured.out
    assert "Urteil: " in captured.out
    assert f"Bericht: {out / REPORT_FILE}" in captured.out
    assert "offen" in captured.out


def test_main_default_out_dir_and_db_option(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moved = tmp_path / "anders.duckdb"
    (data_dir / "market.duckdb").rename(moved)
    monkeypatch.chdir(tmp_path)
    code = main(
        ["--strategy", "trend_sma_cross", "--symbols", "AAA", "--db", str(moved),
         "--end", "2015-12-31", "--simulations", "10", "--workers", "1"]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().err
    report = tmp_path / "research" / "reports" / "trend_sma_cross_1W_liste" / REPORT_FILE
    assert report.is_file()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--strategy", "gibt_es_nicht", "--symbol", "AAA"], "Unbekannte Strategie"),
        (["--strategy", "trend_sma_cross", "--symbol", "AAA", "--tf", "2W"], "Zeitrahmen"),
        (["--strategy", "trend_sma_cross", "--symbol", "AAA", "--param", "x"], "NAME=WERT"),
        (["--strategy", "trend_sma_cross", "--symbol", "ZZZ"], "Keine Kurse"),
    ],
)
def test_main_errors_return_2(
    data_dir: Path, args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == 2
    assert message in capsys.readouterr().err


def test_main_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(tmp_path / "d"))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(tmp_path / "c"))
    assert main(["--strategy", "trend_sma_cross", "--symbol", "AAA"]) == 2
    assert "existiert nicht" in capsys.readouterr().err


def test_script_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(REPO / "research" / "run_study.py"), run_name="__main__")
    assert excinfo.value.code == 0
