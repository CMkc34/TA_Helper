"""Gemeinsame synthetische Kurse und Studien für die Tests des Forschungsrahmens (R-01)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.config import Settings
from trading_helfer.research.study import StudyConfig, StudyResult, run_study
from trading_helfer.strategies.builtin.sma_cross import SmaCross


def wave_bars(
    phase: float, drift: float, amp: float, period: float, start: str = "1995-01-02"
) -> pd.DataFrame:
    """Werktägliche Kurse mit Trend und Welle (deterministisch, synthetisch)."""
    dates = pd.bdate_range(start, "2023-12-29")
    n = np.arange(len(dates))
    close = 50.0 * np.exp(drift * n) * (1 + amp * np.sin(n / period + phase))
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


@pytest.fixture(scope="session")
def bars() -> dict[str, pd.DataFrame]:
    """Drei Werte plus die Benchmarks SPY und QQQ, 1995 bis 2023."""
    return {
        "AAA": wave_bars(0.0, 0.0003, 0.15, 60.0),
        "BBB": wave_bars(1.0, 0.0002, 0.2, 45.0),
        "CCC": wave_bars(2.0, 0.0001, 0.1, 80.0),
        "SPY": wave_bars(3.0, 0.00025, 0.08, 120.0),
        "QQQ": wave_bars(4.0, 0.0003, 0.12, 100.0),
    }


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(data_dir=tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def full_study(bars: dict[str, pd.DataFrame], settings: Settings) -> StudyResult:
    return run_study(
        SmaCross,
        {"fast": 10, "slow": 40},
        "W1",
        settings,
        symbol="AAA",
        bars=bars,
        config=StudyConfig(workers=1, mc_simulations=200),
    )


@pytest.fixture(scope="session")
def report_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("report")
