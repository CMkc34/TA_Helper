"""R-03: Die Momentum-Rotation läuft durch die komplette Studie (synthetische Kurse)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.core.config import Settings
from trading_helfer.research.report import write_report
from trading_helfer.research.study import StudyConfig, run_study
from trading_helfer.strategies.builtin import MomentumRotation


@pytest.mark.slow
def test_momentum_rotation_full_study(
    bars: dict[str, pd.DataFrame], settings: Settings, tmp_path: Path
) -> None:
    study = run_study(
        MomentumRotation,
        {"lookback": 8, "top_n": 2},
        "W1",
        settings,
        universe=["AAA", "BBB", "CCC"],
        bars=bars,
        config=StudyConfig(workers=1, mc_simulations=100, max_positions=2),
    )
    assert study.full.result.trades.shape[0] > 0
    assert len(study.criteria) == 5
    write_report(study, tmp_path / "report")
    summary = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    assert "momentum_rotation" in json.dumps(summary)
