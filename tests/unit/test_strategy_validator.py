"""Tests des Strategie-Validators (S-02)."""

from __future__ import annotations

import pkgutil
import textwrap
from importlib import import_module
from pathlib import Path

import pytest

from trading_helfer.strategies import builtin
from trading_helfer.strategies.validator import (
    StageResult,
    StageStatus,
    ValidationReport,
    ValidationStage,
    synthetic_bars,
    validate,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOOD = sorted((FIXTURES / "good_strategies").glob("*.py"))
BAD = FIXTURES / "bad_strategies"

S = ValidationStage

HEADER = """\
import math

import numpy as np
import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy
"""


def _source(body: str) -> str:
    return HEADER + "\n\n" + textwrap.dedent(body)


def _strategy(on_bar: str = "return []", extra: str = "", cls_extra: str = "") -> str:
    """Minimale Strategie; ``on_bar`` und weitere Methoden als Quelltext."""
    body = f"""
class Demo(Strategy):
    id = "demo"
    name = "Demo"
    version = "1.0.0"
    description = "Test"
    timeframes = [Timeframe.W1]
{textwrap.indent(textwrap.dedent(cls_extra), "    ")}
{textwrap.indent(textwrap.dedent(extra), "    ")}
    def on_bar(self, ctx: BarContext) -> list[Order]:
{textwrap.indent(textwrap.dedent(on_bar), "        ")}
"""
    return HEADER + "\n" + body


def _check(source: str, filename: str = "demo.py") -> ValidationReport:
    return validate(source, filename=filename, isolate=False)


def _assert_rejected(report: ValidationReport, stage: ValidationStage) -> StageResult:
    assert not report.ok
    assert report.failed_stage is stage, report.format()
    result = report.stage(stage)
    assert result.message
    order = list(ValidationStage)
    for other in report.stages:
        if order.index(other.stage) < order.index(stage):
            assert other.status is StageStatus.PASSED, report.format()
        elif order.index(other.stage) > order.index(stage):
            assert other.status is StageStatus.SKIPPED, report.format()
    return result


# ---------------------------------------------------------------------------
# Akzeptanzkriterien
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", GOOD, ids=lambda p: p.stem)
def test_good_example_strategies_pass(path: Path) -> None:
    report = _check(path.read_text("utf-8"))
    assert report.ok, report.format()
    assert report.failed_stage is None
    assert [r.stage for r in report.stages] == list(ValidationStage)
    assert len(report.strategy_ids) == 1
    assert "alle Stufen bestanden" in report.summary()


def _builtin_sources() -> list[Path]:
    package = Path(builtin.__file__).parent
    return sorted(
        package / f"{m.name}.py"
        for m in pkgutil.iter_modules([str(package)])
        if not m.name.startswith("_")
    )


@pytest.mark.parametrize("path", _builtin_sources(), ids=lambda p: p.stem)
def test_builtin_strategies_pass(path: Path) -> None:
    """Alle mitgelieferten Strategien (B-07) bestehen den Validator."""
    report = _check(path.read_text("utf-8"), filename=str(path))
    assert report.ok, report.format()


@pytest.mark.parametrize(
    ("name", "stage", "line"),
    [
        ("network.py", S.SECURITY, 3),
        ("file_access.py", S.SECURITY, 21),
        ("lookahead.py", S.LOOKAHEAD, 17),
        ("unseeded_random.py", S.DETERMINISM, 18),
        ("wrong_signature.py", S.FIELDS, 14),
    ],
)
def test_bad_strategies_rejected_in_right_stage(
    name: str, stage: ValidationStage, line: int
) -> None:
    path = BAD / name
    report = validate(path.read_text("utf-8"), filename=str(path), isolate=False)
    result = _assert_rejected(report, stage)
    assert result.line == line
    assert f"Zeile {line}" in report.summary()


def test_bad_endless_loop_rejected_by_time_limit() -> None:
    path = BAD / "endless_loop.py"
    report = validate(path.read_text("utf-8"), filename=str(path), timeout=2.0)
    result = _assert_rejected(report, S.TRIAL_RUN)
    assert "Zeitlimit von 2 s" in result.message
    assert result.line == 16
    assert report.strategy_ids == ("bad_endless_loop",)


@pytest.mark.parametrize("name", ["network.py", "unseeded_random.py", "lookahead.py"])
def test_isolated_mode_gives_same_stage(name: str) -> None:
    source = (BAD / name).read_text("utf-8")
    isolated = validate(source, filename=name)
    local = validate(source, filename=name, isolate=False)
    assert isolated.failed_stage is local.failed_stage
    assert isolated.stage(isolated.failed_stage).line == local.stage(local.failed_stage).line  # type: ignore[arg-type]


def test_isolated_mode_passes_good_strategy() -> None:
    report = validate(GOOD[0].read_text("utf-8"), filename=GOOD[0].name)
    assert report.ok, report.format()
    assert report.strategy_ids


# ---------------------------------------------------------------------------
# Syntax und Sicherheit
# ---------------------------------------------------------------------------


def test_syntax_error_with_line() -> None:
    report = _check("x = 1\ndef broken(:\n    pass\n")
    result = _assert_rejected(report, S.SYNTAX)
    assert result.line == 2
    assert "Syntaxfehler" in result.message


def test_null_bytes_are_syntax_error() -> None:
    _assert_rejected(_check("x = 1\0"), S.SYNTAX)


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "import requests",
        "import numpy.ctypeslib",
        "from pathlib import Path",
        "from . import nachbar",
        "from numpy import load",
        "from trading_helfer.strategies.base import *",
        "x = eval('1')",
        "exec('x = 1')",
        "m = __import__('os')",
        "f = open",
        "g = getattr(pd, 'read_csv')",
        "b = __builtins__",
        "df = pd.read_csv('kurse.csv')",
        "pd.DataFrame().to_csv('x.csv')",
        "np.save('x.npy', np.zeros(1))",
        "n = np.load('x.npy')",
        "c = (1).__class__.__subclasses__()",
        "p = pd.io",
    ],
)
def test_forbidden_code_rejected(code: str) -> None:
    report = _check(_source(code + "\n"))
    result = _assert_rejected(report, S.SECURITY)
    assert result.line is not None
    assert result.line > HEADER.count("\n")


def test_all_security_issues_listed() -> None:
    report = _check(_source("import os\nimport socket\nx = eval('1')\n"))
    result = _assert_rejected(report, S.SECURITY)
    assert "2 weitere" in result.message
    assert len(result.details) == 3
    assert all(d.startswith("Zeile ") for d in result.details)


@pytest.mark.parametrize(
    "code",
    [
        "from typing import ClassVar",
        "from __future__ import annotations",
        "from trading_helfer.indicators.trend import sma, ema",
        "import trading_helfer.indicators.momentum as mom",
        "from trading_helfer.core.types import OrderType, Side",
        "x = np.lib.stride_tricks.sliding_window_view(np.arange(5), 2)",
        "name = __name__",
        '__all__ = ["x"]',
    ],
)
def test_allowed_code_passes_security(code: str) -> None:
    report = _check(_source(code + "\n"))
    assert report.stage(S.SECURITY).ok, report.format()


# ---------------------------------------------------------------------------
# Pflichtfelder und Typen
# ---------------------------------------------------------------------------


def test_no_strategy_class() -> None:
    result = _assert_rejected(_check(_source("x = 1\n")), S.FIELDS)
    assert "Keine Strategie" in result.message


def test_abstract_strategy_without_on_bar() -> None:
    source = _source(
        """
        class Half(Strategy):
            id = "half"
            name = "Halb"
            version = "1.0.0"
            description = "ohne on_bar"
            timeframes = [Timeframe.W1]
        """
    )
    result = _assert_rejected(_check(source), S.FIELDS)
    assert "on_bar" in result.message
    assert result.line is not None


def test_error_while_loading_has_line() -> None:
    source = _strategy() + "\nx = 1 / 0\n"
    result = _assert_rejected(_check(source), S.FIELDS)
    assert "ZeroDivisionError" in result.message
    assert result.line == source.count("\n")


def test_missing_field() -> None:
    source = _strategy().replace('    description = "Test"\n', "")
    result = _assert_rejected(_check(source), S.FIELDS)
    assert "description" in result.message


def test_invalid_semver() -> None:
    source = _strategy().replace('"1.0.0"', '"eins"')
    result = _assert_rejected(_check(source), S.FIELDS)
    assert "SemVer" in result.message


def test_invalid_param_definition_type() -> None:
    source = _strategy(cls_extra='params = {"n": 5}')
    result = _assert_rejected(_check(source), S.FIELDS)
    assert "params" in result.message


def test_duplicate_ids_in_one_file() -> None:
    first = _strategy()
    second = first.replace(HEADER, "").replace("class Demo", "class Demo2")
    result = _assert_rejected(_check(first + second), S.FIELDS)
    assert "mehrfach" in result.message


@pytest.mark.parametrize(
    "extra",
    [
        "def indicators(self, bars):\n    return pd.DataFrame(index=bars.index)\n",
        "def indicators(self, bars, p, extra):\n    return pd.DataFrame(index=bars.index)\n",
        "def plot(self, what):\n    return []\n",
        "@staticmethod\ndef plot():\n    return []\n",
    ],
)
def test_wrong_method_signatures(extra: str) -> None:
    result = _assert_rejected(_check(_strategy(extra=extra)), S.FIELDS)
    assert "erwartet" in result.message or "Methode" in result.message
    assert result.line is not None


def test_optional_extra_arguments_are_accepted() -> None:
    extra = "def indicators(self, bars, p, scale=1.0):\n    return pd.DataFrame(index=bars.index)\n"
    assert _check(_strategy(extra=extra)).ok


# ---------------------------------------------------------------------------
# Probelauf
# ---------------------------------------------------------------------------


def test_runtime_error_in_on_bar_has_line() -> None:
    source = _strategy(on_bar="x = 1\nreturn [x / 0]")
    result = _assert_rejected(_check(source), S.TRIAL_RUN)
    assert "fehlgeschlagen" in result.message
    assert result.line == source.count("\n")


def test_wrong_return_type() -> None:
    result = _assert_rejected(_check(_strategy(on_bar='return "kaufen"')), S.TRIAL_RUN)
    assert "Liste" in result.message


def test_indicator_with_wrong_length() -> None:
    extra = "def indicators(self, bars, p):\n    return pd.DataFrame({'a': [1.0]})\n"
    _assert_rejected(_check(_strategy(extra=extra)), S.TRIAL_RUN)


def test_plot_must_return_plotspecs() -> None:
    result = _assert_rejected(
        _check(_strategy(extra="def plot(self):\n    return ['linie']\n")), S.TRIAL_RUN
    )
    assert "PlotSpec" in result.message


def test_plot_with_unknown_column() -> None:
    source = _strategy(extra="def plot(self):\n    return [PlotSpec('fehlt')]\n").replace(
        "Sizing, Strategy", "PlotSpec, Sizing, Strategy"
    )
    result = _assert_rejected(_check(source), S.TRIAL_RUN)
    assert "fehlt" in result.message


def test_unexpected_exception_type_is_reported() -> None:
    extra = "def indicators(self, bars, p):\n    raise KeyError('x')\n"
    result = _assert_rejected(_check(_strategy(extra=extra)), S.TRIAL_RUN)
    assert "Indikatoren" in result.message


# ---------------------------------------------------------------------------
# Determinismus und Lookahead
# ---------------------------------------------------------------------------


def test_seeded_random_is_deterministic() -> None:
    extra = """
def indicators(self, bars, p):
    rng = np.random.default_rng(42)
    return pd.DataFrame({"coin": rng.random(len(bars))})
"""
    on_bar = """
if ctx.position is None and ctx.ind["coin"] > 0.8:
    return [ctx.buy(sizing=Sizing.percent_equity(100))]
if ctx.position is not None and ctx.ind["coin"] < 0.2:
    return ctx.exit()
return []
"""
    assert _check(_strategy(on_bar=on_bar, extra=extra)).ok


def test_unseeded_random_in_on_bar_changes_orders() -> None:
    on_bar = """
if ctx.position is None and np.random.random() > 0.8:
    return [ctx.buy(sizing=Sizing.percent_equity(100))]
if ctx.position is not None and np.random.random() < 0.2:
    return ctx.exit()
return []
"""
    result = _assert_rejected(_check(_strategy(on_bar=on_bar)), S.DETERMINISM)
    assert "verschiedene Ergebnisse" in result.message
    assert result.line is not None


def test_lookahead_in_on_bar_via_stored_bars() -> None:
    """Strategie merkt sich die ganze Reihe in indicators() und schaut in on_bar voraus."""
    extra = """
def indicators(self, bars, p):
    self.future = bars["close"].to_numpy().copy()
    return pd.DataFrame(index=bars.index)
"""
    on_bar = """
t = len(ctx.bars) - 1
if t + 1 >= len(self.future):
    return []
rising = self.future[t + 1] > self.future[t]
if ctx.position is None and rising:
    return [ctx.buy(sizing=Sizing.percent_equity(100))]
if ctx.position is not None and not rising:
    return ctx.exit()
return []
"""
    result = _assert_rejected(_check(_strategy(on_bar=on_bar, extra=extra)), S.LOOKAHEAD)
    assert "Order" in result.message


def test_centered_rolling_is_lookahead() -> None:
    extra = """
def indicators(self, bars, p):
    return pd.DataFrame({"m": bars["close"].rolling(5, center=True).mean()})
"""
    result = _assert_rejected(_check(_strategy(extra=extra)), S.LOOKAHEAD)
    assert result.line is not None
    assert "Verdächtig" in result.message


def test_access_beyond_history_is_reported() -> None:
    result = _assert_rejected(
        _check(_strategy(on_bar="ctx.bars.get('close', ago=10_000)\nreturn []")), S.TRIAL_RUN
    )
    assert "Daten" in result.message


# ---------------------------------------------------------------------------
# Bericht und Testdaten
# ---------------------------------------------------------------------------


def test_report_format_lists_every_stage() -> None:
    report = _check(_source("import os\n"))
    text = report.format()
    for stage in ValidationStage:
        assert stage.label in text
    assert "fehlgeschlagen" in text
    assert "übersprungen" in text
    assert report.strategy_ids == ()
    with pytest.raises(KeyError):
        ValidationReport("x", ()).stage(S.SYNTAX)


def test_stage_result_str() -> None:
    result = StageResult(S.SECURITY, StageStatus.FAILED, "Verboten.", line=3)
    assert str(result) == "Sicherheit (Importe und Aufrufe): fehlgeschlagen (Zeile 3) – Verboten."
    assert str(StageResult(S.SYNTAX, StageStatus.PASSED)) == "Syntax: bestanden"


def test_synthetic_bars_are_deterministic_and_valid() -> None:
    from trading_helfer.core.bars import validate_bars

    a, b = synthetic_bars(), synthetic_bars()
    assert a.equals(b)
    validate_bars(a)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert (a["close"] > 0).all()
    assert not synthetic_bars(1).equals(a)


def test_builtin_package_is_importable() -> None:
    """Sicherstellen, dass der Builtin-Test nicht wegen eines Importfehlers leer bleibt."""
    assert import_module(builtin.__name__) is builtin


def test_market_filter_strategy_passes_with_test_bars_as_market() -> None:
    extra = """
def market_indicators(self, market, p):
    return pd.DataFrame({"sma": market["close"].rolling(10).mean()})
"""
    on_bar = """
assert "market_sma" in ctx.ind
up = ctx.value("market_close") > ctx.value("market_sma")
if ctx.position is None and up:
    return [ctx.buy(sizing=Sizing.percent_equity(100))]
if ctx.position is not None and not up:
    return ctx.exit()
return []
"""
    source = _strategy(on_bar=on_bar, extra=extra, cls_extra='market_symbol = "SPY"')
    report = _check(source)
    assert report.ok, report.format()


def test_lookahead_in_market_indicators_is_found() -> None:
    extra = """
def market_indicators(self, market, p):
    return pd.DataFrame({"next": market["close"].shift(-1)})
"""
    on_bar = """
rising = ctx.value("market_next") > ctx.value("market_close")
if ctx.position is None and rising:
    return [ctx.buy(sizing=Sizing.percent_equity(100))]
if ctx.position is not None and not rising:
    return ctx.exit()
return []
"""
    source = _strategy(on_bar=on_bar, extra=extra, cls_extra='market_symbol = "SPY"')
    _assert_rejected(_check(source), S.LOOKAHEAD)
