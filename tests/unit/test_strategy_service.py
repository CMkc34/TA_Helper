"""Tests für den StrategyService (G-08): Liste, Details, Prüfen und Speichern von Plugins."""

from __future__ import annotations

import functools
import textwrap
from pathlib import Path

import pytest

from trading_helfer.services.strategy_service import (
    Origin,
    StrategyExistsError,
    StrategySaveError,
    StrategyService,
    StrategyStatus,
    backup_path,
    format_param_value,
)
from trading_helfer.strategies.validator import (
    StageResult,
    StageStatus,
    ValidationReport,
    ValidationStage,
    validate,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOOD = FIXTURES / "good_strategies" / "sma_cross.py"
NETWORK = FIXTURES / "bad_strategies" / "network.py"


def plugin_source(
    strategy_id: str = "my_plugin",
    *,
    cls: str = "MyPlugin",
    version: str = "1.0.0",
    extra_params: str = "",
) -> str:
    """Quelltext einer einfachen gültigen Plugin-Strategie."""
    return textwrap.dedent(
        f'''
        """Test-Plugin."""

        from trading_helfer.core.types import Timeframe
        from trading_helfer.strategies.base import BarContext, Order, Param, Strategy


        class {cls}(Strategy):
            id = "{strategy_id}"
            name = "Plugin {strategy_id}"
            version = "{version}"
            description = "Beschreibung von {strategy_id}."
            timeframes = [Timeframe.D3, Timeframe.W1]
            params = {{
                "period": Param(20, 5, 100, 5, "Periode"),
                "factor": Param(1.5, 0.5, 3.0, 0.25, "Faktor"),
                "use_filter": Param(True, description="Filter an"),
                "mode": Param("a", choices=("a", "b"), description="Modus"),{extra_params}
            }}

            def on_bar(self, ctx: BarContext) -> list[Order]:
                return []
        '''
    )


def passed(*ids: str) -> ValidationReport:
    """Bestandener Bericht ohne echten Validator-Lauf."""
    stages = tuple(StageResult(stage, StageStatus.PASSED) for stage in ValidationStage)
    return ValidationReport("<test>", stages, ids)


def failed() -> ValidationReport:
    stages = (StageResult(ValidationStage.SYNTAX, StageStatus.FAILED, "kaputt", line=1),)
    return ValidationReport("<test>", stages)


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    return tmp_path / "plugins"


@pytest.fixture
def service(plugins: Path) -> StrategyService:
    return StrategyService(plugins, validator=functools.partial(validate, isolate=False))


def test_list_contains_builtin_strategies_when_no_plugins_dir(service: StrategyService) -> None:
    rows = service.list_strategies()
    keys = {r.key for r in rows}
    assert {"trend_sma_cross", "trend_donchian_breakout", "mr_rsi2_trend"} <= keys
    sma = next(r for r in rows if r.key == "trend_sma_cross")
    assert sma.origin is Origin.BUILTIN
    assert sma.status is StrategyStatus.LOADED
    assert sma.ok
    assert sma.version
    assert "1W" in sma.timeframes


def test_list_shows_plugins_and_load_errors(plugins: Path, service: StrategyService) -> None:
    plugins.mkdir()
    (plugins / "good.py").write_text(plugin_source("good_one"), encoding="utf-8")
    (plugins / "broken.py").write_text("def kaputt(:\n", encoding="utf-8")
    service.reload()
    rows = {r.key: r for r in service.list_strategies()}
    good = rows["good_one"]
    assert good.origin is Origin.PLUGIN
    assert good.ok
    assert good.timeframes == "3D, 1W"
    broken = rows[str(plugins / "broken.py")]
    assert broken.status is StrategyStatus.ERROR
    assert not broken.ok
    assert broken.name == "broken.py"
    assert "Syntaxfehler" in broken.message


def test_details_of_plugin(plugins: Path, service: StrategyService) -> None:
    plugins.mkdir()
    (plugins / "good.py").write_text(plugin_source("good_one"), encoding="utf-8")
    details = service.details("good_one")
    assert details.description == "Beschreibung von good_one."
    assert details.path == plugins / "good.py"
    assert details.class_name == "MyPlugin"
    params = {p.name: p for p in details.params}
    assert list(params) == ["period", "factor", "use_filter", "mode"]
    assert (params["period"].default, params["period"].minimum, params["period"].maximum) == (
        "20",
        "5",
        "100",
    )
    assert params["factor"].step == "0.25"
    assert params["use_filter"].default == "ja"
    assert params["use_filter"].minimum == ""
    assert params["mode"].default == "a (Auswahl: a, b)"
    assert params["period"].description == "Periode"


def test_details_of_builtin_and_error_and_unknown(plugins: Path, service: StrategyService) -> None:
    plugins.mkdir()
    (plugins / "broken.py").write_text("raise RuntimeError('x')\n", encoding="utf-8")
    builtin = service.details("trend_sma_cross")
    assert builtin.path is None
    assert builtin.description
    error = service.details(str(plugins / "broken.py"))
    assert not error.row.ok
    assert error.params == ()
    with pytest.raises(KeyError):
        service.details("gibt_es_nicht")


def test_format_param_value() -> None:
    assert format_param_value(None) == ""
    assert format_param_value(False) == "nein"
    assert format_param_value(2.0) == "2"
    assert format_param_value(0.5) == "0.5"


def test_read_file(tmp_path: Path, service: StrategyService) -> None:
    path = tmp_path / "s.py"
    path.write_text("x = 1\n", encoding="utf-8")
    assert service.read_file(path) == "x = 1\n"
    with pytest.raises(StrategySaveError, match="kann nicht gelesen"):
        service.read_file(tmp_path / "fehlt.py")
    binary = tmp_path / "b.py"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(StrategySaveError, match="UTF-8"):
        service.read_file(binary)


def test_read_file_too_large(
    tmp_path: Path, service: StrategyService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_helfer.services.strategy_service.MAX_SOURCE_BYTES", 3)
    path = tmp_path / "s.py"
    path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(StrategySaveError, match="größer"):
        service.read_file(path)


# -- Hinzufügen mit echtem Validator -------------------------------------------------


def test_add_valid_code_is_saved_and_loaded(plugins: Path, service: StrategyService) -> None:
    code = GOOD.read_text(encoding="utf-8")
    report = service.validate(code, "sma_cross.py")
    assert report.ok, report.format()
    assert report.strategy_ids == ("example_sma_cross",)

    result = service.save_plugin(code, report)

    assert result.path == plugins / "example_sma_cross.py"
    assert result.path.read_text(encoding="utf-8") == code
    assert result.backups == ()
    assert "example_sma_cross" in result.load_report.loaded
    row = next(r for r in service.list_strategies() if r.key == "example_sma_cross")
    assert row.origin is Origin.PLUGIN
    assert row.ok


def test_add_invalid_code_is_rejected(plugins: Path, service: StrategyService) -> None:
    code = NETWORK.read_text(encoding="utf-8")
    report = service.validate(code, "network.py")
    assert not report.ok
    assert report.failed_stage is ValidationStage.SECURITY
    with pytest.raises(StrategySaveError, match="Nur geprüfte"):
        service.save_plugin(code, report)
    assert not plugins.exists() or not list(plugins.glob("*.py"))


def test_save_rejects_failed_report(service: StrategyService) -> None:
    with pytest.raises(StrategySaveError):
        service.save_plugin("x", failed())
    with pytest.raises(StrategySaveError):
        service.save_plugin("x", passed())  # keine IDs


def test_save_rejects_builtin_id(plugins: Path, service: StrategyService) -> None:
    with pytest.raises(StrategySaveError, match="eingebaute"):
        service.save_plugin(plugin_source("trend_sma_cross"), passed("trend_sma_cross"))
    assert not plugins.exists()


# -- Gleiche ID: Nachfrage und Sicherung ---------------------------------------------


def test_same_id_asks_then_backs_up_old_version(plugins: Path, service: StrategyService) -> None:
    old = plugin_source("my_plugin", version="1.0.0")
    first = service.save_plugin(old, passed("my_plugin"))
    assert first.path == plugins / "my_plugin.py"

    new = plugin_source("my_plugin", version="1.1.0")
    report = passed("my_plugin")
    assert [i.id for i in service.conflicts(report)] == ["my_plugin"]
    with pytest.raises(StrategyExistsError) as excinfo:
        service.save_plugin(new, report)
    assert "my_plugin" in str(excinfo.value)
    assert "1.0.0" in str(excinfo.value)
    assert [i.id for i in excinfo.value.existing] == ["my_plugin"]
    # Ohne Bestätigung bleibt alles unverändert.
    assert (plugins / "my_plugin.py").read_text(encoding="utf-8") == old

    result = service.save_plugin(new, report, replace=True)

    backup = plugins / "my_plugin_v1.0.0.py.bak"
    assert result.backups == (backup,)
    assert backup.read_text(encoding="utf-8") == old
    assert (plugins / "my_plugin.py").read_text(encoding="utf-8") == new
    assert service.registry.get("my_plugin").version == "1.1.0"
    assert not service.registry.errors  # .bak wird nicht geladen


def test_replace_plugin_from_other_file_name(plugins: Path, service: StrategyService) -> None:
    plugins.mkdir()
    old = plugin_source("my_plugin", version="0.9.0")
    (plugins / "alt_name.py").write_text(old, encoding="utf-8")
    service.reload()
    result = service.save_plugin(plugin_source("my_plugin"), passed("my_plugin"), replace=True)
    assert not (plugins / "alt_name.py").exists()
    assert result.backups == (plugins / "my_plugin_v0.9.0.py.bak",)
    assert result.path == plugins / "my_plugin.py"
    assert [i.id for i in service.registry.list_all() if not i.is_builtin] == ["my_plugin"]


def test_second_backup_of_same_version_gets_counter(
    plugins: Path, service: StrategyService
) -> None:
    code = plugin_source("my_plugin")
    service.save_plugin(code, passed("my_plugin"))
    service.save_plugin(code, passed("my_plugin"), replace=True)
    result = service.save_plugin(code, passed("my_plugin"), replace=True)
    assert result.backups == (plugins / "my_plugin_v1.0.0_2.py.bak",)
    assert backup_path(plugins, "my_plugin", "1.0.0").name == "my_plugin_v1.0.0_3.py.bak"


def test_unloadable_file_with_target_name_is_backed_up(
    plugins: Path, service: StrategyService
) -> None:
    plugins.mkdir()
    (plugins / "my_plugin.py").write_text("def kaputt(:\n", encoding="utf-8")
    result = service.save_plugin(plugin_source("my_plugin"), passed("my_plugin"))
    assert result.backups == (plugins / "my_plugin_vunbekannt.py.bak",)
    assert service.registry.get("my_plugin").version == "1.0.0"


def test_refuses_to_replace_file_with_other_strategies(
    plugins: Path, service: StrategyService
) -> None:
    plugins.mkdir()
    both = (
        plugin_source("my_plugin")
        + plugin_source("other_one", cls="Other").split('"""Test-Plugin."""', 1)[1]
    )
    (plugins / "both.py").write_text(both, encoding="utf-8")
    service.reload()
    assert {"my_plugin", "other_one"} <= set(service.registry.ids())
    with pytest.raises(StrategySaveError, match="weitere Strategien"):
        service.save_plugin(plugin_source("my_plugin"), passed("my_plugin"), replace=True)
    assert (plugins / "both.py").exists()


def test_write_error_is_reported(
    plugins: Path, service: StrategyService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> int:
        raise OSError("Datenträger voll")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(StrategySaveError, match="Datenträger voll"):
        service.save_plugin(plugin_source("my_plugin"), passed("my_plugin"))


def test_default_registry_uses_plugins_dir(plugins: Path) -> None:
    service = StrategyService(plugins)
    assert service.plugins_dir == plugins
    assert service.registry.plugins_dir == plugins
