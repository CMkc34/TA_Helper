"""Tests für den Plugin-Loader und die Strategie-Registry (S-01)."""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from pathlib import Path

import pytest

from trading_helfer.core.config import Settings
from trading_helfer.strategies.base import Strategy
from trading_helfer.strategies.registry import (
    BUILTIN_SOURCE,
    LoadError,
    LoadErrorKind,
    LoadReport,
    StrategyNotFoundError,
    StrategyRegistry,
    check_fields,
)


def strategy_source(
    strategy_id: str = "plugin_one",
    *,
    cls: str = "PluginOne",
    version: str = "1.0.0",
    fields: str | None = None,
    extra: str = "",
) -> str:
    """Quelltext einer gültigen Plugin-Strategie."""
    if fields is None:
        fields = f"""
    id = "{strategy_id}"
    name = "Plugin {strategy_id}"
    version = "{version}"
    description = "Test-Plugin."
    timeframes = [Timeframe.D3, Timeframe.W1]
    params = {{"period": Param(20, 2, 200, 1, "Periode")}}
"""
    return (
        textwrap.dedent(
            """
            from trading_helfer.core.types import Timeframe
            from trading_helfer.strategies.base import BarContext, Order, Param, Strategy
            """
        )
        + extra
        + f"\n\nclass {cls}(Strategy):"
        + fields
        + "\n    def on_bar(self, ctx: BarContext) -> list[Order]:\n        return []\n"
    )


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    folder = tmp_path / "plugins"
    folder.mkdir()
    return folder


def plugin_registry(folder: Path) -> StrategyRegistry:
    return StrategyRegistry(folder, include_builtin=False)


def only_error(report: LoadReport) -> LoadError:
    assert len(report.errors) == 1, report.errors
    return report.errors[0]


# -- gültige Plugins ----------------------------------------------------------


def test_valid_plugin_is_loaded(plugins: Path) -> None:
    path = write(plugins, "one.py", strategy_source())
    registry = plugin_registry(plugins)

    report = registry.load()

    assert report.ok
    assert report.loaded == ("plugin_one",)
    assert report.summary() == "1 Strategien geladen."
    cls = registry.get("plugin_one")
    assert issubclass(cls, Strategy)
    assert cls.name == "Plugin plugin_one"
    info = registry.info("plugin_one")
    assert info.path == path
    assert info.source == str(path)
    assert not info.is_builtin
    assert info.name == "Plugin plugin_one"
    assert isinstance(registry.create("plugin_one"), cls)
    assert "plugin_one" in registry
    assert len(registry) == 1
    assert registry.ids() == ["plugin_one"]
    assert registry.list_all() == [info]


def test_plugin_with_dataclass_and_helper_base(plugins: Path) -> None:
    extra = textwrap.dedent(
        """
        from abc import abstractmethod
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Helper:
            value: int = 1

        class Base(Strategy):
            @abstractmethod
            def on_bar(self, ctx: BarContext) -> list[Order]: ...
        """
    )
    write(plugins, "two.py", strategy_source("plugin_two", extra=extra))

    report = plugin_registry(plugins).load()

    assert report.ok
    assert report.loaded == ("plugin_two",)


def test_files_load_in_name_order_and_skip_private(plugins: Path) -> None:
    write(plugins, "b.py", strategy_source("strat_b", cls="B"))
    write(plugins, "a.py", strategy_source("strat_a", cls="A"))
    write(plugins, "_helper.py", "raise RuntimeError('darf nicht geladen werden')")
    write(plugins, "notes.txt", "kein Python")

    report = plugin_registry(plugins).load()

    assert report.ok
    assert report.loaded == ("strat_a", "strat_b")


def test_one_file_with_two_strategies(plugins: Path) -> None:
    text = strategy_source("first", cls="First") + strategy_source("second", cls="Second")
    write(plugins, "multi.py", text)

    report = plugin_registry(plugins).load()

    assert set(report.loaded) == {"first", "second"}


def test_imported_strategy_is_not_registered_twice(plugins: Path) -> None:
    extra = "from trading_helfer.strategies.base import Strategy as _S\n"
    write(plugins, "one.py", strategy_source(extra=extra))

    assert plugin_registry(plugins).load().loaded == ("plugin_one",)


def test_missing_or_no_plugin_dir() -> None:
    assert plugin_registry(Path("/gibt/es/nicht")).load().ok
    registry = StrategyRegistry(None, include_builtin=False)
    assert registry.load().loaded == ()
    assert registry.plugins_dir is None


def test_plugin_path_is_file(tmp_path: Path) -> None:
    path = write(tmp_path, "datei.py", "")
    error = only_error(plugin_registry(path).load())
    assert error.kind is LoadErrorKind.IO
    assert "kein Ordner" in error.message


# -- Fehler werden gemeldet ----------------------------------------------------


def test_syntax_error_is_reported(plugins: Path) -> None:
    path = write(plugins, "kaputt.py", "class X(:\n    pass\n")
    write(plugins, "ok.py", strategy_source())

    report = plugin_registry(plugins).load()

    assert report.loaded == ("plugin_one",)
    error = only_error(report)
    assert error.kind is LoadErrorKind.SYNTAX
    assert error.source == str(path)
    assert error.line == 1
    assert "Syntaxfehler in Zeile 1" in error.message
    assert str(error).startswith(f"{path}:1: Syntaxfehler")
    assert report.summary() == "1 Strategien geladen, 1 Fehler."


def test_exception_during_import_is_reported(plugins: Path) -> None:
    write(plugins, "boom.py", "x = 1\nraise ValueError('kaputt')\n")
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.EXECUTION
    assert error.line == 2
    assert "ValueError: kaputt" in error.message
    assert "Zeile 2" in error.message


def test_missing_module_import_is_reported(plugins: Path) -> None:
    write(plugins, "imp.py", "import gibt_es_nicht_xyz\n")
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.EXECUTION
    assert "ModuleNotFoundError" in error.message


def test_system_exit_in_plugin_is_caught(plugins: Path) -> None:
    write(plugins, "exit.py", "raise SystemExit(3)\n")
    assert only_error(plugin_registry(plugins).load()).kind is LoadErrorKind.EXECUTION


def test_keyboard_interrupt_is_not_swallowed(plugins: Path) -> None:
    write(plugins, "ki.py", "raise KeyboardInterrupt\n")
    with pytest.raises(KeyboardInterrupt):
        plugin_registry(plugins).load()


def test_unreadable_file_is_reported(plugins: Path) -> None:
    (plugins / "bin.py").write_bytes(b"\xff\xfe\x00 kein utf-8")
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.IO
    assert "nicht lesbar" in error.message


def test_file_without_strategy(plugins: Path) -> None:
    write(plugins, "leer.py", "VALUE = 1\n")
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.NO_STRATEGY
    assert "Keine Strategie gefunden" in error.message


def test_strategy_without_on_bar(plugins: Path) -> None:
    text = textwrap.dedent(
        """
        from trading_helfer.strategies.base import Strategy

        class Halb(Strategy):
            id = "halb"
        """
    )
    write(plugins, "halb.py", text)
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.NO_STRATEGY
    assert error.strategy == "Halb"
    assert "on_bar fehlt" in error.message


def test_missing_required_fields(plugins: Path) -> None:
    fields = '\n    id = "ohne_felder"\n'
    write(plugins, "felder.py", strategy_source(fields=fields, cls="OhneFelder"))

    report = plugin_registry(plugins).load()

    assert report.loaded == ()
    error = only_error(report)
    assert error.kind is LoadErrorKind.MISSING_FIELD
    assert error.strategy == "OhneFelder"
    assert error.message == "Pflichtfeld fehlt: name, version, description, timeframes."


def test_missing_id_is_reported(plugins: Path) -> None:
    fields = textwrap.indent(
        textwrap.dedent(
            """
            name = "x"
            version = "1.0.0"
            description = "x"
            timeframes = [Timeframe.W1]
            """
        ),
        "    ",
    )
    write(plugins, "ohne_id.py", strategy_source(fields=fields))
    error = only_error(plugin_registry(plugins).load())
    assert error.kind is LoadErrorKind.MISSING_FIELD
    assert "id" in error.message


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ('id = "Mit Leerzeichen"', "Ungültige id"),
        ("id = 5", "Ungültige id"),
        ('name = "  "', "Feld name"),
        ("version = 1", "Feld version"),
        ("timeframes = []", "Feld timeframes"),
        ('timeframes = ["3D"]', "Feld timeframes"),
        ('params = {"x": 1}', "Feld params"),
    ],
)
def test_invalid_fields(plugins: Path, fields: str, expected: str) -> None:
    base = {
        "id": 'id = "ungueltig"',
        "name": 'name = "Name"',
        "version": 'version = "1.0.0"',
        "description": 'description = "Text"',
        "timeframes": "timeframes = [Timeframe.W1]",
    }
    key = fields.split(" ", 1)[0]
    base[key] = fields
    body = "".join(f"\n    {line}" for line in base.values()) + "\n"
    write(plugins, "ungueltig.py", strategy_source(fields=body))

    error = only_error(plugin_registry(plugins).load())

    assert error.kind is LoadErrorKind.INVALID_FIELD
    assert expected in error.message


def test_duplicate_id_is_reported(plugins: Path) -> None:
    first = write(plugins, "a.py", strategy_source("doppelt", cls="Erste"))
    second = write(plugins, "b.py", strategy_source("doppelt", cls="Zweite"))
    registry = plugin_registry(plugins)

    report = registry.load()

    assert report.loaded == ("doppelt",)
    assert registry.get("doppelt").__name__ == "Erste"
    error = only_error(report)
    assert error.kind is LoadErrorKind.DUPLICATE_ID
    assert error.source == str(second)
    assert error.strategy == "Zweite"
    assert "'doppelt' ist bereits vergeben" in error.message
    assert str(first) in error.message


def test_errors_are_logged(plugins: Path, caplog: pytest.LogCaptureFixture) -> None:
    write(plugins, "kaputt.py", "def (\n")
    with caplog.at_level(logging.WARNING, logger="trading_helfer.strategies.registry"):
        registry = plugin_registry(plugins)
        registry.load()
    assert "Syntaxfehler" in caplog.text
    assert registry.errors[0].kind is LoadErrorKind.SYNTAX


def test_unknown_id(plugins: Path) -> None:
    write(plugins, "one.py", strategy_source())
    registry = plugin_registry(plugins)
    registry.load()
    with pytest.raises(StrategyNotFoundError, match="Unbekannte Strategie: 'nein'"):
        registry.get("nein")
    assert str(StrategyNotFoundError()) == ""


# -- Reload ---------------------------------------------------------------------


def test_reload_after_file_change(plugins: Path) -> None:
    path = write(plugins, "one.py", strategy_source(version="1.0.0"))
    registry = plugin_registry(plugins)
    registry.load()
    old_cls = registry.get("plugin_one")
    assert old_cls.version == "1.0.0"

    write(plugins, "one.py", strategy_source(version="2.0.0"))
    # Gleiche Größe und mtime dürfen keinen veralteten Stand liefern.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    report = registry.reload()

    assert report.ok
    new_cls = registry.get("plugin_one")
    assert new_cls.version == "2.0.0"
    assert new_cls is not old_cls


def test_reload_picks_up_new_and_removed_files(plugins: Path) -> None:
    write(plugins, "a.py", strategy_source("strat_a", cls="A"))
    registry = plugin_registry(plugins)
    registry.load()

    write(plugins, "b.py", strategy_source("strat_b", cls="B"))
    (plugins / "a.py").unlink()

    assert registry.reload().loaded == ("strat_b",)
    assert "strat_a" not in registry


def test_reload_reports_new_error_and_recovers(plugins: Path) -> None:
    write(plugins, "one.py", strategy_source())
    registry = plugin_registry(plugins)
    assert registry.load().ok

    write(plugins, "one.py", "class (\n")
    report = registry.reload()
    assert report.loaded == ()
    assert only_error(report).kind is LoadErrorKind.SYNTAX

    write(plugins, "one.py", strategy_source())
    assert registry.reload().ok
    assert registry.errors == ()


def test_reload_cleans_up_modules(plugins: Path) -> None:
    write(plugins, "one.py", strategy_source())
    registry = plugin_registry(plugins)
    registry.load()
    module = registry.get("plugin_one").__module__
    assert module in sys.modules

    registry.reload()

    assert module not in sys.modules
    assert registry.get("plugin_one").__module__ in sys.modules


def test_changing_plugins_dir(tmp_path: Path) -> None:
    first = tmp_path / "eins"
    second = tmp_path / "zwei"
    first.mkdir()
    second.mkdir()
    write(first, "a.py", strategy_source("strat_a", cls="A"))
    write(second, "b.py", strategy_source("strat_b", cls="B"))
    registry = plugin_registry(first)
    registry.load()

    registry.plugins_dir = second

    assert registry.plugins_dir == second
    assert registry.reload().loaded == ("strat_b",)


def test_odd_file_names(plugins: Path) -> None:
    write(plugins, "1-mein plugin.py", strategy_source())
    assert plugin_registry(plugins).load().loaded == ("plugin_one",)


# -- eingebaute Strategien -----------------------------------------------------


@pytest.fixture
def builtin_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "pkgroot"
    pkg = root / "fake_builtin_s01"
    pkg.mkdir(parents=True)
    write(pkg, "__init__.py", "")
    monkeypatch.syspath_prepend(str(root))
    yield pkg
    for name in [n for n in sys.modules if n.startswith("fake_builtin_s01")]:
        del sys.modules[name]


def test_builtin_strategies_come_first(builtin_pkg: Path, plugins: Path) -> None:
    write(builtin_pkg, "sma.py", strategy_source("builtin_sma", cls="Sma"))
    write(builtin_pkg, "_intern.py", "raise RuntimeError('privat')\n")
    write(plugins, "dup.py", strategy_source("builtin_sma", cls="Kopie"))
    write(plugins, "own.py", strategy_source("own", cls="Own"))
    registry = StrategyRegistry(plugins, builtin_package="fake_builtin_s01")

    report = registry.load()

    assert report.loaded == ("builtin_sma", "own")
    info = registry.info("builtin_sma")
    assert info.is_builtin
    assert info.source == f"{BUILTIN_SOURCE}:sma"
    assert only_error(report).kind is LoadErrorKind.DUPLICATE_ID


def test_broken_builtin_is_reported(builtin_pkg: Path) -> None:
    write(builtin_pkg, "broken.py", "raise ImportError('fehlt')\n")
    error = only_error(StrategyRegistry(None, builtin_package="fake_builtin_s01").load())
    assert error.kind is LoadErrorKind.EXECUTION
    assert error.source == f"{BUILTIN_SOURCE}:broken"


def test_builtin_module_without_package_path(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = StrategyRegistry(None, builtin_package="textwrap")
    assert registry.load().ok


def test_real_builtin_package_loads_without_errors() -> None:
    assert StrategyRegistry(None).load().ok


# -- Einstellungen und Feldprüfung ---------------------------------------------


def test_settings_plugins_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.plugins_path == tmp_path / "plugins"
    custom = Settings(data_dir=tmp_path, plugins_dir=tmp_path / "eigene")
    assert custom.plugins_path == tmp_path / "eigene"


def test_check_fields_on_valid_class() -> None:
    from trading_helfer.core.types import Timeframe

    class Ok(Strategy):
        id = "ok"
        name = "Ok"
        version = "1.0.0"
        description = "Ok"
        timeframes = [Timeframe.W1]  # noqa: RUF012

        def on_bar(self, ctx):  # type: ignore[no-untyped-def]
            return []

    assert check_fields(Ok) == []
