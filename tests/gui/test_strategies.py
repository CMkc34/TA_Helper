"""Strategien-Tab (G-08): Liste, Details, Dialog „Strategie hinzufügen“, Backtest-Knopf."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.strategies import (
    COL_NAME,
    COL_ORIGIN,
    COL_STATUS,
    COL_VERSION,
    COLUMNS,
    AddStrategyDialog,
    StrategiesWidget,
)
from trading_helfer.services.app_service import AppService
from trading_helfer.services.strategy_service import SaveResult, StrategyService
from trading_helfer.strategies.validator import validate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOOD = FIXTURES / "good_strategies" / "sma_cross.py"
NETWORK = FIXTURES / "bad_strategies" / "network.py"


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    return tmp_path / "plugins"


@pytest.fixture
def strategy_service(plugins: Path) -> StrategyService:
    return StrategyService(plugins, validator=functools.partial(validate, isolate=False))


@pytest.fixture
def widget(qtbot: QtBot, strategy_service: StrategyService) -> StrategiesWidget:
    w = StrategiesWidget(strategy_service)
    qtbot.addWidget(w)
    return w


def table_text(widget: StrategiesWidget, row: int, column: int) -> str:
    item = widget.table.item(row, column)
    assert item is not None
    return item.text()


def row_index(widget: StrategiesWidget, key: str) -> int:
    return [r.key for r in widget.rows()].index(key)


def test_list_shows_builtin_strategies(widget: StrategiesWidget) -> None:
    assert [widget.table.horizontalHeaderItem(i).text() for i in range(5)] == list(COLUMNS)  # type: ignore[union-attr]
    index = row_index(widget, "trend_sma_cross")
    assert table_text(widget, index, COL_ORIGIN) == "eingebaut"
    assert table_text(widget, index, COL_STATUS) == "geladen"
    assert table_text(widget, index, COL_VERSION)
    assert widget.selected_key() is None
    assert widget.header_label.text() == texts.STRATEGIES_NONE_SELECTED
    assert not widget.backtest_button.isEnabled()
    assert "3 Strategien" in widget.summary_label.text()


def test_details_and_open_in_backtest(qtbot: QtBot, widget: StrategiesWidget) -> None:
    assert widget.select("trend_sma_cross")
    assert widget.selected_key() == "trend_sma_cross"
    assert "Version" in widget.header_label.text()
    assert "trend_sma_cross" in widget.meta_label.text()
    assert widget.description_label.text()
    assert widget.params_table.rowCount() > 0
    assert widget.file_label.text() == ""
    assert widget.backtest_button.isEnabled()
    with qtbot.waitSignal(widget.open_in_backtest) as blocker:
        widget.backtest_button.click()
    assert blocker.args == ["trend_sma_cross"]


def test_load_error_row_in_red_with_message(
    qtbot: QtBot, plugins: Path, strategy_service: StrategyService
) -> None:
    plugins.mkdir()
    (plugins / "broken.py").write_text("def kaputt(:\n", encoding="utf-8")
    widget = StrategiesWidget(strategy_service)
    qtbot.addWidget(widget)
    key = str(plugins / "broken.py")
    index = row_index(widget, key)
    assert table_text(widget, index, COL_NAME) == "broken.py"
    assert table_text(widget, index, COL_STATUS) == "Fehler"
    assert table_text(widget, index, COL_ORIGIN) == "Plugin"
    widget.select(key)
    assert widget.header_label.text() == "broken.py"
    assert "Syntaxfehler" in widget.error_view.toPlainText()
    assert not widget.backtest_button.isEnabled()
    with qtbot.assertNotEmitted(widget.open_in_backtest):
        widget._emit_open_in_backtest()


def test_reload_picks_up_new_plugin_and_keeps_selection(
    plugins: Path, widget: StrategiesWidget
) -> None:
    widget.select("mr_rsi2_trend")
    plugins.mkdir()
    (plugins / "copy.py").write_text(GOOD.read_text(encoding="utf-8"), encoding="utf-8")
    widget.reload_button.click()
    assert "example_sma_cross" in [r.key for r in widget.rows()]
    assert widget.selected_key() == "mr_rsi2_trend"


def test_strategy_without_params_shows_hint(
    qtbot: QtBot, plugins: Path, strategy_service: StrategyService
) -> None:
    plugins.mkdir()
    code = GOOD.read_text(encoding="utf-8")
    start = code.index("    params = {")
    end = code.index("    }\n", start) + len("    }\n")
    (plugins / "plain.py").write_text(code[:start] + code[end:], encoding="utf-8")
    widget = StrategiesWidget(strategy_service)
    qtbot.addWidget(widget)
    widget.select("example_sma_cross")
    assert widget.params_table.isHidden()
    assert not widget.no_params_label.isHidden()
    assert "plain.py" in widget.file_label.text()


# -- Dialog „Strategie hinzufügen“ -------------------------------------------------------


def test_add_valid_file(
    qtbot: QtBot, plugins: Path, widget: StrategiesWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(GOOD), "Python-Dateien (*.py)")
    )
    dialog = widget.open_add_dialog()
    qtbot.addWidget(dialog)
    assert not dialog.save_button.isEnabled()
    dialog.open_button.click()
    assert dialog.code() == GOOD.read_text(encoding="utf-8")
    assert str(GOOD) in dialog.file_label.text()

    dialog.validate_button.click()
    assert dialog.report is not None
    assert dialog.report.ok
    assert "alle Stufen bestanden" in dialog.status_label.text()
    assert "Lookahead" in dialog.report_view.toPlainText()
    assert dialog.save_button.isEnabled()

    with qtbot.waitSignal(dialog.saved) as blocker:
        dialog.save_button.click()
    result = blocker.args[0]
    assert isinstance(result, SaveResult)
    assert (plugins / "example_sma_cross.py").exists()
    assert dialog.result() == AddStrategyDialog.DialogCode.Accepted
    # Liste neu geladen und neue Strategie gewählt
    assert widget.selected_key() == "example_sma_cross"
    assert table_text(widget, row_index(widget, "example_sma_cross"), COL_ORIGIN) == "Plugin"


def test_add_invalid_code_shows_report_and_cannot_save(
    qtbot: QtBot, plugins: Path, strategy_service: StrategyService
) -> None:
    dialog = AddStrategyDialog(strategy_service)
    qtbot.addWidget(dialog)
    dialog.set_code(NETWORK.read_text(encoding="utf-8"), "network.py")
    dialog.validate()
    assert dialog.report is not None
    assert not dialog.report.ok
    assert "abgelehnt" in dialog.status_label.text()
    assert "socket" in dialog.report_view.toPlainText()
    assert not dialog.save_button.isEnabled()
    assert not dialog.save()
    assert not plugins.exists()


def test_empty_code_and_changed_code(qtbot: QtBot, strategy_service: StrategyService) -> None:
    dialog = AddStrategyDialog(strategy_service)
    qtbot.addWidget(dialog)
    dialog.validate()
    assert dialog.status_label.text() == texts.ADD_STRATEGY_EMPTY
    assert dialog.report is None

    dialog.set_code(GOOD.read_text(encoding="utf-8"))
    dialog.validate()
    assert dialog.save_button.isEnabled()
    dialog.editor.appendPlainText("# geändert")
    assert not dialog.save_button.isEnabled()
    assert dialog.status_label.text() == texts.ADD_STRATEGY_CHANGED
    assert not dialog.save()


def test_load_unreadable_file_warns(
    qtbot: QtBot, tmp_path: Path, strategy_service: StrategyService, dialogs: list[tuple[str, str]]
) -> None:
    dialog = AddStrategyDialog(strategy_service)
    qtbot.addWidget(dialog)
    assert not dialog.load_file(tmp_path / "fehlt.py")
    assert dialogs
    assert dialogs[0][0] == "warning"
    assert dialog.code() == ""


def test_cancel_file_dialog_keeps_code(
    qtbot: QtBot, strategy_service: StrategyService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: ("", ""))
    dialog = AddStrategyDialog(strategy_service)
    qtbot.addWidget(dialog)
    dialog.set_code("x = 1")
    dialog.choose_file()
    assert dialog.code() == "x = 1"


def _save_version(
    qtbot: QtBot, service: StrategyService, version: str
) -> tuple[AddStrategyDialog, str]:
    code = GOOD.read_text(encoding="utf-8").replace('"1.0.0"', f'"{version}"')
    dialog = AddStrategyDialog(service)
    qtbot.addWidget(dialog)
    dialog.set_code(code)
    dialog.validate()
    assert dialog.report is not None
    assert dialog.report.ok
    return dialog, code


@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No])
def test_same_id_asks_and_backs_up(
    qtbot: QtBot,
    plugins: Path,
    strategy_service: StrategyService,
    monkeypatch: pytest.MonkeyPatch,
    answer: QMessageBox.StandardButton,
) -> None:
    first, old = _save_version(qtbot, strategy_service, "1.0.0")
    assert first.save()

    questions: list[str] = []

    def question(_parent: object, _title: str, text: str, *_a: object) -> object:
        questions.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    second, new = _save_version(qtbot, strategy_service, "2.0.0")
    saved = second.save()

    assert len(questions) == 1
    assert "example_sma_cross_v1.0.0.py.bak" in questions[0]
    target = plugins / "example_sma_cross.py"
    backup = plugins / "example_sma_cross_v1.0.0.py.bak"
    if answer == QMessageBox.StandardButton.Yes:
        assert saved
        assert target.read_text(encoding="utf-8") == new
        assert backup.read_text(encoding="utf-8") == old
        assert "gesichert" in second.status_label.text()
        assert second.result_saved is not None
        assert second.result_saved.backups == (backup,)
    else:
        assert not saved
        assert target.read_text(encoding="utf-8") == old
        assert not backup.exists()


def test_builtin_id_is_refused(qtbot: QtBot, strategy_service: StrategyService) -> None:
    code = GOOD.read_text(encoding="utf-8").replace("example_sma_cross", "trend_sma_cross")
    dialog = AddStrategyDialog(strategy_service)
    qtbot.addWidget(dialog)
    dialog.set_code(code)
    dialog.validate()
    assert not dialog.save()
    assert "eingebaute" in dialog.status_label.text()


def test_replace_error_after_confirmation(
    qtbot: QtBot,
    plugins: Path,
    strategy_service: StrategyService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _old = _save_version(qtbot, strategy_service, "1.0.0")
    assert first.save()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)
    second, _new = _save_version(qtbot, strategy_service, "2.0.0")

    def fail(*_a: object, **_k: object) -> None:
        raise OSError("gesperrt")

    monkeypatch.setattr(Path, "replace", fail)
    assert not second.save()
    assert "gesperrt" in second.status_label.text()


# -- Hauptfenster ------------------------------------------------------------------------


def test_main_window_has_strategies_tab_and_opens_backtest(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]], plugins: Path
) -> None:
    strategy_service = StrategyService(plugins)
    win = MainWindow(service, strategy_service=strategy_service)
    qtbot.addWidget(win)
    assert win.tab_widget("strategies") is win.strategies
    assert win.strategies.rows() == []  # erst beim Anzeigen laden
    win.show()
    win.tabs.setCurrentWidget(win.strategies)
    qtbot.waitUntil(lambda: len(win.strategies.rows()) >= 3)

    win.strategies.select("trend_donchian_breakout")
    win.strategies.backtest_button.click()

    assert win.backtest_strategy == "trend_donchian_breakout"
    assert win.tabs.currentWidget() is win.tab_widget("backtest")
    assert "gewählt" in win.statusBar().currentMessage()


def test_main_window_calls_select_strategy_of_backtest_tab(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]]
) -> None:
    from PySide6.QtWidgets import QWidget

    class FakeBacktest(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.selected: list[str] = []

        def select_strategy(self, strategy_id: str) -> None:
            self.selected.append(strategy_id)

    win = MainWindow(service)
    qtbot.addWidget(win)
    assert win.strategy_service.plugins_dir == service.settings.plugins_path
    fake = FakeBacktest()
    win.set_view("backtest", fake)
    win.strategies.refresh()
    qtbot.waitUntil(lambda: len(win.strategies.rows()) >= 3)
    win.open_in_backtest("mr_rsi2_trend")
    assert fake.selected == ["mr_rsi2_trend"]


def test_tab_does_not_widen_main_window(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]]
) -> None:
    """Der Tab darf die Mindestbreite des Fensters nicht über die des Chart-Tabs heben."""
    win = MainWindow(service)
    qtbot.addWidget(win)
    win.strategies.refresh()
    qtbot.waitUntil(lambda: len(win.strategies.rows()) >= 3)
    win.strategies.select("trend_sma_cross")
    assert win.strategies.minimumSizeHint().width() <= win.chart.minimumSizeHint().width()
