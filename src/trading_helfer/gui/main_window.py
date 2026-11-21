"""Hauptfenster: Menüs, Statusleiste, Watchlist links und Tabs für alle Ansichten."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Final

from PySide6.QtCore import QByteArray, Qt, QThreadPool
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from trading_helfer import __version__
from trading_helfer.gui import texts
from trading_helfer.gui.theme import apply_theme
from trading_helfer.gui.widgets.chart import ChartWidget
from trading_helfer.gui.widgets.data import DataWidget
from trading_helfer.gui.widgets.fundamentals import FundamentalsWidget
from trading_helfer.gui.widgets.strategies import StrategiesWidget
from trading_helfer.gui.widgets.watchlist import WatchlistWidget
from trading_helfer.gui.worker import Worker, WorkerError
from trading_helfer.services.app_service import AppService, ThemeName
from trading_helfer.services.chart_service import ChartService
from trading_helfer.services.data_service import DataService
from trading_helfer.services.fundamentals_service import FundamentalsService
from trading_helfer.services.indicator_service import TEMPLATES_FILE_NAME, IndicatorService
from trading_helfer.services.strategy_service import StrategyService

__all__ = ["TABS", "MainWindow"]

logger = logging.getLogger(__name__)

TABS: Final[tuple[tuple[str, str], ...]] = (
    ("chart", texts.TAB_CHART),
    ("fundamentals", texts.TAB_FUNDAMENTALS),
    ("backtest", texts.TAB_BACKTEST),
    ("optimization", texts.TAB_OPTIMIZATION),
    ("scanner", texts.TAB_SCANNER),
    ("strategies", texts.TAB_STRATEGIES),
    ("data", texts.TAB_DATA),
)
"""Schlüssel und Beschriftung der Tabs in fester Reihenfolge."""

DEFAULT_SIZE: Final = (1280, 800)


def _to_b64(data: QByteArray) -> str:
    return bytes(data.toBase64().data()).decode("ascii")


def _from_b64(text: str) -> QByteArray:
    return QByteArray.fromBase64(QByteArray(text.encode("ascii")))


class MainWindow(QMainWindow):
    """Hauptfenster, in das alle Ansichten als Tabs eingehängt werden.

    Args:
        service: Fassade zur Logik; liefert Einstellungen und speichert den GUI-Zustand.
        thread_pool: Pool für Hintergrundaufgaben; Standard ist der globale Pool.
        chart_service: Liefert die Kerzen für den Chart-Tab; Standard ist ein
            :class:`ChartService` ohne Kursquelle (leere Charts).
        fundamentals_service: Liefert die Kennzahlen für den Kennzahlen-Tab; Standard
            ist ein :class:`FundamentalsService` ohne Datenbank (leere Kennzahlen).
        strategy_service: Liefert die Strategien für den Strategien-Tab; Standard ist
            ein :class:`StrategyService` für den Plugin-Ordner aus den Einstellungen.
        data_service: Datenupdate, Qualitätsbericht und Quellen für den Daten-Tab;
            Standard ist ein :class:`DataService` mit den Einstellungen von ``service``,
            der Änderungen nur im Speicher hält.
    """

    def __init__(
        self,
        service: AppService,
        thread_pool: QThreadPool | None = None,
        parent: QWidget | None = None,
        chart_service: ChartService | None = None,
        fundamentals_service: FundamentalsService | None = None,
        strategy_service: StrategyService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.thread_pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._active_workers: set[Worker] = set()
        self.setWindowTitle(texts.APP_TITLE)
        self.setObjectName("MainWindow")

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self._tab_keys: list[str] = []
        for key, title in TABS:
            self._tab_keys.append(key)
            self.tabs.addTab(self._placeholder(title), title)
        self.setCentralWidget(self.tabs)

        self.chart_service = chart_service if chart_service is not None else ChartService()
        self.indicator_service = IndicatorService(service.state_path.parent / TEMPLATES_FILE_NAME)
        self.chart = ChartWidget(
            self.chart_service, runner=self.run_task, indicator_service=self.indicator_service
        )
        self.set_view("chart", self.chart)

        self.fundamentals_service = (
            fundamentals_service if fundamentals_service is not None else FundamentalsService()
        )
        self.fundamentals = FundamentalsWidget(self.fundamentals_service, runner=self.run_task)
        self.set_view("fundamentals", self.fundamentals)

        self.strategy_service = (
            strategy_service
            if strategy_service is not None
            else StrategyService(service.settings.plugins_path)
        )
        self.strategies = StrategiesWidget(
            self.strategy_service, runner=self.run_task, load_on_show=True
        )
        self.strategies.open_in_backtest.connect(self.open_in_backtest)
        self.set_view("strategies", self.strategies)
        self.backtest_strategy: str | None = None

        self.data_service = (
            data_service if data_service is not None else DataService(service.settings)
        )
        self.data = DataWidget(self.data_service, runner=self.run_task)
        self.data.update_finished.connect(self._on_data_changed)
        self.data.csv_imported.connect(self._on_data_changed)
        self.set_view("data", self.data)

        self.watchlist = WatchlistWidget(service)
        self.watchlist.symbol_selected.connect(self._on_symbol_selected)
        self.watchlist_dock = QDockWidget(texts.WATCHLIST_TITLE, self)
        self.watchlist_dock.setObjectName("WatchlistDock")
        self.watchlist_dock.setWidget(self.watchlist)
        self.watchlist_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.watchlist_dock)

        self._build_menus()
        self.statusBar().showMessage(texts.STATUS_READY)
        self.restore_layout()

    # -- Aufbau ----------------------------------------------------------------------

    @staticmethod
    def _placeholder(title: str) -> QWidget:
        label = QLabel(texts.PLACEHOLDER.format(name=title))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("placeholder", True)
        return label

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu(texts.MENU_FILE)
        self.quit_action = QAction(texts.ACTION_QUIT, self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)
        file_menu.addAction(self.quit_action)

        view_menu = menu_bar.addMenu(texts.MENU_VIEW)
        theme_menu = view_menu.addMenu(texts.MENU_THEME)
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        self.theme_actions: dict[ThemeName, QAction] = {}
        for theme, label in (
            ("light", texts.ACTION_THEME_LIGHT),
            ("dark", texts.ACTION_THEME_DARK),
        ):
            action = QAction(label, self, checkable=True)
            action.setData(theme)
            action.triggered.connect(lambda _checked=False, t=theme: self.set_theme(t))
            self.theme_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[theme] = action  # type: ignore[index]
        toggle = self.watchlist_dock.toggleViewAction()
        toggle.setText(texts.ACTION_TOGGLE_WATCHLIST)
        view_menu.addAction(toggle)

        help_menu = menu_bar.addMenu(texts.MENU_HELP)
        self.about_action = QAction(texts.ACTION_ABOUT, self)
        self.about_action.triggered.connect(self.show_about)
        help_menu.addAction(self.about_action)

    # -- Tabs ------------------------------------------------------------------------

    def tab_keys(self) -> list[str]:
        """Schlüssel aller Tabs in Anzeigereihenfolge."""
        return list(self._tab_keys)

    def tab_widget(self, key: str) -> QWidget:
        """Gibt das Widget des Tabs ``key`` zurück."""
        widget = self.tabs.widget(self._tab_keys.index(key))
        assert widget is not None
        return widget

    def set_view(self, key: str, widget: QWidget) -> None:
        """Ersetzt den Platzhalter des Tabs ``key`` durch ``widget``."""
        index = self._tab_keys.index(key)
        title = self.tabs.tabText(index)
        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, widget, title)
        if old is not None:
            old.deleteLater()

    # -- Theme und Layout ------------------------------------------------------------

    def set_theme(self, theme: ThemeName) -> None:
        """Wendet ``theme`` an, markiert den Menüeintrag und speichert die Wahl."""
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, theme)
        self.theme_actions[theme].setChecked(True)
        if self.service.theme != theme:
            self.service.set_theme(theme)

    def restore_layout(self) -> None:
        """Stellt Theme, Größe, Position und Dock-Anordnung aus dem gespeicherten Zustand her."""
        self.set_theme(self.service.theme)
        geometry, state = self.service.window_layout()
        if geometry is None or not self.restoreGeometry(_from_b64(geometry)):
            self.resize(*DEFAULT_SIZE)
        if state is not None:
            self.restoreState(_from_b64(state))

    def save_layout(self) -> None:
        """Speichert Größe, Position und Dock-Anordnung."""
        self.service.set_window_layout(_to_b64(self.saveGeometry()), _to_b64(self.saveState()))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt-Override)
        """Speichert das Layout vor dem Schließen."""
        self.save_layout()
        super().closeEvent(event)

    # -- Hintergrundaufgaben ---------------------------------------------------------

    def run_task(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        name: str = texts.TASK_DEFAULT_NAME,
        report_progress: bool = False,
        on_progress: Callable[[int, str], None] | None = None,
        on_error: Callable[[WorkerError], None] | None = None,
        **kwargs: Any,
    ) -> Worker:
        """Startet ``fn`` im Thread-Pool; Fortschritt landet in der Statusleiste.

        Fehler werden geloggt und als Dialog gezeigt; die App läuft weiter.

        Args:
            fn: Auszuführende Funktion.
            *args: Positionsargumente für ``fn``.
            on_result: Erhält das Ergebnis im GUI-Thread.
            name: Name der Aufgabe für Statusleiste und Fehlerdialog.
            report_progress: ``fn`` erhält einen ``progress``-Rückruf.
            on_progress: Erhält zusätzlich Prozent und Meldung im GUI-Thread.
            on_error: Erhält den Fehler im GUI-Thread (zusätzlich zum Dialog).
            **kwargs: Schlüsselwortargumente für ``fn``.
        """
        worker = Worker(fn, *args, report_progress=report_progress, name=name, **kwargs)
        worker.setAutoDelete(False)
        self._active_workers.add(worker)
        worker.signals.progress.connect(
            lambda percent, message: self.statusBar().showMessage(
                texts.STATUS_PROGRESS.format(name=name, percent=percent, message=message).strip()
            )
        )
        if on_progress is not None:
            worker.signals.progress.connect(on_progress)
        if on_result is not None:
            worker.signals.result.connect(on_result)
        if on_error is not None:
            worker.signals.error.connect(on_error)
        worker.signals.result.connect(
            lambda _r: self.statusBar().showMessage(texts.STATUS_DONE.format(name=name))
        )
        worker.signals.error.connect(lambda err: self.show_task_error(name, err))
        worker.signals.finished.connect(lambda: self._active_workers.discard(worker))
        self.statusBar().showMessage(texts.STATUS_RUNNING.format(name=name))
        self.thread_pool.start(worker)
        return worker

    def show_task_error(self, name: str, error: WorkerError) -> None:
        """Zeigt den Fehler einer Hintergrundaufgabe als Dialog (geloggt hat der Worker)."""
        self.statusBar().showMessage(texts.STATUS_FAILED.format(name=name))
        QMessageBox.critical(
            self, texts.ERROR_TITLE, texts.ERROR_TASK.format(name=name, message=error.message)
        )

    # -- Sonstiges -------------------------------------------------------------------

    def show_about(self) -> None:
        """Zeigt den Über-Dialog."""
        QMessageBox.about(self, texts.ABOUT_TITLE, texts.ABOUT_TEXT.format(version=__version__))

    def open_in_backtest(self, strategy_id: str) -> None:
        """Merkt sich ``strategy_id`` für den Backtest-Tab und wechselt dorthin.

        Hat der Backtest-Tab eine Methode ``select_strategy``, wird sie aufgerufen.
        """
        self.backtest_strategy = strategy_id
        backtest = self.tab_widget("backtest")
        select = getattr(backtest, "select_strategy", None)
        if callable(select):
            select(strategy_id)
        self.tabs.setCurrentWidget(backtest)
        name = self.strategy_service.registry.info(strategy_id).name
        self.statusBar().showMessage(texts.STRATEGIES_STATUS_BACKTEST.format(name=name))

    def _on_data_changed(self, _result: object) -> None:
        """Verwirft zwischengespeicherte Kurse und Kennzahlen nach Update oder Import."""
        self.chart_service.invalidate()
        self.fundamentals_service.invalidate()

    def _on_symbol_selected(self, symbol: str) -> None:
        self.statusBar().showMessage(texts.STATUS_SYMBOL.format(symbol=symbol))
        # Im Kennzahlen-Tab bleiben; sonst den Chart zeigen.
        if self.tabs.currentWidget() is not self.fundamentals:
            self.tabs.setCurrentWidget(self.chart)
        self.chart.load_symbol(symbol)
        self.fundamentals.load_symbol(symbol)
