# PyInstaller-Spezifikation für das Windows-Paket (Issue P-01).
#
# Bauen (aus dem Repo-Wurzelordner):
#     uv sync --group package
#     uv run pyinstaller packaging/trading_helfer.spec --noconfirm
#
# Ergebnis: dist/TradingHelfer/ mit
#   TradingHelfer.exe       – die App mit Fenster (ohne Konsole)
#   trading-helfer.exe      – dieselbe App mit Konsole für die Unterbefehle
#                             (update, scan, options-snapshot, import-csv, --selftest)
#   _internal/              – Python, Qt, Bibliotheken und Paketdaten
#
# Daten, Konfiguration, Logs und Plugins liegen im Benutzerverzeichnis
# (platformdirs, unter Windows %LOCALAPPDATA%\trading-helfer), nie im App-Ordner.
# Der Ordner kann also ersetzt werden, ohne Benutzerdaten zu verlieren.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH setzt PyInstaller

hiddenimports = (
    collect_submodules("trading_helfer")
    + collect_submodules("exchange_calendars")
)
datas = (
    # SQL-Migrationen und Indexmitglieder (CSV) aus dem Paket
    collect_data_files("trading_helfer", excludes=["**/__pycache__"])
    + collect_data_files("exchange_calendars")
    # Versionen aus den Paket-Metadaten (importlib.metadata)
    + copy_metadata("trading-helfer")
    + copy_metadata("yfinance")
    + copy_metadata("exchange-calendars")
)

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=["tkinter", "pytest", "IPython", "matplotlib"],
    # Eingebaute Strategien zusätzlich als .py beilegen: der Backtest speichert einen
    # Hash ihres Quelltexts (backtest.results.strategy_source_hash).
    module_collection_mode={"trading_helfer.strategies.builtin": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

gui_exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingHelfer",
    console=False,
    upx=False,
)
cli_exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="trading-helfer",
    console=True,
    upx=False,
)
coll = COLLECT(  # noqa: F821
    gui_exe,
    cli_exe,
    a.binaries,
    a.datas,
    upx=False,
    name="TradingHelfer",
)
