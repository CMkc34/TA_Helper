"""Tests für scripts/build_universe.py mit synthetischen Wikipedia-Seiten (offline)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from trading_helfer.data.universe import load_history

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia"
SYMS = [f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(100)]
AS_OF = dt.date(2026, 11, 21)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_universe", ROOT / "scripts" / "build_universe.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_universe"] = module
    spec.loader.exec_module(module)
    return module


bu = _load_script()


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """Alte CSV-Fassung: letzte Änderung 2025-12-01."""
    sp_current = sorted({*SYMS[1:98], "OLDA", "OLDB", "OLDC", "BRK.B"})
    (tmp_path / "sp500_current.csv").write_text(
        "symbol\n" + "".join(f"{s}\n" for s in sp_current), encoding="utf-8"
    )
    (tmp_path / "sp500_changes.csv").write_text(
        "date,added,removed,quelle\n2025-12-01,OLDA,GONE,alt\n", encoding="utf-8"
    )
    (tmp_path / "ndx100_current.csv").write_text(
        "symbol\n" + "".join(f"{s}\n" for s in [*SYMS[:99], "GONE"]), encoding="utf-8"
    )
    (tmp_path / "ndx100_changes.csv").write_text(
        "date,added,removed,quelle\n2024-12-23,GONE,OLD,alt\n", encoding="utf-8"
    )
    return tmp_path


def test_parse_tables_resolves_spans() -> None:
    tables = bu.parse_tables(_html("sp500_sample.html"))
    changes = tables[1]
    assert changes[0] == ["Effective Date", "Added", "Added", "Removed", "Removed", "Reason"]
    assert changes[1][0] == "Effective Date"
    assert changes[3][0] == "March 2, 2026"  # rowspan
    assert changes[2][5] == "Rebalancing"  # <sup> entfernt
    assert changes[3][5] == "Rebalancing"


def test_extract_current_and_changes() -> None:
    tables = bu.parse_tables(_html("sp500_sample.html"))
    current = bu.extract_current(tables)
    assert current == {*SYMS[:98], "NEWA", "BRK.B"}
    assert bu.extract_changes(tables) == [
        (dt.date(2026, 1, 5), {SYMS[0]}, {"OLDC"}),
        (dt.date(2026, 3, 2), {"NEWA", "NEWB"}, {"OLDA", "OLDB"}),
        (dt.date(2030, 12, 1), {"FUTR"}, set()),
    ]


def test_extract_without_tables() -> None:
    tables = bu.parse_tables(_html("ndx100_sample.html"))
    assert bu.extract_changes(tables) == []
    assert len(bu.extract_current(tables)) == 100
    with pytest.raises(ValueError, match="Bestandstabelle"):
        bu.extract_current([[["Symbol"], ["AAA"]]])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2025-09-22", dt.date(2025, 9, 22)),
        ("September 22, 2025", dt.date(2025, 9, 22)),
        ("Sep 22, 2025", dt.date(2025, 9, 22)),
        ("22 September 2025", dt.date(2025, 9, 22)),
        ("unbekannt", None),
    ],
)
def test_parse_date(text: str, expected: dt.date | None) -> None:
    assert bu.parse_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("brk-b", "BRK.B"), (" AAPL ", "AAPL"), ("", None), ("Apple Inc.", None), ("—", None)],
)
def test_normalize_ticker(text: str, expected: str | None) -> None:
    assert bu.normalize_ticker(text) == expected


def test_update_sp500_with_reconciliation(csv_dir: Path) -> None:
    result = bu.update_index("SP500", _html("sp500_sample.html"), csv_dir, AS_OF)
    assert [(r.date, r.added, r.removed) for r in result.new_rows] == [
        (dt.date(2026, 1, 5), SYMS[0], "OLDC"),
        (dt.date(2026, 3, 2), "NEWA", "OLDA"),
        (dt.date(2026, 3, 2), "NEWB", "OLDB"),
    ]
    # NEWB steht nicht mehr in der Bestandsliste → Abgleichzeile zum Stichtag
    assert [(r.date, r.added, r.removed, r.source) for r in result.reconciled] == [
        (AS_OF, "", "NEWB", bu.SOURCE_RECONCILE)
    ]
    bu.write_update(result, csv_dir)
    history = load_history("SP500", csv_dir)
    assert history.current == frozenset({*SYMS[:98], "NEWA", "BRK.B"})
    assert history.members_on("2026-01-04") == {*SYMS[1:98], "OLDA", "OLDB", "OLDC", "BRK.B"}
    assert "NEWB" in history.members_on("2026-03-02")
    lines = (csv_dir / "sp500_changes.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,added,removed,quelle"
    assert lines[1].startswith("2025-12-01")
    assert lines[-1] == f"{AS_OF},,NEWB,{bu.SOURCE_RECONCILE}"


def test_main_dry_run_and_write(csv_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pages = {"SP500": _html("sp500_sample.html"), "NDX100": _html("ndx100_sample.html")}
    fetched: list[str] = []

    def fetcher(url: str) -> str:
        fetched.append(url)
        code = next(c for c, u in bu.WIKIPEDIA_URLS.items() if u == url)
        return pages[code]

    before = (csv_dir / "ndx100_changes.csv").read_text(encoding="utf-8")
    args = ["--dir", str(csv_dir), "--as-of", "2026-11-21"]
    assert bu.main([*args, "--dry-run"], fetcher=fetcher) == 0
    assert (csv_dir / "ndx100_changes.csv").read_text(encoding="utf-8") == before
    assert len(fetched) == 2
    out = capsys.readouterr().out
    assert "NDX100: 100 aktuelle Mitglieder, 0 neue Änderungen, 1 Abgleichzeilen." in out

    assert bu.main([*args, "--index", "NDX100"], fetcher=fetcher) == 0
    history = load_history("NDX100", csv_dir)
    assert history.current == frozenset(SYMS)
    assert history.changes[-1].date == AS_OF
    assert history.changes[-1].added == {SYMS[99]}
    assert history.changes[-1].removed == {"GONE"}


def test_main_with_html_file(csv_dir: Path) -> None:
    html = FIXTURES / "sp500_sample.html"
    args = ["--dir", str(csv_dir), "--index", "SP500", "--html", str(html), "--as-of", "2026-11-21"]
    assert bu.main(args) == 0
    assert "NEWA" in load_history("SP500", csv_dir).current
    with pytest.raises(SystemExit):
        bu.main(["--dir", str(csv_dir), "--html", str(html)])
