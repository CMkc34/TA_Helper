"""Tests für den Strategie-Leitfaden (S-03): Vorlage und alle Code-Blöcke bestehen den Validator."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from trading_helfer.strategies.registry import StrategyRegistry
from trading_helfer.strategies.validator import ALLOWED_IMPORTS, FORBIDDEN_NAMES, validate

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "strategy_guide.md"
TEMPLATE = ROOT / "plugins" / "_template.py"

_FENCE = re.compile(r"^(?P<fence>`{3,}|~{3,})\s*(?P<lang>[\w+-]*)\s*$")


@dataclass(frozen=True)
class CodeBlock:
    """Ein Code-Block aus Markdown mit Sprache, Inhalt und Startzeile."""

    lang: str
    code: str
    line: int


def code_blocks(text: str) -> list[CodeBlock]:
    """Alle Code-Blöcke der obersten Ebene (äußere Zäune, verschachtelte zählen nicht)."""
    blocks: list[CodeBlock] = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i].rstrip("\n"))
        if match is None:
            i += 1
            continue
        fence, lang, start = match["fence"], match["lang"], i + 1
        body: list[str] = []
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                break
            body.append(lines[i])
            i += 1
        else:
            raise AssertionError(f"Code-Block ab Zeile {start} wird nicht geschlossen.")
        blocks.append(CodeBlock(lang, "".join(body), start))
        i += 1
    return blocks


def _guide_text() -> str:
    return GUIDE.read_text("utf-8")


def _python_blocks() -> list[CodeBlock]:
    return [b for b in code_blocks(_guide_text()) if b.lang == "python"]


def _prompt_block() -> CodeBlock:
    blocks = [b for b in code_blocks(_guide_text()) if b.lang == "text"]
    assert len(blocks) == 1, "Leitfaden braucht genau einen Prompt-Baustein (```text)."
    return blocks[0]


# ---------------------------------------------------------------------------
# Hilfsfunktion zum Extrahieren
# ---------------------------------------------------------------------------


def test_code_blocks_handles_nested_fences() -> None:
    text = "a\n````text\nx\n```python\ny = 1\n```\n````\n```python\nz = 2\n```\n"
    blocks = code_blocks(text)
    assert [(b.lang, b.line) for b in blocks] == [("text", 2), ("python", 8)]
    assert blocks[0].code == "x\n```python\ny = 1\n```\n"
    assert blocks[1].code == "z = 2\n"


def test_code_blocks_rejects_unclosed_fence() -> None:
    with pytest.raises(AssertionError, match="nicht geschlossen"):
        code_blocks("```python\nx = 1\n")


# ---------------------------------------------------------------------------
# Akzeptanzkriterien
# ---------------------------------------------------------------------------


def test_template_passes_validator() -> None:
    report = validate(TEMPLATE.read_text("utf-8"), filename=str(TEMPLATE), isolate=False)
    assert report.ok, report.format()


def test_guide_has_template_and_two_examples() -> None:
    blocks = _python_blocks()
    assert len(blocks) >= 3
    assert blocks[0].code == TEMPLATE.read_text("utf-8"), "Vorlage im Leitfaden veraltet."


@pytest.mark.parametrize(
    "block", _python_blocks(), ids=lambda b: f"zeile_{b.line}" if isinstance(b, CodeBlock) else ""
)
def test_every_python_block_passes_validator(block: CodeBlock) -> None:
    filename = f"{GUIDE.name}:{block.line}"
    report = validate(block.code, filename=filename, isolate=False)
    assert report.ok, report.format()


def test_all_code_blocks_are_python_or_prompt() -> None:
    langs = {b.lang for b in code_blocks(_guide_text())}
    assert langs <= {"python", "text"}, f"Unerwartete Code-Blöcke: {langs}"


def test_prompt_contains_template_and_passes_validator() -> None:
    inner = [b for b in code_blocks(_prompt_block().code) if b.lang == "python"]
    assert len(inner) == 1
    assert inner[0].code == TEMPLATE.read_text("utf-8"), "Vorlage im Prompt-Baustein veraltet."


def test_example_ids_are_unique_and_not_builtin() -> None:
    registry = StrategyRegistry(None)
    registry.load()
    builtin_ids = set(registry.ids())
    ids: list[str] = []
    for block in _python_blocks():
        report = validate(block.code, isolate=False)
        ids.extend(report.strategy_ids)
    assert len(ids) == len(set(ids)), f"Doppelte IDs: {ids}"
    assert not set(ids) & builtin_ids


def test_template_is_skipped_by_registry() -> None:
    registry = StrategyRegistry(TEMPLATE.parent)
    registry.load()
    assert "meine_strategie" not in registry.ids()
    assert not registry.errors


# ---------------------------------------------------------------------------
# Leitfaden passt zum Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ALLOWED_IMPORTS)
def test_guide_lists_allowed_imports(module: str) -> None:
    assert f"`{module}`" in _guide_text()
    assert module in _prompt_block().code


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NAMES))
def test_guide_lists_forbidden_names(name: str) -> None:
    assert f"`{name}`" in _guide_text()
    assert re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", _prompt_block().code)


def test_examples_only_use_allowed_imports() -> None:
    for block in _python_blocks():
        for node in ast.walk(ast.parse(block.code)):
            if isinstance(node, ast.ImportFrom):
                assert node.module is not None
                assert node.module in ALLOWED_IMPORTS or node.module.startswith(
                    "trading_helfer.indicators."
                )
