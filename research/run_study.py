"""Studie für eine Kandidaten-Strategie ausführen (R-01).

Aufruf aus dem Repo-Ordner, z. B.::

    uv run python research/run_study.py --strategy trend_sma_cross --tf W1 --index SP500 \\
        --param fast=10 --param slow=40 --in-sample-only

Ablauf und Bewertungskriterien: ``research/README.md``.
"""

from __future__ import annotations

import sys

from trading_helfer.research.cli import main

if __name__ == "__main__":
    sys.exit(main())
