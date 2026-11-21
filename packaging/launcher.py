"""Startskript für PyInstaller: ruft den Einstiegspunkt ``trading_helfer.app:main`` auf."""

import sys

from trading_helfer.app import main

if __name__ == "__main__":
    sys.exit(main())
