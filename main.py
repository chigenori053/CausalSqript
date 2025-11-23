"""Backward-compatible entry point that delegates to the Edu CLI."""

from __future__ import annotations

import sys

from edu.cli import main as edu_main


def main(argv: list[str] | None = None) -> int:
    return edu_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
