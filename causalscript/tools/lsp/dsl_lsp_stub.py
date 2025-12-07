"""Minimal stub for DSL LSP / completion proof-of-concept."""

from __future__ import annotations

import argparse
import json
from typing import List

PREDEFINED_KEYWORDS = [
    "meta",
    "config",
    "mode",
    "problem",
    "prepare",
    "step",
    "end",
    "counterfactual",
]


def suggest(prefix: str) -> List[str]:
    return [kw for kw in PREDEFINED_KEYWORDS if kw.startswith(prefix.lower())]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MathLang DSL completion stub")
    parser.add_argument("prefix", help="Current word prefix")
    args = parser.parse_args(argv)
    print(json.dumps({"suggestions": suggest(args.prefix)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
