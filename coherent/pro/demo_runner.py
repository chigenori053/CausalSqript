"""CLI runner for MathLang Pro demos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from coherent.pro.cli import main as pro_main


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MathLang Pro Demo Runner")
    parser.add_argument("scenario", choices=["counterfactual"], help="Demo scenario name")
    args = parser.parse_args(argv)

    if args.scenario == "counterfactual":
        demo_path = Path("pro/examples/polynomial_analysis.mlang")
        if demo_path.exists():
            print("Running Pro counterfactual demo...")
            pro_main(["--file", str(demo_path)])
        else:
            print("Demo file not found:", demo_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
