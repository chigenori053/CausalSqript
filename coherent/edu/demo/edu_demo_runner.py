"""CLI helper for running MathLang Edu demo scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from coherent.edu.cli import main as edu_cli_main

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "edu_demo_config.yaml"


def _load_scenarios() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    scenarios = data.get("scenarios")
    return scenarios if isinstance(scenarios, dict) else {}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MathLang Edu Demo Runner")
    parser.add_argument("scenario", help="Scenario name defined in edu_demo_config.yaml")
    parser.add_argument(
        "--with-counterfactual",
        action="store_true",
        help="Apply counterfactual if the scenario defines it.",
    )
    args = parser.parse_args(argv)

    scenarios = _load_scenarios()
    scenario = scenarios.get(args.scenario)
    if scenario is None:
        parser.error(f"Unknown scenario '{args.scenario}'. Available: {', '.join(scenarios)}")

    file_path = scenario.get("file")
    if not file_path:
        parser.error(f"Scenario '{args.scenario}' does not define a file path.")

    cli_args = ["--file", file_path]
    counterfactual = scenario.get("counterfactual")
    if args.with_counterfactual and counterfactual:
        cli_args.extend(["--counterfactual", json.dumps(counterfactual)])
    return edu_cli_main(cli_args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
