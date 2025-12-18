"""Demo CLI wrapper that reuses the Edu CLI with predefined scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from coherent.edu.cli import main as edu_cli_main

SCENARIO_PATH = Path(__file__).resolve().parent / "scenarios" / "config.json"


def _load_scenarios() -> Dict[str, Dict[str, Any]]:
    if not SCENARIO_PATH.exists():
        return {}
    data = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios")
    return scenarios if isinstance(scenarios, dict) else {}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MathLang Demo CLI")
    parser.add_argument("--scenario", required=True, help="Scenario name defined in demo/scenarios/config.json")
    parser.add_argument(
        "--with-counterfactual",
        action="store_true",
        help="Apply counterfactual defined in the scenario (if available).",
    )
    args = parser.parse_args(argv)

    scenarios = _load_scenarios()
    scenario = scenarios.get(args.scenario)
    if scenario is None:
        parser.error(f"Unknown scenario '{args.scenario}'. Available: {', '.join(sorted(scenarios))}")

    cli_args: List[str] = ["--file", scenario["file"]]
    if scenario.get("mode") == "polynomial":
        cli_args.extend(["--mode", "polynomial"])
    if args.with_counterfactual and scenario.get("counterfactual"):
        cli_args.extend(["--counterfactual", json.dumps(scenario["counterfactual"])])
    return edu_cli_main(cli_args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
