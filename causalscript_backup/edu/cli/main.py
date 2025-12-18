"""Edu CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from causalscript.cli.base_runner import run_cli
from causalscript.cli.modes import run_symbolic_mode, run_polynomial_mode
from causalscript.cli.output import (
    print_records,
    print_causal_summary,
    print_counterfactual_summary,
    should_run_causal_analysis,
    rule_metadata_from_registry,
)
from causalscript.cli.scenarios import scenario_loader_from_file
from causalscript.edu.dsl import EduParser

SCENARIO_CONFIG = Path(__file__).resolve().parent / "scenarios" / "config.json"
_load_scenario = scenario_loader_from_file(SCENARIO_CONFIG)


def _postprocess(records, knowledge_registry, counterfactual_arg: Optional[str]) -> None:
    print_records(records)
    if should_run_causal_analysis(records):
        print_causal_summary(records, rule_info=rule_metadata_from_registry(knowledge_registry))
    if counterfactual_arg:
        print_counterfactual_summary(records, counterfactual_arg)


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(
        argv,
        EduParser,
        symbolic_runner=run_symbolic_mode,
        polynomial_runner=run_polynomial_mode,
        postprocess=_postprocess,
        scenario_loader=_load_scenario,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
