"""Shared CLI runner utilities for edu/pro/demo entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Any

from coherent.engine.errors import CoherentError
from coherent.engine.learning_logger import LearningLogger


def _load_source(file_path: Optional[str], inline_code: Optional[str]) -> str:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Script '{file_path}' not found")
        return path.read_text(encoding="utf-8")
    if inline_code:
        return inline_code
    raise CoherentError("Provide either --file or --code.")


def run_cli(
    argv: Optional[Sequence[str]],
    parser_cls: type,
    *,
    symbolic_runner: Callable[[Any, LearningLogger], Optional[Any]],
    polynomial_runner: Callable[[Any, LearningLogger], None],
    postprocess: Callable[[list[dict[str, Any]], Optional[Any], Optional[str]], None],
    scenario_loader: Callable[[str], dict[str, Any]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Coherent Program Runner")
    parser.add_argument("--file", help="Path to a .mlang file.")
    parser.add_argument("-c", "--code", help="Inline MathLang source.")
    parser.add_argument("--scenario", help="Scenario name defined in the CLI's scenario config.")
    parser.add_argument("--hello-world-test", action="store_true", help="Run built-in hello world test.")
    parser.add_argument("--test-core-arch", action="store_true", help="Run Core Architecture capabilities test (Notebook).")
    parser.add_argument(
        "--mode",
        choices=("symbolic", "polynomial", "causal"),
        default=None,
        help="Evaluation mode (symbolic or polynomial).",
    )
    parser.add_argument(
        "--counterfactual",
        help="Apply a simple counterfactual intervention (JSON string). Example: "
        '\'{"phase": "step", "index": 2, "expression": "8 * 4"}\'',
    )
    args = parser.parse_args(argv)

    if args.hello_world_test:
        print("== Coherent Execution (hello-world self-test) ==")
        print("Hello World")
        return 0

    if args.test_core_arch:
        print("== Running Core Architecture Test (Notebook) ==")
        import subprocess
        import os
        
        notebook_path = Path("tests/core_architecture_test.ipynb")
        if not notebook_path.exists():
            print(f"Error: Notebook not found at {notebook_path}", file=sys.stderr)
            return 1
            
        try:
            # Execute notebook using nbconvert
            cmd = [
                "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute", 
                "--inplace", 
                str(notebook_path)
            ]
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("\n✅ Core Architecture Test Passed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Core Architecture Test Failed with error code {e.returncode}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"\n❌ Error executing notebook: {e}", file=sys.stderr)
            return 1

    scenario_data: dict[str, Any] | None = None
    if args.scenario:
        if scenario_loader is None:
            print("Error: --scenario is not supported for this CLI", file=sys.stderr)
            return 1
        if args.file or args.code:
            print("Error: --scenario cannot be combined with --file or --code", file=sys.stderr)
            return 1
        try:
            scenario_data = scenario_loader(args.scenario)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    file_arg = args.file or (scenario_data.get("file") if scenario_data else None)
    code_arg = args.code or (scenario_data.get("code") if scenario_data else None)
    counterfactual_arg = args.counterfactual

    if scenario_data and not counterfactual_arg and scenario_data.get("counterfactual"):
        counterfactual_val = scenario_data["counterfactual"]
        if isinstance(counterfactual_val, str):
            counterfactual_arg = counterfactual_val
        else:
            counterfactual_arg = json.dumps(counterfactual_val)

    if (file_arg and code_arg) or (not file_arg and not code_arg):
        print("Error: Provide either --file or --code snippet", file=sys.stderr)
        return 1

    selected_mode = args.mode or (scenario_data.get("mode") if scenario_data else None) or "symbolic"
    effective_mode = "symbolic" if selected_mode == "causal" else selected_mode

    learning_logger = LearningLogger()
    knowledge_registry: Any | None = None
    try:
        source = _load_source(file_arg, code_arg)
        program = parser_cls(source).parse()
        if effective_mode == "polynomial":
            polynomial_runner(program, learning_logger)
        else:
            knowledge_registry = symbolic_runner(program, learning_logger)
        records = learning_logger.to_list()
        postprocess(records, knowledge_registry, counterfactual_arg)
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
    except CoherentError as exc:
        records = learning_logger.to_list()
        postprocess(records, knowledge_registry, counterfactual_arg)
        print(f"Error: {exc}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        print(f"Unexpected error: {exc}", file=sys.stderr)
    return 1
