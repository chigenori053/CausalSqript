# Coherent Pro Edition

## Overview
- Extends core Coherent engine with research-oriented CLI (`pro/cli.py`) and demos (`pro/examples/`).
- Default CLI entry: `python -m coherent.pro.cli -c "problem: (x + 1) * (x + 2)\nend: (x + 1) * (x + 2)"`
- Demo runner: `python -m coherent.pro.demo_runner counterfactual`

## Features
- Shared core (SymbolicEngine, Evaluator, CausalEngine, FuzzyJudge).
- Pro DSL wrapper (`ProParser`) for future extensions.
- Counterfactual CLI flag inherited from Edu edition.
- `coherent/pro/config/pro_settings.yaml` for logging + feature toggles.

## Structure
```
coherent/
  core/
  edu/
  pro/
    cli.py
    dsl/
    examples/
    config/
    demo_runner.py
tests/
docs/
```

## TODO
- Integrate Pro-specific notebooks (e.g., `pro/notebooks/pro_intro_causal.ipynb`).
- Expand `pro/examples/` with advanced scenarios.
