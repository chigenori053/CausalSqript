# CausalScript CLI UI Specification (edu / pro / demo)
Version: 0.1
Status: Draft
Target: Codex / CausalScript Dev Environment

## 1. Purpose
This document defines the CLI directory layout and execution rules for CausalScript's edu, pro, and demo environments. It ensures that each variant can be executed and tested using CLI-based workflows compatible with Codex.

## 2. Directory Structure
mathlang/
 ├── core/
 ├── edu/
 │    ├── cli/
 │    │     ├── main.py
 │    │     ├── commands.py
 │    │     └── scenarios/
 │    ├── ui/
 │    ├── notebooks/
 │    └── ...
 ├── pro/
 │    ├── cli/
 │    │     ├── main.py
 │    │     ├── commands.py
 │    │     └── scenarios/
 │    ├── tools/
 │    ├── notebooks/
 │    └── ...
 ├── demo/
 │    ├── demo_cli.py
 │    ├── demo_config.yaml
 │    └── scenarios/
 └── tests/

## 3. CLI Execution Formats
Edu:
  python -m edu.cli.main --scenario <scenario>

Pro:
  python -m pro.cli.main --mode <mode> --scenario <scenario>

Demo:
  python -m demo.demo_cli --scenario <scenario>

### 3.1 Scenario Configuration
- Each CLI reads scenario metadata from `scenarios/config.json` directly under its CLI package:
  - Edu: `edu/cli/scenarios/config.json`
  - Pro: `pro/cli/scenarios/config.json`
  - Demo: `demo/scenarios/config.json`
- Every scenario entry must declare either `file` or `code`. Optional fields:
  - `mode`: `symbolic`, `polynomial`, or `causal` (alias of symbolic for consistency with specs).
  - `counterfactual`: JSON dict/string that is forwarded to `--counterfactual`.
- CI / documentation examples must reference scenarios defined in those files so the commands above always succeed without extra flags.

## 4. Output Format
Each CLI must print step-by-step logs including:
- Before/After
- Rule ID
- Fuzzy Score
- Causal Explanation

## 5. Testing Requirements
- pytest must include:
  - test_edu_cli.py
  - test_pro_cli.py
  - test_demo_cli.py

## 6. CI Integration
CI must run:
  python -m edu.cli.main --scenario arithmetic
  python -m pro.cli.main --mode causal --scenario basic
  python -m demo.demo_cli --scenario minimal
