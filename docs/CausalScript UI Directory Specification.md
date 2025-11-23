# CausalScript UI Directory Specification (edu / pro)
Version: 0.1  
Status: Draft  
Target: Codex / CausalScript Dev Environment  

---

## 1. Purpose

This document defines the directory layout for **CausalScript UI modules**:

- `edu/` : Education-focused UI (students / teachers)
- `pro/` : Research-focused UI (R&D / researchers)

The goal is:

- To **share a single Core engine** (`core/`) across both variants.
- To keep **UI code, demos, and docs** separated by target user.
- To allow Codex-based tooling (CodexCLI, CodexAgent) to easily locate:
  - UI entry points
  - Demo notebooks
  - Scenarios / examples
  - Config / presets

---

## 2. Top-Level Project Layout

The project root MUST follow this structure:

```text
mathlang/
 ├── core/          # Shared symbolic/causal engine (NO UI)
 ├── edu/           # Education-oriented UI, DSL, demos (includes edu/examples/)
 ├── pro/           # Research-oriented UI, DSL, demos (includes pro/examples/)
 ├── docs/          # Specs, reports, demo walkthroughs
 ├── tests/         # Core-level tests
 ├── README.md
 └── Task_*.md      # Task / roadmap files
2.1 Directory Responsibilities
core/

Symbolic engine, KnowledgeRegistry, FuzzyJudge, CausalEngine, Counterfactual engine

NO UI logic

NO user-facing notebook (except internal dev tests if needed)

edu/

UI, notebooks, lessons for education

Simplified scenarios, progressive difficulty

Student/teacher-friendly UI defaults

pro/

UI, notebooks, tools for R&D / research

Causal graphs, counterfactual simulations, advanced visualizations

API usage examples, research workflows

3. edu/ Directory Specification
3.1 Layout
text
コードをコピーする
mathlang/
 └── edu/
      ├── notebooks/
      │    ├── edu_intro.ipynb
      │    ├── edu_arithmetic_steps.ipynb
      │    ├── edu_fractions.ipynb
      │    └── edu_causal_explain_basic.ipynb
      │
      ├── ui/
      │    ├── widgets.py
      │    ├── render_steps.py
      │    ├── layout.py
      │    └── __init__.py
      │
      ├── lessons/
      │    ├── lesson_01_arithmetic.yaml
      │    ├── lesson_02_fractions.yaml
      │    └── lesson_index.json
      │
      ├── demo/
      │    ├── edu_demo_runner.py
      │    ├── edu_demo_config.yaml
      │    └── edu_demo_scenarios.mlang
      │
      ├── config/
      │    ├── edu_ui_settings.yaml
      │    └── edu_theme.json
      │
      └── README.md
3.2 Module Roles
3.2.1 edu/notebooks/
Purpose: Jupyter Notebook based demo and teaching materials.

Naming convention:

edu_<topic>.ipynb

Examples:

edu_intro.ipynb — overview demo of CausalScript for education

edu_arithmetic_steps.ipynb — step-by-step arithmetic explanation

edu_fractions.ipynb — fraction simplification demos

edu_causal_explain_basic.ipynb — basic causal explanations in simple problems

Requirements:

Import from core.* (no logic duplication)

Use edu.ui.* helpers for rendering

Example (Python cell snippet):

python
コードをコピーする
from core.evaluator import Evaluator
from core.knowledge import KnowledgeRegistry
from edu.ui.render_steps import render_step_log

log = Evaluator.run_from_file("edu/examples/edu_arithmetic_demo.mlang")
render_step_log(log)
3.2.2 edu/ui/
Purpose: Rendering and layout utilities for edu notebooks / demos.

Files:

widgets.py

Jupyter widgets (dropdowns, text inputs, buttons).

e.g., select_example_widget(), run_demo_button().

render_steps.py

Step-by-step visualization:

pretty-print equations

show rule_id, fuzzy_score, causal.cause / causal.effect

layout.py

Standard layout utilities:

two-column layout (problem / step log)

theme / color settings

__init__.py

Re-exports for easy imports:

python
コードをコピーする
from .widgets import *
from .render_steps import *
from .layout import *
3.2.3 edu/lessons/
Purpose: Lesson definitions (curriculum) as data, NOT code.

Format:

lesson_XX_<name>.yaml for each lesson.

lesson_index.json listing available lessons and metadata.

Example: lesson_01_arithmetic.yaml

yaml
コードをコピーする
id: lesson_01_arithmetic
title: "Basic Arithmetic Steps"
level: "beginner"
examples:
  - file: "edu/examples/edu_arith_01.mlang"
    description: "Addition and subtraction with step explanations."
  - file: "edu/examples/edu_arith_02.mlang"
    description: "Multiplication and division."
3.2.4 edu/demo/
Purpose: CLI-friendly edu demo entry point.

edu_demo_runner.py

CLI entry to run typical educational scenarios.

Example usage:

bash
コードをコピーする
python -m edu.demo.edu_demo_runner --scenario basic_arithmetic
edu_demo_config.yaml

Global demo settings (default example paths, log options, etc.)

edu_demo_scenarios.mlang

DSL-based demo scripts for edu.

Should use simple, age-appropriate problems.

3.2.5 edu/config/
UI-level configuration:

edu_ui_settings.yaml — font size, language options, display flags.

edu_theme.json — color scheme for notebooks/UI.

3.2.6 edu/README.md
Should contain:

Purpose of edu/

How to run edu demo:

CLI (if applicable)

Notebook (edu_intro.ipynb)

Target users (students / teachers)

Link to lesson definitions in edu/lessons/

4. pro/ Directory Specification
4.1 Layout
text
コードをコピーする
mathlang/
 └── pro/
      ├── notebooks/
      │    ├── pro_intro_causal.ipynb
      │    ├── pro_counterfactual_walkthrough.ipynb
      │    ├── pro_fuzzy_analysis.ipynb
      │    └── pro_rule_debugger.ipynb
      │
      ├── api/
      │    ├── client.py
      │    ├── examples.py
      │    └── __init__.py
      │
      ├── tools/
      │    ├── graph_viewer.py
      │    ├── dot_export.py
      │    └── log_inspector.py
      │
      ├── demo/
      │    ├── pro_demo_runner.py
      │    ├── pro_causal_demo_config.yaml
      │    └── pro_counterfactual_demo.mlang
      │
      ├── config/
      │    ├── pro_settings.yaml
      │    └── pro_logging.yaml
      │
      └── README.md
4.2 Module Roles
4.2.1 pro/notebooks/
Purpose: R&D / research workflows and examples.

Naming convention:

pro_<topic>.ipynb

Examples:

pro_intro_causal.ipynb

Overview of CausalEngine usage.

pro_counterfactual_walkthrough.ipynb

Counterfactual scenario demos.

pro_fuzzy_analysis.ipynb

FuzzyJudge real-data analysis.

pro_rule_debugger.ipynb

Rule matching and KnowledgeRegistry inspection.

Example (Python cell snippet):

python
コードをコピーする
from core.evaluator import Evaluator
from core.causal import CausalEngine
from pro.tools.graph_viewer import show_causal_graph

log = Evaluator.run_from_file("edu/examples/counterfactual_demo.mlang")
show_causal_graph(log)
4.2.2 pro/api/
Purpose: Programmatic API layer for research usage.

client.py

High-level entry functions:

run_causal_analysis(source: str)

run_counterfactual(source: str, intervention: dict)

Should wrap calls to core.* and return standardized JSON.

examples.py

Example scripts showing inferencing from Python (no Notebook needed).

__init__.py

Re-export API functions.

4.2.3 pro/tools/
Purpose: Helper tools for visualization / log analysis.

graph_viewer.py

Functions to visualize causal/counterfactual graphs (e.g. via Graphviz or text-based view).

dot_export.py

Export internal causal graph to DOT format:

export_to_dot(log, path: str).

log_inspector.py

Utilities to inspect StepLog structures:

filter by rule_id

filter by fuzzy_score threshold

filter by causal.causal_score threshold

4.2.4 pro/demo/
Purpose: CLI-based research demos.

pro_demo_runner.py

CLI entry for research demos:

bash
コードをコピーする
python -m pro.demo.pro_demo_runner --mode causal
python -m pro.demo.pro_demo_runner --mode counterfactual
pro_causal_demo_config.yaml

Config for which example file / settings to use.

pro_counterfactual_demo.mlang

DSL script for representative counterfactual scenario.

4.2.5 pro/config/
pro_settings.yaml

Default settings for research use (e.g., logging verbosity, advanced flags).

pro_logging.yaml

Logging configuration (log level, file handlers).

4.2.6 pro/README.md
Should contain:

Purpose of pro/

How to run:

CLI demos (pro_demo_runner.py)

Notebooks (pro_intro_causal.ipynb)

Intended users (researchers, R&D engineers)

Basic API usage patterns (pro.api.client)

5. Shared Resources
5.1 edu/examples/ と pro/examples/
Edu 版・Pro 版それぞれで使用する .mlang デモを配置。共有が必要な場合は両ディレクトリに配置 or `docs/demo/` で案内。

例:

```text
edu/examples/
 ├── counterfactual_demo.mlang
 ├── polynomial_arithmetic.mlang
 └── pythagorean.mlang
pro/examples/
 └── polynomial_analysis.mlang
```
5.2 docs/
Doc examples relevant for edu and pro:

docs/demo/counterfactual_walkthrough.md

docs/demo/edu_walkthrough.md (optional)

docs/spec/CausalScript_Core_Spec.md

docs/spec/CausalScript_CausalEngine_Spec.md

6. Coding / Naming Guidelines
Python modules:

Use snake_case.py

Avoid mixing UI logic into core/.

edu/ and pro/ MUST import from core/, NOT the other way around.

Notebooks:

Prefix with edu_ or pro_.

Provide a top-level markdown cell explaining:

Purpose

Required imports

How to run in order.

Config files:

Use .yaml for structural config.

Use .json when interoperability with external tools is required.

7. CI / Testing Notes (Optional)
Core tests stay in tests/.

tests/test_core_*.py

edu/pro specific integration tests (optional):

tests/test_edu_demo.py

tests/test_pro_demo.py

CI can run:

pytest for core

smoke tests for edu_demo_runner.py and pro_demo_runner.py.

8. Versioning and Extensions
This spec is v0.1 and focuses on directory layout and high-level roles.

Future extensions may define:

Detailed API signatures in pro/api/client.py

Standard lesson formats for edu/lessons/

Common visualization contracts between core/* and edu/pro UIs.
