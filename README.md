# COHERENT

> **Intelligence in Phase.**
> Where Waves Become Logic.

COHERENTは、光学干渉演算（Optical Interference Calculation）を基盤とした次世代の論理推論システムです。

## Why Coherent?
- **Process-first**: emphasizes intermediate working steps over final answers.
- **AI-assisted**: pairs human reasoning with SymbolicAI for explanation and transformation of expressions.
- **Reproducible**: identical inputs always re-create the same execution trace.
- **Lightweight**: Python 3.12 stack with modular components for education and research.

## Architecture Snapshot
| Layer | Key Modules | Purpose |
|-------|-------------|---------|
| DSL Core | `coherent/core/parser.py`, `coherent/core/ast_nodes.py` | Parse Coherent syntax into AST structures. |
| Execution | `coherent/core/evaluator.py`, `coherent/core/core_runtime.py` | Orchestrate reasoning steps, validation, and extensions. |
| SymbolicAI | `coherent/core/symbolic_engine.py` | SymPy-based expression manipulation and equivalency checking. |
| Knowledge | `coherent/core/knowledge_registry.py` | Rule-based prediction, semantic linking, and concept mapping. |
| Validation | `coherent/core/validation_engine.py` | Integrated pipeline: Symbolic -> Fuzzy -> Decision. |
| Decision | `coherent/core/decision_theory.py` | Strategic judgment (Accept/Review/Reject) based on utility/regret. |
| Heuristics | `coherent/core/heuristics.py` | Detect common misconceptions (e.g., Freshman's Dream). |
| Causal | `coherent/core/causal/` | Diagnosis, graph construction, and fix suggestion. |
| Interfaces | JupyterLab / Streamlit | Provide interactive teaching and demo surfaces. |

Reference structure and full requirements live in `docs/`.

## DSL Glimpse
```text
# Coherent DSL v2.5 example
meta:
    id: lesson_01
    topic: arithmetic
config:
    causal: true
    fuzzy-threshold: 0.6
problem: (3 + 5) * 4
prepare:
    - a = 3 + 5
    - b = a * 4
step:
    before: (3 + 5) * 4
    after: 8 * 4
    note: simplify addition
step:
    before: 8 * 4
    after: 32
end: 32
counterfactual:
    assume:
        a: 10
    expect: a * 3 + 2
```
Expected output:
```
Config: {'causal': True, 'fuzzy-threshold': 0.6}
Problem: (3 + 5) * 4
Prepare: a = 3 + 5
Prepare: b = a * 4
Step (unnamed): 8 * 4
Step (unnamed): 32
End: 32
```

## CLI Usage
- Run a `.mlang` file: `python main.py --file coherent/edu/examples/pythagorean.mlang`
- Run an inline snippet: `python main.py -c "problem: 1 + 1\nend: 2"`
- Switch to the polynomial evaluator: `python main.py --mode polynomial --file coherent/edu/examples/polynomial_arithmetic.mlang`
- Run the built-in Hello World self-test: `python main.py --hello-world-test`
- Simulate a counterfactual: `python main.py --file coherent/edu/examples/counterfactual_demo.mlang --counterfactual '{"phase": "step", "index": 2, "expression": "8 * 4"}'`
- Run the Edu demo runner: `python -m coherent.edu.demo.edu_demo_runner basic_arithmetic`
- Run the Pro CLI: `python -m coherent.pro.cli -c "problem: (x + 1) * (x + 2)\nend: (x + 1) * (x + 2)"`
- Run the Pro demo runner: `python -m coherent.pro.demo_runner counterfactual`
- Run the Edu CLI by scenario name (see `coherent/edu/cli/scenarios/config.json`): `python -m coherent.edu.cli.main --scenario arithmetic`
- Run the Pro CLI scenario: `python -m coherent.pro.cli.main --mode causal --scenario basic` (`coherent/pro/cli/scenarios/config.json`)
- Run the Demo CLI minimal scenario: `python -m demo.demo_cli --scenario minimal` (`demo/scenarios/config.json`)

The CLI prints rendered text for every `problem`, `step`, `explain`, and `end` clause. Symbolic mode verifies steps with SymPy and the knowledge registry, while polynomial mode expands expressions before comparing them.  
エラーが発生した場合は、収集した LearningLogger レコードを元に因果推論エンジンが解析され、`== Causal Analysis ==` セクションとして推定原因と修正候補ステップが追加表示されます。

### Scenario Configuration Files
Each CLI keeps scenario definitions under its `scenarios/config.json`. These JSON files list `file`, optional `mode` (`symbolic` / `polynomial` / `causal`), and `counterfactual` payloads that `--scenario` picks up automatically:

- Edu: `coherent/edu/cli/scenarios/config.json`
- Pro: `coherent/pro/cli/scenarios/config.json`
- Demo: `demo/scenarios/config.json`

You can add new entries to those files to expose additional `.mlang` programs without changing the CLI flags used by CI.

## Pro Edition
プロフェッショナル向け CLI / デモについては `README_PRO.md` を参照。`python -m coherent.pro.cli ...` で直接呼び出せます。

## Learning Logs & Notebook Demo
Coherent now ships with a lightweight `LearningLogger` that records `problem → step → end` events (including rule IDs supplied by the knowledge base) as JSON. You can pass the logger into an evaluator:

```python
from pathlib import Path

from coherent.core.learning_logger import LearningLogger
from coherent.core.parser import Parser
from coherent.core.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.knowledge_registry import KnowledgeRegistry

source = """problem: (2 + 3) * 4
step: 5 * 4
end: 20"""

logger = LearningLogger()
program = Parser(source).parse()
symbolic_engine = SymbolicEngine()
registry = KnowledgeRegistry(Path("coherent/core/knowledge"), symbolic_engine)
engine = SymbolicEvaluationEngine(symbolic_engine, registry)
Evaluator(program, engine=engine, learning_logger=logger).run()
print(logger.to_list())
```

An executable walkthrough lives in `notebooks/Learning_Log_Demo.ipynb`, which illustrates how to import the repo modules inside Jupyter and display the collected JSON using `IPython.display.JSON`.

To render the same records as compact human-readable messages (useful for notebooks or debugging), use `coherent.core.log_formatter`:

```python
from coherent.core.log_formatter import format_records

for line in format_records(logger.to_list()):
    print(line)
```
Output:
```
problem: (3 + 5) * 4
step1: 8 * 4
step2: 32
end: 32
```

Looking for quick notebook inputs? `notebooks/sample_programs.py` ships multiple reusable MathLang snippets:

```python
from notebooks.sample_programs import get_sample_program

sample = get_sample_program("binomial_square")
print(sample.description)
source = sample.source
```

## Notebook Cell Magic
Jupyter 上で DSL をそのままタイプして実行したい場合は、専用マジックを読み込んで `problem:`/`step:`/`end:` を直接入力できます。

1. `%load_ext coherent.tools.notebook_magic`
2. DSL セルの先頭に `%%mathlang [--mode symbolic|polynomial] [--no-meta]`
3. 以降の行に DSL を記述して実行

例:
```text
%load_ext coherent.tools.notebook_magic

%%mathlang
problem: (x - y)^2
step1: (x - y)(x - y)
step: x^2 -2xy +y^2
end: done
```
`x^2` や `2xy`、`(x - y)(x - y)` のような人間向け表記は自動で `x**2` / `2*x*y` / `(x - y)*(x - y)` に正規化され、SyntaxError を気にせず Notebook 上で MathLang を検証できます。

## Causal Analysis in Notebooks
Notebook から因果推論を呼び出す場合は、LearningLogger の記録を `coherent.core.causal.integration.run_causal_analysis` に渡すだけです。

```python
from coherent.core.causal.integration import run_causal_analysis
from coherent.core.causal.graph_utils import graph_to_text

records = logger.to_list()
engine, report = run_causal_analysis(records, include_graph=True)
report["explanations"]  # 各エラーの原因サマリ

# 修正候補（例: step ノードID）
# Heuristic Intensification により、"Freshman's Dream" などの誤用パターンは優先的に提案されます。
for error_id in report["errors"]:
    fix_nodes = engine.suggest_fix_candidates(error_id)
    print(error_id, [node.node_id for node in fix_nodes])

# グラフの簡易表示
print(graph_to_text(report["graph"]))
```

`suggest_fix_candidates` は、単純な直前ステップだけでなく、**誤用パターン（例: $(a+b)^n \to a^n+b^n$）** を検出して根本原因を特定・優先順位付けします。

`report["graph"]` にはノード・エッジ情報が入るため、テキスト可視化に加えて Graphviz/DOT ツールへもそのまま渡せます（`coherent.core.causal.graph_utils.graph_to_dot` 参照）。

## Roadmap (Draft)
1. **Phase 1** – Parser/Evaluator foundation (target: mid Nov 2025)
2. **Phase 2** – SymbolicAI integration & AST optimization (early Dec 2025)
3. **Phase 3** – Jupyter UI + learning logs (late Dec 2025)
4. **Phase 4** – Educational beta release v0.9 (Jan 2026)

## Environment Setup (macOS)
```bash
uv init
uv python install 3.12
uv python pin 3.12
uv venv
uv add sympy pyyaml pytest
```

## Contribution Basics
- Branching: `main` (stable), `dev`, `feature/*`
- Commits: `feat(parser): implement expression parsing`
- Tests: run `pytest` before PRs
- Documentation: update both `README.md` and specifications in `docs/` when altering scope or design.

### Release Checklist (Snapshot)
- [ ] Parser/Evaluator tests green (`pytest`).
- [ ] CLI usage in `README` matches `main.py` behavior for `--file`, `--code`, `--mode`, and `--hello-world-test`.
- [ ] Examples under `edu/examples/` (Edu) and `pro/examples/` (Pro) execute without modification.
- [ ] Symbolic mode verifies steps and prints rendered output.
- [ ] Polynomial mode expands polynomials and reports matching steps.
- [ ] `--hello-world-test` prints the expected Hello World trace.
- [ ] Specification updated for any CLI or directory layout changes.

For coordination guidelines, see `Agent.md`.
