# CausalScript Core Architecture Extension Specification (Computation / Validation / Hinting)

## Overview
This document defines the official updated specification for integrating Computation, Answer Validation, and Hint Generation into CausalScript Core Architecture. It is formatted for compatibility with Codex, Gemini, and other LLM-based code agents.

## 1. Architecture Structure
```
CausalScript Source (.mlang)
    ↓ Parser
AST (Program)
    ↓ Evaluator
    ↓ CoreRuntime
        ├── ComputationEngine
        ├── ValidationEngine
        └── HintEngine
    ↓ LearningLogger
```

## 2. ExerciseSpec (Problem Specification)
```python
@dataclass
class ExerciseSpec:
    id: str
    target_expression: str
    validation_mode: str = "symbolic_equiv"
    canonical_form: str | None = None
    intermediate_steps: list[str] | None = None
    hint_rules: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
```

## 3. ComputationEngine
Provides symbolic and numeric evaluation using existing SymbolicEngine and SymPy.

### Interface
```python
class ComputationEngine:
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic_engine = symbolic_engine
        self.variables = {}

    def to_sympy(self, expr: str | ASTNode) -> sympy.Expr: ...
    def simplify(self, expr): ...
    def numeric_eval(self, expr): ...
    def bind(self, name, value): ...
```

## 4. ValidationEngine
Performs mathematical equivalence checking and format-based validation.

### Interface
```python
class ValidationEngine:
    def check_answer(self, user_expr, spec: ExerciseSpec) -> ValidationResult:
        ...
```

## 5. HintEngine
Produces structured hints based on symbolic differences and known error patterns.

### Interface
```python
class HintEngine:
    def generate_hint(self, user_expr, target_expr, spec=None) -> HintResult:
        ...
```

## 6. CoreRuntime (Execution Orchestrator)
Integrates the three engines and connects to Evaluator and LearningLogger.

### Interface
```python
class CoreRuntime:
    def __init__(self, computation, validator, hinter, logger=None, exercise_spec=None):
        ...

    def eval_step(self, before_expr, step_expr):
        ...
```

## 7. Evaluator Integration
Evaluator is extended to call CoreRuntime for:
- step validation
- final answer checking
- hint generation
- enhanced logging

## 8. CLI / Notebook Usage
```python
runtime.validator.check_answer("x^2 - 2xy + y^2", spec)
runtime.hinter.generate_hint("x^2 - 2*y + y^2", spec.target_expression)
```

## 9. Roadmap
1. Implement CoreRuntime skeleton
2. Integrate engines with Evaluator
3. Add I/O schema for ExerciseSpec
4. Build CLI & Notebook demo
