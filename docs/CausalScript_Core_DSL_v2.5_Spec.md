# Coherent Core DSL v2.5 Specification
Version: 2.5
Status: Draft
Target: Codex / Coherent Core Engine

## 1. Purpose
This document defines the **Core DSL v2.5** used by Coherent to describe:
- mathematical problems
- step-by-step reasoning
- causal analysis
- fuzzy judgment
- counterfactual evaluation
- execution configuration

The DSL is designed to be parsed into AST nodes consumed by:
- Parser
- Evaluator
- KnowledgeRegistry
- FuzzyJudge
- CausalEngine
- CounterfactualEngine

This specification is fully machine-readable and suitable for Codex-based tooling.

---

## 2. File Format
Coherent Core DSL files use UTF-8 text with the extension:
```
.mlang
```

Sections are written using:
```
keyword: value
keyword:
    key: value
```

Comments:
```
# this is a comment
```

---

## 3. Top-Level DSL Structure
A valid DSL document contains the following optional sections:

```
meta:
config:
mode:
problem:
prepare:
step:
end:
counterfactual:
```

Order does not matter, but a recommended order is:

```
meta:
config:
mode:
problem:
prepare:
step:
end:
counterfactual:
```

---

## 4. Section Specifications

### 4.1 meta:
Metadata describing the problem.

Example:
```
meta:
    id: demo_01
    topic: arithmetic
    level: beginner
```

Keys are arbitrary strings.

---

### 4.2 config:
Execution options passed to Evaluator, FuzzyJudge, CausalEngine.

Example:
```
config:
    fuzzy-threshold: 0.65
    causal: true
    normalize: true
```

Supported keys:
- `fuzzy-threshold: float`
- `causal: bool`
- `normalize: bool`

---

### 4.3 mode:
Defines evaluator mode.

```
mode: strict
```

Supported modes:
- `strict` — exact rule matching only
- `fuzzy` — allow approximate matches
- `causal` — enable causal reasoning
- `cf` — enable counterfactual reasoning

---

### 4.4 problem:
The initial mathematical expression.

```
problem: (3+5)*4
```

Parsed into:
- problem.expr: string
- problem.ast: AST

---

### 4.5 prepare:
Optional preparation steps that may include variable declarations or preliminary reasoning.

Example:
```
prepare:
    - x = 5
    - y = x + 2
```

---

### 4.6 step:
One or more step sections describing expected transformations.

Format:
```
step:
    before: (3+5)*4
    after: 8*4
    note: "simplify the addition"
```

Multiple steps allowed:
```
step:
    before: ...
    after: ...

step:
    before: ...
    after: ...
```

Fields:
- `before: string`
- `after: string`
- `note: string (optional)`

---

### 4.7 end:
Final expected result.

```
end: 32
```

Evaluator ensures the final expression matches.

---

### 4.8 counterfactual:
Counterfactual simulation block.

```
counterfactual:
    assume:
        x: 10
    expect: 3*x + 2
```

Fields:
- assume: dict of variable interventions
- expect: expression to evaluate under intervention

---

## 5. AST Mapping Rules

### Expressions
Parsed using Coherent Expression Grammar:

```
Expr := Term (("+"|"-") Term)*
Term := Factor (("*"|"/") Factor)*
Factor := Number | Identifier | "(" Expr ")"
```

Mapped into AST nodes:
- Int
- Var
- Add
- Sub
- Mul
- Div

---

### DSL Nodes → AST Nodes

| DSL Section | AST Node |
|-------------|----------|
| meta | MetaNode |
| config | ConfigNode |
| mode | ModeNode |
| problem | ProblemNode |
| prepare | PrepareNode |
| step | StepNode (with optional before/note) |
| end | EndNode |
| counterfactual | CounterfactualNode |

---

## 6. Example (Complete DSL v2.5)

```
meta:
    id: sample_01
    topic: arithmetic

config:
    causal: true
    fuzzy-threshold: 0.5

mode: causal

problem: (3 + 5) * 4

step:
    before: (3+5)*4
    after: 8*4
    note: "simplify addition"

step:
    before: 8*4
    after: 32
    note: "multiplication"

end: 32

counterfactual:
    assume:
        x: 10
    expect: 3*x + 2
```

---

## 7. Error Conditions

A DSL is invalid when:

- Missing `problem:` section
- step.before or step.after missing
- invalid expression syntax
- unknown mode
- counterfactual.expect missing

Evaluator must report:
- SyntaxError
- StepMismatchError
- RuleNotFoundError
- CounterfactualError

---

## 8. Versioning

DSL v2.5 is backward compatible with v2.0 and v1.x.

Future versions may add:
- hint:
- explain:
- visualize:

---

End of Spec
