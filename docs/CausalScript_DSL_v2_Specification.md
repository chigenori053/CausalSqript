# Coherent DSL v2 Specification (prepare-integrated)

Version: 2.0  
Status: Official Draft  
Audience: Implementers (Codex / GeminiCLI / Coherent Engine)  
Scope: DSL Grammar, Execution Model, Normalization, AST Interface

---

# 0. Overview

Coherent DSL v2 defines a four-phase, strictly ordered execution model:

```
problem → prepare → step* → end
```

The goal of DSL v2 is to unify the calculation workflow and ensure that all
input expressions pass through:

1. **Normalizer (Edu or Pro)**
2. **Parser (DSL)**
3. **AST Builder**
4. **Evaluator v2 (mistake / fatal model)**

This version incorporates the `prepare` phase, providing a flexible pre-processing
layer for normalization, symbolic preprocessing, counterfactual editing, and
AI-driven initialization.

---

# 1. Execution Model

## 1.1 Allowed Phase Order

The only valid sequence of top-level statements is:

```
ProblemDecl
PrepareDecl?     # optional
StepDecl+        # one or more steps
EndDecl
```

Any deviation is a *fatal error*.

---

# 2. Grammar Specification

## 2.1 BNF Grammar

```
Program ::= ProblemDecl PrepareDecl? StepDecl+ EndDecl

ProblemDecl ::= 'problem' ':' Expr

PrepareDecl ::= 'prepare' ':' PrepareBody
PrepareBody ::= Expr
              | Directive
              | 'auto'
              | ''   # empty prepare allowed

Directive ::= IDENT '(' DirectiveArgs? ')'
DirectiveArgs ::= (IDENT '=' STRING) (',' IDENT '=' STRING)*

StepDecl ::= 'step' StepId? ':' Expr
StepId ::= INTEGER

EndDecl ::= 'end' ':' (Expr | 'done')

Expr ::= A mathematical expression accepted by the Normalizer
```

---

## 2.2 Lark Grammar (machine-usable)

```
?start: statement+

?statement: problem
          | prepare
          | step
          | end

problem: "problem" ":" expr

prepare: "prepare" ":" prepare_body
prepare_body: expr
             | DIRECTIVE
             | "auto"
             | -> empty_prepare

step: "step" INTEGER? ":" expr

end: "end" ":" expr
   | "end" ":" "done"

# Expressions are normalized before this grammar.
# They follow strict SymPy/Python arithmetic syntax.
?expr: term
     | expr "+" term   -> add
     | expr "-" term   -> sub

?term: factor
     | term "*" factor -> mul
     | term "/" factor -> div

?factor: NUMBER        -> number
       | SYMBOL        -> var
       | "(" expr ")"
       | SYMBOL "(" expr ")"  -> funccall

DIRECTIVE: /[a-zA-Z_][a-zA-Z0-9_]*\(.*?\)/

%import common.CNAME -> SYMBOL
%import common.INT -> INTEGER
%import common.NUMBER
%import common.WS
%ignore WS
%ignore /#[^\n]*/
```

---

# 3. Normalizer Integration (Edu / Pro)

All expressions in DSL v2 are processed by a Normalizer before parsing.

```
original_expr
    ↓
Normalizer (Edu or Pro)
    ↓
strict_expr
    ↓
Parser → AST
```

## 3.1 Normalizer Output Requirements

Normalizer MUST output:

- SymPy-compatible expressions
- explicit multiplication (`*`)
- explicit powers (`**`)
- normalized whitespace
- well-formed parentheses

### Example Transformations

| Input (Edu)       | Output (strict) |
|--------------------|------------------|
| `2x`               | `2*x`            |
| `(x-y)(x+y)`       | `(x-y)*(x+y)`    |
| `x^2`              | `x**2`           |
| `sin x`            | `sin(x)`         |
| `1/2x`             | `(1/2)*x`        |

Pro Normalizer may allow strict-only, hybrid, or passthrough modes.

---

# 4. AST Specification

```
ProgramNode
    problem: ProblemNode
    prepare: PrepareNode | None
    steps: [StepNode]
    end: EndNode
```

### Node Definitions

```
ProblemNode:
    expr: str

PrepareNode:
    kind: "expr" | "directive" | "auto" | "empty"
    expr: str?         # if kind == expr
    directive: str?    # if kind == directive

StepNode:
    step_id: int | None
    expr: str

EndNode:
    is_done: bool
    expr: str | None
```

---

# 5. Evaluator v2 Compatibility

All DSL v2 statements map directly to Evaluator v2 phases:

| DSL node | Evaluator function | Behavior |
|----------|--------------------|----------|
| ProblemNode | set(expr) | Initializes expression |
| PrepareNode | prepare(expr/dir) | symbolic preprocess |
| StepNode | check_step(expr) | ok / mistake / fatal |
| EndNode | finalize(expr) | ok / mistake / fatal |

Evaluator v2 error model:

| status | meaning |
|--------|----------|
| **ok** | equivalent expressions |
| **mistake** | computation mismatch |
| **fatal** | parsing/engine failure |

---

# 6. Prepare Phase Specification

`prepare` provides symbolic preprocessing.

### 6.1 prepare as expression
```
prepare: (x - y)*(x - y)
```

Equivalent check is performed:
```
problem_expr ≡ prepare_expr
```

### 6.2 prepare: auto
Performs default symbolic preprocessing:
- Normalizer (strict)
- symbolic simplify
- optional expansion

### 6.3 prepare: directive
Examples:
```
prepare: expand()
prepare: factor()
prepare: normalize()
prepare: counterfactual(step=1, expr="5")
```

---

# 7. Examples

## 7.1 Standard algebra

```
problem: (x - y)^2
prepare: auto
step1: (x - y)*(x + y)
step2: x^2 - 2*x*y + y^2
end: x^2 - 2*x*y + y^2
```

## 7.2 Manual prepare expression

```
problem: (x - y)^2
prepare: (x - y)*(x - y)
step1: x^2 - 2*x*y + y^2
end: x^2 - 2*x*y + y^2
```

## 7.3 No prepare

```
problem: x + y
step1: y + x
end: x + y
```

---

# 8. Error Rules

## 8.1 fatal (syntax/engine failure)
- invalid expression after Normalizer
- invalid DSL structure
- missing problem
- prepare after first step
- malformed directive

## 8.2 mistake (non-equivalent steps)
- step not equivalent
- prepare not equivalent
- end mismatch

---

# 9. Backward Compatibility

DSL v1 scripts are valid DSL v2 scripts because `prepare` is optional.

---

# 10. Implementation Notes

- Normalizer MUST run before DSL parsing
- prepare is the only optional phase
- directives MUST NOT modify AST directly
- Evaluator v2 ensures strict phase ordering
- test cases must include no-prepare / expr-prepare / directive-prepare

---

# End of Specification
