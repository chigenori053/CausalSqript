# Coherent Evaluator v2 — Specification (Draft v2.0)

Document Version: v2.0  
Status: Internal Design Specification  
Scope: Evaluator / Logging System / Error Model  
Author: Coherent Project Specification

---

# 0. Overview

This document defines the complete specification for **Coherent Evaluator v2**,  
including its execution model (`problem → step* → end`) and the unified logging  
system (`ok`, `mistake`, `fatal`).

Evaluator v2 introduces a new error-handling philosophy:

- **Computation mistakes are not exceptions.**  
  They are logged as `mistake` and evaluation continues.

- **Only fatal errors produce exceptions.**  
  These are situations where Coherent cannot continue evaluating the program  
  (e.g., syntax errors, corrupted AST, SymPy failures).

This design aligns with Coherent’s educational philosophy:
**"Even wrong steps are meaningful steps in the learning process."**

---

# 1. Design Principles

## 1.1 Computation mistakes are normal events
Incorrect intermediate expressions and wrong final answers must not cause exceptions.  
They are logged as:

```
status: "mistake"
```

Evaluator continues execution.

## 1.2 Fatal errors indicate program breakdown
Only unrecoverable errors stop evaluation and raise exceptions:

- Syntax errors  
- Invalid or incomplete expressions  
- SymbolicEngine conversion failures  
- AST structural corruption  
- Missing `problem` before `step` or `end`  
- KnowledgeRegistry load failure

Fatal errors are logged with:

```
status: "fatal"
```

## 1.3 Unified log format
All phases output logs with the same structure.

---

# 2. Evaluation States

Evaluator v2 classifies all evaluation results into three states:

| status    | meaning                         | exception | example |
|-----------|---------------------------------|-----------|---------|
| ok        | successful evaluation           | no        | correct step |
| mistake   | computational error (user side) | no        | wrong step / wrong end |
| fatal     | execution impossible            | yes       | syntax error, internal failure |

---

# 3. Evaluator v2 Flow

```
Parser → AST → Evaluator.run() → LearningLogger
```

# 3.1 Execution Sequence

1. `problem`  
2. Zero or more `step`  
3. `end`  

Logs are always recorded regardless of correctness.

# 3.2 run() return value

```
True  → no fatal errors (only ok/mistake)
False → fatal was detected before raising (internal use)
```

If a fatal error occurs, an exception is raised and run() does not complete normally.

---

# 4. Phase Specifications

## 4.1 problem phase

### ok
Logged when the initial expression is valid.

```
{
  "phase": "problem",
  "expression": "(3+6)*4",
  "status": "ok"
}
```

### fatal
Occurs when:

- Expression cannot be parsed or converted
- Duplicate `problem`
- AST corrupted

```
{
  "phase": "problem",
  "status": "fatal",
  "meta": {
    "exception": "InvalidExprError",
    "message": "Unexpected end of input"
  }
}
```

---

## 4.2 step phase

### ok (equivalent to previous expression)

```
{
  "phase": "step",
  "expression": "9*4",
  "status": "ok",
  "meta": {"rule": "ARITH-ADD-001"}
}
```

### mistake (not equivalent)

```
{
  "phase": "step",
  "expression": "7*4",
  "status": "mistake",
  "meta": {
    "reason": "invalid_step",
    "expected": "9*4"
  }
}
```

### fatal
Occurs when:

- No preceding `problem`
- Unparseable expression
- AST corruption
- SymbolicEngine conversion error

---

## 4.3 end phase

### ok (correct final expression)

```
{
  "phase": "end",
  "expression": "36",
  "status": "ok"
}
```

### mistake (final expression differs)

```
{
  "phase": "end",
  "expression": "35",
  "status": "mistake",
  "meta": {
    "reason": "final_result_mismatch",
    "expected": "36"
  }
}
```

### fatal
Occurs when:

- `end` appears with no `problem`
- Multiple `end` declarations
- Expression is invalid
- Internal state invalid

---

# 5. Fatal Error Definition

Fatal errors represent **program-level failure**, not student mistakes.

## 5.1 Fatal error categories

| category | description | example |
|----------|-------------|---------|
| Syntax error | DSL cannot be parsed | `stpe: 3+4` |
| Expression error | Expression incomplete | `problem: 3+` |
| SymbolicEngine failure | Cannot convert to SymPy | `9*?` |
| AST structural error | Node missing fields | StepNode with no expr |
| Execution order error | `end` before `problem` | invalid sequence |
| Knowledge failure | JSON/YAML corrupted | load error |

Fatal errors raise exceptions.

---

# 6. Logging Specification (LearningLogger v2)

## 6.1 Unified Log Structure

```
{
  "phase": "<problem|step|end|internal>",
  "expression": "<string or null>",
  "rendered": "<string>",
  "status": "<ok|mistake|fatal>",
  "rule_id": "<string or null>",
  "meta": { ... }
}
```

## 6.2 status values

### ok
Correct evaluation.

### mistake
Incorrect computation (user error).  
Never throws exceptions.

### fatal
System/engine/DSL error.  
Exception thrown after log is recorded.

---

# 7. meta Field Specifications

## ok example

```
"meta": {
  "rule": "ARITH-ADD-001",
  "explanation": "Expressions are equivalent."
}
```

## mistake example

```
"meta": {
  "reason": "invalid_step",
  "expected": "9*4"
}
```

## fatal example

```
"meta": {
  "exception": "InvalidExprError",
  "message": "Unexpected end of expression"
}
```

---

# 8. Log Examples

## 8.1 All correct

```
[
  {"phase": "problem", "expression": "(3+6)*4", "status": "ok"},
  {"phase": "step", "expression": "9*4", "status": "ok"},
  {"phase": "end", "expression": "36", "status": "ok"}
]
```

## 8.2 Contains mistakes (no fatal)

```
[
  {"phase": "problem", "expression": "(3+6)*4", "status": "ok"},
  {
    "phase": "step",
    "expression": "7*4",
    "status": "mistake",
    "meta": {"reason": "invalid_step", "expected": "9*4"}
  },
  {
    "phase": "end",
    "expression": "35",
    "status": "mistake",
    "meta": {"reason": "final_result_mismatch", "expected": "36"}
  }
]
```

## 8.3 Fatal example

```
[
  {
    "phase": "problem",
    "expression": "3 +",
    "status": "fatal",
    "meta": {
      "exception": "InvalidExprError",
      "message": "Unexpected end of input"
    }
  }
]
```

---

# 9. Differences from Evaluator v1

| item | v1 | v2 |
|------|-----|-----|
| Wrong step | Exception | mistake |
| Wrong final result | Exception | mistake |
| fatal | Several cases | only unrecoverable errors |
| log format | inconsistent | unified |
| run() behavior | stops often | stops only on fatal |
| CLI stability | low | high |

---

# 10. Future Extensions

- More detailed mistake categories  
- Standardized fatal error codes  
- AI step / explain integration  
- Learning analytics based on mistake history  

---

# End of Specification
