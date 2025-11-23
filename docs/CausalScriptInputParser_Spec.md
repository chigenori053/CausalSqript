# CausalScriptInputParser Specification
Version: 0.1
Status: Draft
Target: Codex / CausalScript Core Engine

## 1. Purpose
CausalScriptInputParser converts *human-friendly mathematical expressions* into
CausalScript-compatible normalized expressions usable by the main DSL parser.

This parser hides Python/SymPy-like syntax from educational users while producing
valid internal expressions for:
- Parser (DSL v2.5)
- SymbolicEngine
- Evaluator
- KnowledgeRegistry

This specification is fully machine-readable by Codex.

---

## 2. Input → Output Overview

### Input (user-facing)
Examples:
- (x - y)^2
- 2xy + 3x
- (a+b)^3
- √x + 2y
- (x - 1)(x + 1)

### Output (internal normalized CausalScript/SymPy form)
- (x - y)**2
- 2*x*y + 3*x
- (a+b)**3
- sqrt(x) + 2*y
- (x - 1)*(x + 1)

---

## 3. Features

### 3.1 Supported transformations
| Feature | Example Input | Output |
|--------|---------------|--------|
| Power operator | x^2 | x**2 |
| Unicode power | x² | x**2 |
| Implicit multiplication | 2xy | 2*x*y |
| sqrt function | √x | sqrt(x) |
| Paren adjacency | (x+1)(x-1) | (x+1)*(x-1) |
| Number + paren | 3(x+1) | 3*(x+1) |

### 3.2 Not supported (v0.1)
- sin x → sin(x) (v0.2)
- ambiguous fractions like 1/2x

---

## 4. Processing Pipeline

```
User String
   ↓
Tokenizer
   ↓
Unicode Normalizer
   ↓
Power Operator Normalizer
   ↓
Implicit Multiplication Resolver
   ↓
Function Normalizer
   ↓
Output Expression
```

---

## 5. Tokenizer Specification

### Token types:
- NUMBER
- IDENT
- OP (+ - * / ^ **)
- UNICODE (² ³ √)
- PAREN ( ( ) )
- SPACE (ignored)

Example:
```
2xy → ["2", "x", "y"]
√x + 1 → ["√", "x", "+", "1"]
```

---

## 6. Implicit Multiplication Rules

Insert "*" for:
- IDENT IDENT → x y → x*y
- NUMBER IDENT → 2x → 2*x
- IDENT PAREN → x(x+1) → x*(x+1)
- PAREN IDENT → (x+1)y → (x+1)*y
- NUMBER PAREN → 3(x+1) → 3*(x+1)
- PAREN PAREN → (x)(y)

Do not insert "*" after:
- OP tokens
- function names (sqrt)

---

## 7. Power Normalization Rules

- "^" → "**"
- "²" → "**2"
- "³" → "**3"
- (x+1)² → (x+1)**2

---

## 8. Function Normalization

### sqrt:
- √x → sqrt(x)
- √(x+1) → sqrt(x+1)

Extendable to:
- sin x → sin(x)
- log x → log(x)

---

## 9. Output Requirements

The output must:
- Use "**" for exponentiation
- Explicitly insert "*" for all multiplications
- Use ASCII except identifiers
- Be a valid SymPy/CausalScript-compatible expression

Example final output:
```
3*x*(x+1)**2
sqrt(x) + 2*y
```

---

## 10. Integration with DSL Parser

CausalScriptInputParser is applied **before** DSL parsing:

```
expr = CausalScriptInputParser.normalize("(x-y)^2")
source = f"problem: {expr}"
program = Parser(source).parse()
```

Evaluator receives normalized expressions for:
- problem
- prepare items
- step.before
- step.after
- end
- counterfactual.expect

---

## 11. Public API

```python
class CausalScriptInputParser:
    @staticmethod
    def normalize(expr: str) -> str:
        ...
    @staticmethod
    def tokenize(expr: str) -> list:
        ...
    @staticmethod
    def normalize_unicode(tokens: list) -> list:
        ...
    @staticmethod
    def normalize_power(tokens: list) -> list:
        ...
    @staticmethod
    def insert_implicit_multiplication(tokens: list) -> list:
        ...
    @staticmethod
    def normalize_functions(tokens: list) -> list:
        ...
    @staticmethod
    def to_string(tokens: list) -> str:
        ...
```

---

## 12. Test Cases

### T1
```
input: x^2
output: x**2
```

### T2
```
input: 2xy
output: 2*x*y
```

### T3
```
input: (a+b)²
output: (a+b)**2
```

### T4
```
input: √x + y
output: sqrt(x) + y
```

### T5
```
input: 3x(x+1)^2
output: 3*x*(x+1)**2
```

---

## 13. Error Handling
Raise:
- CausalScriptInputError("invalid parentheses")
- CausalScriptInputError("unknown unicode token")
- CausalScriptInputError("ambiguous implicit multiplication")

---

## 14. Versioning

v0.1 — Minimal viable parser for Demo版  
v0.2 — Add trig/log support  
v1.0 — Complete symbolic grammar

---

End of Spec
