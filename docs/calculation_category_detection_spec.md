# Implementation Directive: Calculation Category Detection & Engine Dispatching
**Status**: Draft
**Target**: Coherent Core Architecture

## Goal
Automatically detect the mathematical category (Algebra, Geometry, Calculus, etc.) from an input expression and dispatch processing to the optimal calculation engine.

## 1. Overview
Currently, `ComputationEngine` holds multiple specialized engines (GeometryEngine, CalculusEngine, etc.), but lacks logic to automatically switch between them based on the input expression. This task introduces `CategoryAnalyzer` to parse expression strings and integrates routing logic into `CoreRuntime` and `ComputationEngine`.

## 2. Architecture Changes

| Component | Change |
| :--- | :--- |
| `core/category_analyzer.py` | **[New]** Implements logic to determine the category from an expression. |
| `core/computation_engine.py` | Adds methods to use `CategoryAnalyzer` and delegate to specialized sub-engines. |
| `core/core_runtime.py` | Determines category at `set()` time and stores it in the context. |

## 3. Implementation Steps

### Step 1: Implement `CategoryAnalyzer`
**File**: `core/category_analyzer.py` (New)

Implement a class that determines `MathCategory` based on tokens and keywords.

```python
from enum import Enum
from typing import Set
from core.input_parser import CoherentInputParser
from core.math_category import MathCategory

class CategoryAnalyzer:
    """Analyzes expression strings to determine their mathematical category."""

    # Keywords mapping for specialized domains
    _KEYWORDS = {
        MathCategory.CALCULUS: {"diff", "integrate", "limit", "d/dx", "∫", "Subs", "Derivative", "Integral"},
        MathCategory.GEOMETRY: {"Point", "Line", "Circle", "Triangle", "Segment", "Ray", "Polygon", "area", "volume"},
        MathCategory.LINEAR_ALGEBRA: {"Matrix", "Vector", "dot", "cross", "det", "eigenvals", "eigenvects", "inverse", "transpose"},
        MathCategory.STATISTICS: {"mean", "median", "mode", "variance", "std_dev", "normal", "uniform", "binomial", "pdf", "cdf"},
    }

    @staticmethod
    def detect(expr: str) -> MathCategory:
        try:
            # Reuse existing tokenizer for consistency
            # Note: CoherentInputParser might need to be imported or we use a simple split/regex if dependency is heavy
            # For now, assuming we can access tokens.
            # If CoherentInputParser is complex, we can use a simpler tokenizer or regex.
            tokens = set(CoherentInputParser.tokenize(expr))
        except Exception:
            # Fallback for simple strings
            tokens = set(expr.replace("(", " ").replace(")", " ").replace(",", " ").split())

        # Check specific keywords first
        for category, keywords in CategoryAnalyzer._KEYWORDS.items():
            if not tokens.isdisjoint(keywords):
                return category
        
        # Check for variables (Algebra vs Arithmetic)
        # Exclude constants like pi, e from variable detection
        # Simple heuristic: if there are alpha characters that are not keywords/constants
        if any(t.isidentifier() and t not in {"pi", "e", "done", "true", "false"} for t in tokens):
            return MathCategory.ALGEBRA
            
        return MathCategory.ARITHMETIC
```

### Step 2: Extend `ComputationEngine`
**File**: `core/computation_engine.py`

Add category detection and optimized computation methods.

```python
# [Imports]
from .category_analyzer import CategoryAnalyzer
from .math_category import MathCategory

class ComputationEngine:
    # ... (existing init)

    def detect_category(self, expr: str) -> MathCategory:
        """Detect the mathematical category of an expression."""
        return CategoryAnalyzer.detect(expr)

    def compute_optimized(self, expr: str, category: MathCategory | None = None) -> Any:
        """
        Perform computation using the engine optimized for the detected category.
        Returns the simplified string or computed value.
        """
        target_category = category or self.detect_category(expr)

        # Dispatch to specialized engines
        if target_category == MathCategory.CALCULUS:
            # Attempt to solve derivatives/integrals if explicitly requested in syntax
            if "diff" in expr or "d/dx" in expr or "Derivative" in expr:
                # Fallback to symbolic simplify which handles 'diff(x**2, x)' via SymPy
                pass 

        elif target_category == MathCategory.GEOMETRY and hasattr(self, 'geometry_engine'):
            # If the expression evaluates to a Geometric entity, return its properties
            pass
            
        # Default fallback: Symbolic simplification
        return self.simplify(expr)
```

### Step 3: Integrate into `CoreRuntime`
**File**: `core/core_runtime.py`

Determine category in `set()` and store it.

```python
# [Imports]
from .math_category import MathCategory

class CoreRuntime(Engine):
    def __init__(self, ...):
        # ... (existing init)
        self._current_category: MathCategory = MathCategory.ALGEBRA

    def set(self, expr: str) -> None:
        super().set(expr) # Sets self._current_expr
        self._equation_mode = "=" in expr
        
        # Auto-detect and store category
        self._current_category = self.computation_engine.detect_category(self._normalize_expression(expr))
        
        # Propagate to SymbolicEngine (Strategy Pattern)
        # We can map single category to list for set_context
        self.computation_engine.symbolic_engine.set_context([self._current_category])
        
        # (Optional) Log detection result
        print(f"DEBUG: Detected Category: {self._current_category.value}")
```

## 4. Verification Plan
Create `tests/test_category_detection.py` to verify:
- Arithmetic vs Algebra detection.
- Keyword-based detection for Calculus, Geometry, etc.
- Integration with `CoreRuntime.set()`.

```python
def test_category_detection():
    from core.category_analyzer import CategoryAnalyzer
    from core.math_category import MathCategory
    
    assert CategoryAnalyzer.detect("1 + 2") == MathCategory.ARITHMETIC
    assert CategoryAnalyzer.detect("x^2 + 2*x + 1") == MathCategory.ALGEBRA
    assert CategoryAnalyzer.detect("diff(x^2, x)") == MathCategory.CALCULUS
    assert CategoryAnalyzer.detect("Triangle(p1, p2, p3)") == MathCategory.GEOMETRY
    assert CategoryAnalyzer.detect("mean([1, 2, 3])") == MathCategory.STATISTICS
```
