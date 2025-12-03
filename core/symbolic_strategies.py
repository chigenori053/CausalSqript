"""
Strategies for symbolic manipulation based on mathematical categories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from fractions import Fraction
import math

from .errors import InvalidExprError, EvaluationError
from .simple_algebra import SimpleAlgebra
from .math_category import MathCategory

try:
    import sympy as _sympy
except ImportError:
    _sympy = None


class SymbolicStrategy(ABC):
    """Base class for symbolic manipulation strategies."""

    def __init__(self, fallback_evaluator: Any = None):
        self._fallback = fallback_evaluator

    @abstractmethod
    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        """
        Check equivalence. Returns True/False if definitive, None if cannot decide.
        'engine' is passed to allow access to shared utilities like to_internal.
        """
        pass

    @abstractmethod
    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        """
        Simplify expression. Returns simplified string or None.
        """
        pass

    @abstractmethod
    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        """
        Convert to LaTeX. Returns LaTeX string or None.
        """
        pass


class ArithmeticStrategy(SymbolicStrategy):
    """Strategy optimized for arithmetic (numeric) expressions."""

    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        # If both are purely numeric, we can evaluate them
        if engine.is_numeric(expr1) and engine.is_numeric(expr2):
            try:
                # Use numeric evaluation
                val1 = engine.evaluate_numeric(expr1, {})
                val2 = engine.evaluate_numeric(expr2, {})
                
                # Handle float precision
                if isinstance(val1, float) or isinstance(val2, float):
                    return abs(float(val1) - float(val2)) < 1e-9
                
                return val1 == val2
            except Exception:
                pass
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        if engine.is_numeric(expr):
            try:
                val = engine.evaluate_numeric(expr, {})
                return str(val)
            except Exception:
                pass
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        # Arithmetic usually needs \cdot for multiplication
        # But we can let the default engine handle it if we don't have special needs
        return None


class AlgebraStrategy(SymbolicStrategy):
    """Strategy for general algebraic expressions."""

    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        # This is the default fallback behavior usually, but we can be explicit
        if _sympy:
            try:
                internal1 = engine.to_internal(expr1)
                internal2 = engine.to_internal(expr2)
                diff = _sympy.simplify(internal1 - internal2)
                return diff == 0
            except Exception:
                pass
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        if _sympy:
            try:
                internal = engine.to_internal(expr)
                simplified = _sympy.simplify(internal)
                return str(simplified)
            except Exception:
                pass
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        # Algebra often prefers implicit multiplication
        if _sympy:
            try:
                from sympy.parsing.sympy_parser import parse_expr
                local_dict = {"e": _sympy.E, "pi": _sympy.pi}
                internal = parse_expr(expr.replace("^", "**"), evaluate=False, local_dict=local_dict)
                return _sympy.latex(internal, mul_symbol="")
            except Exception:
                pass
        return None


class CalculusStrategy(SymbolicStrategy):
    """Strategy for calculus (integrals, derivatives)."""

    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        if _sympy:
            try:
                # For calculus, we often need to 'doit()' to evaluate integrals/derivatives
                internal1 = engine.to_internal(expr1)
                internal2 = engine.to_internal(expr2)
                
                # Try evaluating pending operations
                if hasattr(internal1, 'doit'):
                    internal1 = internal1.doit()
                if hasattr(internal2, 'doit'):
                    internal2 = internal2.doit()
                    
                diff = _sympy.simplify(internal1 - internal2)
                return diff == 0
            except Exception:
                pass
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        if _sympy:
            try:
                internal = engine.to_internal(expr)
                if hasattr(internal, 'doit'):
                    internal = internal.doit()
                simplified = _sympy.simplify(internal)
                return str(simplified)
            except Exception:
                pass
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        # Standard LaTeX is usually fine, but maybe we want specific integral notation?
        return None


class LinearAlgebraStrategy(SymbolicStrategy):
    """Strategy for linear algebra."""
    # Placeholder for now, can be expanded
    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        return None


class StatisticsStrategy(SymbolicStrategy):
    """Strategy for statistics."""
    # Placeholder
    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        return None


class GeometryStrategy(SymbolicStrategy):
    """Strategy for geometry."""
    # Placeholder
    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        return None
