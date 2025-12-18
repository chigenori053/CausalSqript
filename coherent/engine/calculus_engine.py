"""Calculus helpers for derivative and integral calculations."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .computation_engine import ComputationEngine
from .errors import EvaluationError, InvalidExprError
from .simple_algebra import SimpleAlgebra

try:  # pragma: no cover - optional dependency
    import sympy as _sympy
except Exception:  # pragma: no cover
    _sympy = None


class CalculusEngine:
    """Provides derivative and integral utilities with polynomial fallbacks."""

    def __init__(self, computation_engine: ComputationEngine) -> None:
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine

    def derivative(
        self,
        expr: str,
        variable: str = "x",
        at: float | None = None,
    ) -> Dict[str, Any]:
        """
        Compute derivative expression and optionally evaluate at a point.
        """
        derivative_expr: Optional[str] = None
        if _sympy is not None:
            internal = self.symbolic_engine.to_internal(expr)
            sym_var = _sympy.Symbol(variable)
            diff_expr = _sympy.diff(internal, sym_var)
            derivative_expr = str(diff_expr)
        else:
            try:
                derivative_expr = SimpleAlgebra.derivative(expr, variable)
            except InvalidExprError:
                derivative_expr = None

        value = None
        if at is not None:
            value = self._evaluate_derivative(expr, variable, at, derivative_expr)

        if derivative_expr is None:
            derivative_expr = "numeric_only"

        return {
            "expression": derivative_expr,
            "value": value,
        }

    def _evaluate_derivative(
        self,
        expr: str,
        variable: str,
        at: float,
        derivative_expr: Optional[str],
    ) -> Optional[float]:
        if derivative_expr and derivative_expr != "numeric_only":
            try:
                result = self.computation_engine.numeric_eval(
                    derivative_expr, {variable: at}
                )
                if isinstance(result, dict):
                    return None
                return float(result)
            except (EvaluationError, InvalidExprError):
                pass
        # fallback to numeric approximation
        h = 1e-5
        upper = self._numeric_eval(expr, variable, at + h)
        lower = self._numeric_eval(expr, variable, at - h)
        if upper is None or lower is None:
            return None
        return (upper - lower) / (2 * h)

    def integral(
        self,
        expr: str,
        variable: str = "x",
        lower: float | None = None,
        upper: float | None = None,
    ) -> Dict[str, Any]:
        """
        Compute indefinite or definite integrals.
        """
        expression = None
        if _sympy is not None:
            internal = self.symbolic_engine.to_internal(expr)
            sym_var = _sympy.Symbol(variable)
            expression = str(_sympy.integrate(internal, sym_var))
        else:
            try:
                expression = SimpleAlgebra.integral(expr, variable)
            except InvalidExprError:
                expression = None

        value = None
        if lower is not None and upper is not None:
            value = self._definite_integral(expr, variable, lower, upper)

        if expression is None:
            expression = "numeric_only"

        return {"expression": expression, "value": value}

    def _numeric_eval(self, expr: str, variable: str, at: float) -> Optional[float]:
        try:
            result = self.computation_engine.numeric_eval(expr, {variable: at})
            if isinstance(result, dict):
                return None
            return float(result)
        except EvaluationError:
            return None

    def _definite_integral(
        self,
        expr: str,
        variable: str,
        lower: float,
        upper: float,
    ) -> Optional[float]:
        if _sympy is not None:
            internal = self.symbolic_engine.to_internal(expr)
            sym_var = _sympy.Symbol(variable)
            result = _sympy.integrate(internal, (sym_var, lower, upper))
            return float(result)

        # numeric Simpson's rule fallback
        n = 200  # even number
        if upper < lower:
            lower, upper = upper, lower
        h = (upper - lower) / n
        result = self._numeric_eval(expr, variable, lower) or 0.0
        result += self._numeric_eval(expr, variable, upper) or 0.0
        for i in range(1, n):
            coeff = 4 if i % 2 == 1 else 2
            point = lower + i * h
            value = self._numeric_eval(expr, variable, point)
            if value is None:
                continue
            result += coeff * value
        return result * h / 3

    def slope_of_tangent(
        self,
        expr: str,
        variable: str,
        at: float,
    ) -> Optional[float]:
        derivative = self.derivative(expr, variable, at=at)
        return derivative["value"]

    def area_under_curve(
        self,
        expr: str,
        variable: str,
        lower: float,
        upper: float,
    ) -> Optional[float]:
        integral = self.integral(expr, variable, lower=lower, upper=upper)
        return integral["value"]
