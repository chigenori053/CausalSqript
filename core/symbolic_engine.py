"""Symbolic manipulation helpers built on top of SymPy (with a fallback)."""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Sequence, Set

from .errors import InvalidExprError, EvaluationError
from .simple_algebra import SimpleAlgebra

try:  # pragma: no cover - SymPy is an optional dependency at import time.
    import sympy as _sympy
except Exception:  # pragma: no cover
    _sympy = None


_SAMPLE_ASSIGNMENTS: Sequence[Dict[str, int]] = (
    {"x": -2, "y": 1, "z": 3, "a": 1},
    {"x": 0, "y": 0, "z": 0, "a": 2, "b": -1},
    {"x": 1, "y": 2, "z": -1, "b": 3, "c": 4},
    {"a": 2, "b": 5, "c": -3},
    {},
)


class _FallbackEvaluator:
    """Very small arithmetic evaluator used when SymPy is unavailable."""

    def parse(self, expr: str) -> py_ast.AST:
        expr = expr.replace("^", "**")
        try:
            tree = py_ast.parse(expr, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - Python syntax errors.
            raise InvalidExprError(str(exc)) from exc
        return tree

    def evaluate(self, expr: str, values: Dict[str, Any]) -> Fraction:
        tree = self.parse(expr)
        return self._eval_node(tree.body, values)

    def _eval_node(self, node: py_ast.AST, values: Dict[str, Any]) -> Fraction:
        if isinstance(node, py_ast.Constant):
            return Fraction(node.value)
        if isinstance(node, py_ast.Name):
            val = values.get(node.id)
            if val is None:
                raise EvaluationError(f"Symbol '{node.id}' not defined.")
            return Fraction(val)
        if isinstance(node, py_ast.BinOp):
            left = self._eval_node(node.left, values)
            right = self._eval_node(node.right, values)
            if isinstance(node.op, py_ast.Add):
                return left + right
            if isinstance(node.op, py_ast.Sub):
                return left - right
            if isinstance(node.op, py_ast.Mult):
                return left * right
            if isinstance(node.op, py_ast.Div):
                if right == 0:
                    raise InvalidExprError("Division by zero")
                return left / right
            if isinstance(node.op, py_ast.Pow):
                if right.denominator != 1:
                    raise InvalidExprError("Exponent must be an integer.")
                exponent = right.numerator
                return Fraction(left ** exponent)
        if isinstance(node, py_ast.UnaryOp):
            operand = self._eval_node(node.operand, values)
            if isinstance(node.op, py_ast.UAdd):
                return operand
            if isinstance(node.op, py_ast.USub):
                return -operand
        raise InvalidExprError(f"Unsupported expression: {py_ast.dump(node)}")

    def symbols(self, expr: str) -> Set[str]:
        tree = self.parse(expr)
        return {node.id for node in py_ast.walk(tree) if isinstance(node, py_ast.Name)}


@dataclass
class SymbolicEngine:
    """Thin wrapper providing equivalence and simplification utilities."""

    def __post_init__(self) -> None:
        self._fallback = _FallbackEvaluator() if _sympy is None else None

    def has_sympy(self) -> bool:
        return self._fallback is None

    def to_internal(self, expr: str) -> Any:
        if self._fallback is not None:
            return self._fallback.parse(expr)
        try:
            # Ensure 'e' is treated as Euler's number and 'pi' as pi
            local_dict = {"e": _sympy.E, "pi": _sympy.pi}
            return _sympy.sympify(expr, locals=local_dict)
        except Exception as exc:  # pragma: no cover - SymPy provides details.
            raise InvalidExprError(str(exc)) from exc

    def is_equiv(self, expr1: str, expr2: str) -> bool:
        if self._fallback is not None:
            return self._fallback_is_equiv(expr1, expr2)
        internal1 = self.to_internal(expr1)
        internal2 = self.to_internal(expr2)
        try:
            diff = _sympy.simplify(internal1 - internal2)
            if diff == 0:
                return True
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Failed to compare expressions: {exc}")
        return self._numeric_sampling_equiv(expr1, expr2)


    def _fallback_is_equiv(self, expr1: str, expr2: str) -> bool:
        assert self._fallback is not None
        symbols = self._fallback.symbols(expr1) | self._fallback.symbols(expr2)
        success = False
        for assignment in _SAMPLE_ASSIGNMENTS:
            subset = {name: assignment.get(name, 1) for name in symbols}
            try:
                left = self._fallback.evaluate(expr1, subset)
                right = self._fallback.evaluate(expr2, subset)
            except (InvalidExprError, EvaluationError):
                continue
            if left != right:
                return False
            success = True
        if not success:
            raise InvalidExprError("Unable to evaluate expressions for comparison.")
        return True

    def _numeric_sampling_equiv(self, expr1: str, expr2: str) -> bool:
        try:
            internal1 = self.to_internal(expr1)
            internal2 = self.to_internal(expr2)
        except InvalidExprError:
            raise
        success = False
        for assignment in _SAMPLE_ASSIGNMENTS:
            try:
                subs1 = {sym: assignment.get(str(sym), 1) for sym in internal1.free_symbols}
                subs2 = {sym: assignment.get(str(sym), 1) for sym in internal2.free_symbols}
                val1 = internal1.subs(subs1)
                val2 = internal2.subs(subs2)
                if val1.free_symbols or val2.free_symbols:
                    continue
                if val1.evalf() != val2.evalf():
                    return False
                success = True
            except Exception:
                continue
        return success

    def simplify(self, expr: str) -> str:
        if self._fallback is not None:
            try:
                return SimpleAlgebra.simplify(expr)
            except InvalidExprError:
                pass
            try:
                value = self._fallback.evaluate(expr, {})
                if value.denominator == 1:
                    return str(value.numerator)
                return f"{value.numerator}/{value.denominator}"
            except (InvalidExprError, EvaluationError):
                return expr
        internal = self.to_internal(expr)
        return str(_sympy.simplify(internal))

    def to_latex(self, expr: str) -> str:
        """
        Convert an expression to its LaTeX representation.
        
        Args:
            expr: The expression string to convert.
            
        Returns:
            LaTeX string representation.
        """
        if self._fallback is not None:
            # Fallback: basic conversion (e.g., * to \cdot, / to \frac)
            # This is a very simple approximation
            latex = expr.replace("*", r" \cdot ").replace("pi", r"\pi")
            # Handle simple fractions if possible, but regex is tricky.
            # For now, return the expression with minor tweaks.
            return latex
        
        try:
            internal = self.to_internal(expr)
            return _sympy.latex(internal)
        except Exception:
            return expr

    def evaluate(self, expr: str, context: Dict[str, Any]) -> Any:
        if self._fallback is not None:
            symbols = self._fallback.symbols(expr)
            if not context and symbols:
                return {"not_evaluatable": True}
            return self._fallback.evaluate(expr, context)

        internal_expr = self.to_internal(expr)

        free_symbols = internal_expr.free_symbols
        if not context and free_symbols:
            return {"not_evaluatable": True}
        undefined_symbols = [s for s in free_symbols if str(s) not in context]
        if undefined_symbols:
            raise EvaluationError(f"Undefined symbols: {', '.join(map(str, undefined_symbols))}")

        subs = {s: context.get(str(s)) for s in free_symbols}

        try:
            result = internal_expr.subs(subs)
            if not result.free_symbols:
                simplified = _sympy.simplify(result)
                return self._to_python_number(simplified)
            return {"not_evaluatable": True}
        except Exception as exc:
            raise EvaluationError(f"Failed to evaluate expression: {exc}")


    def evaluate_numeric(self, expr: str, assignment: Dict[str, int]) -> Any:
        if self._fallback is not None:
            return self._fallback.evaluate(expr, assignment)
        internal = self.to_internal(expr)
        subs = {symbol: assignment.get(str(symbol), 0) for symbol in internal.free_symbols}
        return internal.subs(subs)

    def explain(self, before: str, after: str) -> str:
        try:
            if self.is_equiv(before, after):
                return "Expressions are equivalent."
        except EvaluationError:
            pass # Fallback to simplification comparison
            
        simplified_before = self.simplify(before)
        simplified_after = self.simplify(after)
        hint = "sympy" if self._fallback is None else "numeric sampling"
        return f"Compared via {hint}: {simplified_before} → {simplified_after}."

    def _to_python_number(self, value: Any) -> Any:
        if _sympy is None:
            return value

        is_integer_attr = getattr(value, "is_integer", None)
        is_integer = False
        if callable(is_integer_attr):
            try:
                is_integer = bool(is_integer_attr())
            except Exception:
                is_integer = False
        elif is_integer_attr is not None:
            is_integer = bool(is_integer_attr)
        if is_integer:
            try:
                return int(value)
            except Exception:
                pass

        is_rational_attr = getattr(value, "is_rational", None)
        is_rational = False
        if callable(is_rational_attr):
            try:
                is_rational = bool(is_rational_attr())
            except Exception:
                is_rational = False
        elif is_rational_attr is not None:
            is_rational = bool(is_rational_attr)
        if is_rational:
            try:
                numer, denom = value.as_numer_denom()
                numer = int(numer)
                denom = int(denom)
                return Fraction(numer, denom)
            except Exception:
                pass

        try:
            return float(value)
        except Exception:
            return value
