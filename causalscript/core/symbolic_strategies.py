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
    
    def is_equiv(self, expr1: str, expr2: str, engine: Any) -> Optional[bool]:
        return None

    def simplify(self, expr: str, engine: Any) -> Optional[str]:
        return None

    def to_latex(self, expr: str, engine: Any) -> Optional[str]:
        return None

    def solve_system(self, expr: str, engine: Any) -> Any:
        """Solve a system of equations."""
        try:
            internal = engine.to_internal(expr)
            
            if self._fallback is not None:
                # Fallback implementation using LinearAlgebraEngine
                # We need to import here to avoid circular imports if any, 
                # but LinearAlgebraEngine is in core.linear_algebra_engine
                from .linear_algebra_engine import LinearAlgebraEngine
                from .simple_algebra import _Polynomial
                import ast as py_ast
                
                la_engine = LinearAlgebraEngine()
                
                # Extract equations from AST
                if not (isinstance(internal, py_ast.Expression) and 
                        isinstance(internal.body, py_ast.Call) and 
                        isinstance(internal.body.func, py_ast.Name) and 
                        internal.body.func.id == "System"):
                    return None
                
                equations = []
                variables = set()
                
                for arg in internal.body.args:
                    # Expect Eq(lhs, rhs)
                    if not (isinstance(arg, py_ast.Call) and 
                            isinstance(arg.func, py_ast.Name) and 
                            arg.func.id == "Eq"):
                        continue
                    
                    # Convert AST back to string for SimpleAlgebra
                    try:
                        lhs_str = py_ast.unparse(arg.args[0])
                        rhs_str = py_ast.unparse(arg.args[1])
                    except AttributeError:
                        return None
                        
                    eq_expr = f"{lhs_str} - ({rhs_str})"
                    poly = _Polynomial.from_expr(eq_expr)
                    
                    # Check linearity
                    for key in poly.terms:
                        degree = sum(power for _, power in key)
                        if degree > 1:
                            return None # Non-linear
                    
                    equations.append(poly)
                    variables.update(poly.variables())
                
                sorted_vars = sorted(list(variables))
                if not sorted_vars:
                    return None
                    
                # Build matrix
                matrix = []
                constants = []
                
                for poly in equations:
                    row = []
                    for var in sorted_vars:
                        coeff = poly.coefficient(((var, 1),))
                        row.append(float(coeff))
                    
                    # Constant term is on LHS, so move to RHS -> -constant
                    const_val = poly.coefficient(())
                    constants.append(float(-const_val))
                    matrix.append(row)
                
                try:
                    solution_vals = la_engine.solve_linear_system(matrix, constants)
                    solution = {var: val for var, val in zip(sorted_vars, solution_vals)}
                    return [solution] # Return list of dicts
                except Exception:
                    return None

            # SymPy implementation
            if isinstance(internal, _sympy.FiniteSet):
                system_args = list(internal)
            else:
                system_args = internal
                
            solutions = _sympy.solve(system_args)
            return solutions
        except Exception:
            return None

    def check_implication(self, system_expr: str, step_expr: str, engine: Any) -> bool:
        """Check if system_expr implies step_expr."""
        try:
            # 1. Solve system
            solutions = self.solve_system(system_expr, engine)
            
            if not solutions:
                return False 
            
            # Standardize solutions to a list of dicts
            if isinstance(solutions, dict):
                solutions = [solutions]
            elif isinstance(solutions, list):
                pass
            else:
                return False

            # 2. Check step
            if self._fallback is not None:
                from .simple_algebra import _Polynomial
                import ast as py_ast
                
                # Fallback check
                step_internal = engine.to_internal(step_expr)
                
                step_equations = []
                if isinstance(step_internal, py_ast.Expression):
                    body = step_internal.body
                    if isinstance(body, py_ast.Call) and isinstance(body.func, py_ast.Name):
                        if body.func.id == "Eq":
                            step_equations.append(step_internal)
                        elif body.func.id == "System":
                            # Extract equations from System
                            for arg in body.args:
                                if isinstance(arg, py_ast.Call) and isinstance(arg.func, py_ast.Name) and arg.func.id == "Eq":
                                    step_equations.append(arg)
                
                if not step_equations:
                    return False
                
                for eq_node in step_equations:
                    # Handle both Expression(body=Call) and Call
                    if isinstance(eq_node, py_ast.Expression):
                        call_node = eq_node.body
                    else:
                        call_node = eq_node
                        
                    lhs_str = py_ast.unparse(call_node.args[0])
                    rhs_str = py_ast.unparse(call_node.args[1])
                    eq_expr = f"{lhs_str} - ({rhs_str})"
                    poly = _Polynomial.from_expr(eq_expr)
                    
                    for sol in solutions:
                        # Substitute
                        val = poly.substitute(sol)
                        const_val = val.coefficient(())
                        if abs(float(const_val)) > 1e-9:
                            return False
                return True

            # SymPy check
            step_internal = engine.to_internal(step_expr)
            
            # Handle System step in SymPy
            if isinstance(step_internal, _sympy.FiniteSet):
                step_exprs = list(step_internal)
            else:
                step_exprs = [step_internal]
            
            for expr in step_exprs:
                for sol in solutions:
                    check = expr.subs(sol)
                    if hasattr(check, 'simplify'):
                        check = check.simplify()
                    
                    if check is True or check == True:
                        continue
                    if isinstance(check, _sympy.Eq):
                        if check.lhs == check.rhs:
                            continue
                    
                    return False
                
            return True
        except Exception:
            return False


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
