"""Fraction Engine for symbolic fraction manipulation."""

from __future__ import annotations
import math

from . import ast_nodes as ast
from .arithmetic_engine import ArithmeticEngine

class FractionEngine:
    """Handles symbolic manipulation of fractions, represented as Div nodes."""

    def __init__(self):
        self._arithmetic_engine = ArithmeticEngine()

    def normalize(self, expression: ast.Expr) -> ast.Expr:
        """
        Normalizes a fraction expression.
        - Reduces fractions by finding the greatest common divisor (GCD) for integers.
        - Reduces fractions by canceling common symbolic factors.
        - Normalizes signs (e.g., a/-b -> -a/b).
        - Recursively normalizes numerator and denominator.
        """
        if isinstance(expression, ast.Div):
            numerator_expr = expression.left
            denominator_expr = expression.right
        elif isinstance(expression, ast.RationalNode):
            numerator_expr = expression.numerator
            denominator_expr = expression.denominator
        else:
            return self._arithmetic_engine.normalize(expression)

        num = self.normalize(numerator_expr)
        den = self.normalize(denominator_expr)

        # Sign normalization
        if isinstance(den, ast.Neg):
            den = den.expr
            num = ast.Neg(expr=num)
        if isinstance(den, ast.Int) and den.value < 0:
            den = ast.Int(value=-den.value)
            num = ast.Neg(expr=num)
        num = self._arithmetic_engine.normalize(num)

        # Denominator of 1 simplification
        if isinstance(den, ast.Int) and den.value == 1:
            return num

        # Integer reduction
        if isinstance(num, ast.Int) and isinstance(den, ast.Int):
            if den.value == 0:
                return ast.Div(num, den)  # Cannot simplify division by zero
            common_divisor = self._gcd(num.value, den.value)
            new_num = num.value // common_divisor
            new_den = den.value // common_divisor
            if new_den == 1:
                return ast.Int(new_num)
            return ast.Div(ast.Int(new_num), ast.Int(new_den))

        # Symbolic reduction
        num_factors = list(self._get_factors(num))
        den_factors = list(self._get_factors(den))

        if num_factors and den_factors:
            remaining_den_factors = den_factors.copy()
            reduced_num_factors: list[ast.Expr] = []
            for factor in num_factors:
                if factor in remaining_den_factors:
                    remaining_den_factors.remove(factor)
                else:
                    reduced_num_factors.append(factor)

            if len(reduced_num_factors) != len(num_factors) or len(remaining_den_factors) != len(den_factors):
                num = self._reconstruct_mul(reduced_num_factors)
                den = self._reconstruct_mul(remaining_den_factors)

        # After symbolic reduction, try integer reduction again
        if isinstance(num, ast.Int) and isinstance(den, ast.Int):
            if den.value == 0:
                return ast.Div(num, den)
            common_divisor = self._gcd(num.value, den.value)
            new_num_val = num.value // common_divisor
            new_den_val = den.value // common_divisor
            if new_den_val == 1:
                return ast.Int(new_num_val)
            return ast.Div(ast.Int(new_num_val), ast.Int(new_den_val))
            
        # After reduction, if denominator is 1, return numerator
        if isinstance(den, ast.Int) and den.value == 1:
            return num

        return ast.Div(num, den)

    def add(self, left: ast.Expr, right: ast.Expr) -> ast.Expr:
        """Adds two expressions, handling fractions."""
        a, b = self._get_num_den(left)
        c, d = self._get_num_den(right)

        # (a/b) + (c/d) = (ad + bc) / bd
        new_num = ast.Add(terms=[ast.Mul(factors=[a, d]), ast.Mul(factors=[b, c])])
        new_den = ast.Mul(factors=[b, d])
        
        return self.normalize(ast.Div(new_num, new_den))

    def multiply(self, left: ast.Expr, right: ast.Expr) -> ast.Expr:
        """Multiplies two expressions, handling fractions."""
        a, b = self._get_num_den(left)
        c, d = self._get_num_den(right)

        # (a/b) * (c/d) = ac / bd
        new_num = ast.Mul(factors=[a, c])
        new_den = ast.Mul(factors=[b, d])

        return self.normalize(ast.Div(new_num, new_den))

    def divide(self, left: ast.Expr, right: ast.Expr) -> ast.Expr:
        """Divides two expressions, handling fractions."""
        a, b = self._get_num_den(left)
        c, d = self._get_num_den(right)

        # (a/b) / (c/d) = ad / bc
        new_num = ast.Mul(factors=[a, d])
        new_den = ast.Mul(factors=[b, c])

        return self.normalize(ast.Div(new_num, new_den))

    def _get_num_den(self, expr: ast.Expr) -> tuple[ast.Expr, ast.Expr]:
        """Returns the numerator and denominator of an expression."""
        if isinstance(expr, ast.Div):
            return expr.left, expr.right
        if isinstance(expr, ast.RationalNode):
            return expr.numerator, expr.denominator
        return expr, ast.Int(1)

    def _get_factors(self, expr: ast.Expr) -> List[ast.Expr]:
        """Returns a list of factors for an expression."""
        if isinstance(expr, ast.Mul):
            return expr.factors
        return [expr]

    def _reconstruct_mul(self, factors: List[ast.Expr]) -> ast.Expr:
        """Reconstructs an expression from a list of factors."""
        if not factors:
            return ast.Int(1)
        if len(factors) == 1:
            return factors[0]
        return ast.Mul(factors=factors)

    def _gcd(self, a: int, b: int) -> int:
        """Computes the greatest common divisor of two integers."""
        return math.gcd(a, b)
