"""Lightweight polynomial algebra helpers used when SymPy is unavailable."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, Tuple

from .errors import InvalidExprError

MonomialKey = Tuple[Tuple[str, int], ...]


class SimpleAlgebra:
    """Tiny algebra helper that supports a subset of SymPy operations."""

    @staticmethod
    def simplify(expr: str) -> str:
        poly = _Polynomial.from_expr(expr)
        return poly.to_string()

    @staticmethod
    def expand(expr: str) -> str:
        # Expansion falls out of polynomial construction.
        return SimpleAlgebra.simplify(expr)

    @staticmethod
    def substitute(expr: str, substitutions: Dict[str, Any]) -> str:
        poly = _Polynomial.from_expr(expr)
        substituted = poly.substitute(substitutions)
        return substituted.to_string()

    @staticmethod
    def factor(expr: str) -> str:
        diff = _factor_difference_of_squares(expr)
        if diff:
            return diff
        poly = _Polynomial.from_expr(expr)
        square = _factor_perfect_square(poly)
        if square:
            return square
        return poly.to_string()

    @staticmethod
    def derivative(expr: str, variable: str) -> str:
        poly = _Polynomial.from_expr(expr)
        derived = poly.derivative(variable)
        return derived.to_string()

    @staticmethod
    def integral(expr: str, variable: str) -> str:
        poly = _Polynomial.from_expr(expr)
        integrated = poly.integral(variable)
        return integrated.to_string()


@dataclass(frozen=True)
class _Polynomial:
    terms: Dict[MonomialKey, Fraction]

    def __post_init__(self) -> None:
        cleaned: Dict[MonomialKey, Fraction] = {}
        for key, coeff in self.terms.items():
            if coeff == 0:
                continue
            cleaned[key] = coeff
        object.__setattr__(self, "terms", cleaned)

    @classmethod
    def from_expr(cls, expr: str) -> _Polynomial:
        try:
            node = ast.parse(expr, mode="eval").body
        except SyntaxError as exc:
            raise InvalidExprError(f"Invalid expression: {expr}") from exc
        return cls._from_ast(node)

    @classmethod
    def _from_ast(cls, node: ast.AST) -> _Polynomial:
        if isinstance(node, ast.Constant):
            return cls({(): _to_fraction(node.value)})
        if isinstance(node, ast.Name):
            key = ((node.id, 1),)
            return cls({key: Fraction(1)})
        if isinstance(node, ast.BinOp):
            left = cls._from_ast(node.left)
            right = cls._from_ast(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Pow):
                exponent = cls._extract_exponent(node.right)
                return left.pow(exponent)
        if isinstance(node, ast.UnaryOp):
            value = cls._from_ast(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        raise InvalidExprError(f"Unsupported expression: {ast.dump(node)}")

    @staticmethod
    def _extract_exponent(node: ast.AST) -> int:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, int):
                if value < 0:
                    raise InvalidExprError("Negative exponents are not supported.")
                return value
        raise InvalidExprError("Exponent must be a non-negative integer.")

    def __add__(self, other: _Polynomial) -> _Polynomial:
        terms = dict(self.terms)
        for key, coeff in other.terms.items():
            terms[key] = terms.get(key, Fraction(0)) + coeff
            if terms[key] == 0:
                del terms[key]
        return _Polynomial(terms)

    def __sub__(self, other: _Polynomial) -> _Polynomial:
        return self + (-other)

    def __mul__(self, other: _Polynomial) -> _Polynomial:
        terms: Dict[MonomialKey, Fraction] = {}
        for key_a, coeff_a in self.terms.items():
            for key_b, coeff_b in other.terms.items():
                merged_key = _merge_keys(key_a, key_b)
                terms[merged_key] = terms.get(merged_key, Fraction(0)) + coeff_a * coeff_b
                if terms[merged_key] == 0:
                    del terms[merged_key]
        return _Polynomial(terms)

    def __neg__(self) -> _Polynomial:
        return _Polynomial({key: -coeff for key, coeff in self.terms.items()})

    def pow(self, exponent: int) -> _Polynomial:
        if exponent == 0:
            return _Polynomial({(): Fraction(1)})
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def substitute(self, substitutions: Dict[str, Any]) -> _Polynomial:
        terms: Dict[MonomialKey, Fraction] = {}
        for key, coeff in self.terms.items():
            new_coeff = coeff
            remaining: Dict[str, int] = {}
            for var, power in key:
                if var in substitutions:
                    value = _to_fraction(substitutions[var])
                    new_coeff *= value ** power
                else:
                    remaining[var] = remaining.get(var, 0) + power
            new_key = tuple(sorted((var, power) for var, power in remaining.items()))
            terms[new_key] = terms.get(new_key, Fraction(0)) + new_coeff
            if terms[new_key] == 0:
                del terms[new_key]
        return _Polynomial(terms)

    def derivative(self, variable: str) -> _Polynomial:
        terms: Dict[MonomialKey, Fraction] = {}
        for key, coeff in self.terms.items():
            new_coeff = coeff
            found = False
            new_key_parts: Dict[str, int] = {}
            for var, power in key:
                if var == variable:
                    if power == 0:
                        continue
                    found = True
                    new_coeff *= power
                    if power - 1 > 0:
                        new_key_parts[var] = power - 1
                else:
                    new_key_parts[var] = power
            if not found:
                continue
            new_key = tuple(sorted(new_key_parts.items()))
            terms[new_key] = terms.get(new_key, Fraction(0)) + new_coeff
            if terms[new_key] == 0:
                del terms[new_key]
        return _Polynomial(terms)

    def integral(self, variable: str) -> _Polynomial:
        terms: Dict[MonomialKey, Fraction] = {}
        for key, coeff in self.terms.items():
            new_key_parts: Dict[str, int] = {}
            added = False
            for var, power in key:
                if var == variable:
                    new_power = power + 1
                    new_key_parts[var] = new_power
                    new_coeff = coeff / Fraction(new_power)
                    added = True
                else:
                    new_key_parts[var] = power
            if not added:
                new_key_parts[variable] = 1
                new_coeff = coeff
            new_key = tuple(sorted(new_key_parts.items()))
            terms[new_key] = terms.get(new_key, Fraction(0)) + new_coeff
            if terms[new_key] == 0:
                del terms[new_key]
        return _Polynomial(terms)

    def coefficient(self, key: MonomialKey) -> Fraction:
        return self.terms.get(key, Fraction(0))

    def variables(self) -> Iterable[str]:
        for key in self.terms:
            for var, _ in key:
                yield var

    def to_string(self) -> str:
        if not self.terms:
            return "0"
        sorted_terms = sorted(
            self.terms.items(), key=lambda item: (-_total_degree(item[0]), item[0])
        )
        parts = []
        for idx, (key, coeff) in enumerate(sorted_terms):
            term = _format_term(abs(coeff), key)
            if idx == 0:
                parts.append(term if coeff >= 0 else f"-{term}")
            else:
                op = "+" if coeff >= 0 else "-"
                parts.append(f"{op} {term}")
        return " ".join(parts)


def _total_degree(key: MonomialKey) -> int:
    return sum(power for _, power in key)


def _merge_keys(key_a: MonomialKey, key_b: MonomialKey) -> MonomialKey:
    powers: Dict[str, int] = {}
    for var, power in key_a:
        powers[var] = powers.get(var, 0) + power
    for var, power in key_b:
        powers[var] = powers.get(var, 0) + power
    return tuple(sorted((var, power) for var, power in powers.items()))


def _format_term(coeff: Fraction, key: MonomialKey) -> str:
    if not key:
        return _format_fraction(coeff)
    var_parts = [
        var if power == 1 else f"{var}**{power}"
        for var, power in key
    ]
    vars_str = "*".join(var_parts)
    if coeff == 1:
        return vars_str
    return f"{_format_fraction(coeff)}*{vars_str}"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _to_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, (int, float)):
        return Fraction(value)
    raise InvalidExprError(f"Unsupported numeric value: {value}")


def _factor_difference_of_squares(expr: str) -> str | None:
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
        return None
    left_base = _square_base(node.left)
    right_base = _square_base(node.right)
    if not left_base or not right_base:
        return None
    return f"({left_base} - {right_base})*({left_base} + {right_base})"


def _square_base(node: ast.AST) -> str | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            if node.right.value == 2:
                return ast.unparse(node.left)
        return None
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                return None
            number = int(value)
            if number < 0:
                return None
            root = math.isqrt(number)
            if root * root == number:
                return str(root)
    return None


def _factor_perfect_square(poly: _Polynomial) -> str | None:
    variables = sorted(set(poly.variables()))
    if len(variables) != 1:
        return None
    var = variables[0]
    coeff_x2 = poly.coefficient(((var, 2),))
    if coeff_x2 != 1:
        return None
    coeff_x = poly.coefficient(((var, 1),))
    constant = poly.coefficient(())
    offset = coeff_x / 2
    if constant != offset * offset:
        return None
    offset_str = _format_fraction(abs(offset))
    if offset == 0:
        return f"({var})**2"
    sign = "+" if offset > 0 else "-"
    return f"({var} {sign} {offset_str})**2"
