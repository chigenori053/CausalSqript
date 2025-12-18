"""多項式演算のためのユーティリティ."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Tuple, Union

Monomial = Tuple[Tuple[str, int], ...]


def _normalize_monomial(items: Iterable[Tuple[str, int]]) -> Monomial:
    filtered = [(name, power) for name, power in items if power != 0]
    return tuple(sorted(filtered))


@dataclass(frozen=True)
class Polynomial:
    """多項式を {単項式: 係数} の辞書で表現する."""

    terms: Dict[Monomial, Fraction]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            {
                monomial: coeff
                for monomial, coeff in self.terms.items()
                if coeff != 0
            },
        )

    @staticmethod
    def zero() -> "Polynomial":
        return Polynomial({})

    @staticmethod
    def constant(value: Union[Fraction, int, float]) -> "Polynomial":
        coeff = _to_fraction(value)
        if coeff == 0:
            return Polynomial.zero()
        return Polynomial({(): coeff})

    @staticmethod
    def variable(name: str) -> "Polynomial":
        return Polynomial({((name, 1),): Fraction(1)})

    def __bool__(self) -> bool:  # noqa: D401
        return bool(self.terms)

    def __add__(self, other: "Polynomial") -> "Polynomial":
        new_terms: Dict[Monomial, Fraction] = dict(self.terms)
        for monomial, coeff in other.terms.items():
            new_terms[monomial] = new_terms.get(monomial, Fraction(0)) + coeff
            if new_terms[monomial] == 0:
                del new_terms[monomial]
        return Polynomial(new_terms)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self + (-other)

    def __neg__(self) -> "Polynomial":
        return Polynomial({mono: -coeff for mono, coeff in self.terms.items()})

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        if not self or not other:
            return Polynomial.zero()

        new_terms: Dict[Monomial, Fraction] = {}
        for mono_a, coeff_a in self.terms.items():
            for mono_b, coeff_b in other.terms.items():
                combined: Dict[str, int] = {}
                for name, power in mono_a:
                    combined[name] = combined.get(name, 0) + power
                for name, power in mono_b:
                    combined[name] = combined.get(name, 0) + power
                monomial = _normalize_monomial(combined.items())
                new_terms[monomial] = new_terms.get(monomial, Fraction(0)) + (coeff_a * coeff_b)
                if new_terms[monomial] == 0:
                    del new_terms[monomial]

        return Polynomial(new_terms)

    def pow(self, exponent: int) -> "Polynomial":
        if exponent < 0:
            raise ValueError("Polynomial exponent must be non-negative")
        if exponent == 0:
            return Polynomial.constant(1)
        result = Polynomial(dict(self.terms))
        for _ in range(exponent - 1):
            result = result * self
        return result

    def divide_by_scalar(self, scalar: Union[Fraction, int, float]) -> "Polynomial":
        scalar_fraction = _to_fraction(scalar)
        if scalar_fraction == 0:
            raise ZeroDivisionError("Division by zero polynomial scalar")
        return Polynomial({mono: coeff / scalar_fraction for mono, coeff in self.terms.items()})

    def is_zero(self) -> bool:
        return not self.terms

    def is_constant(self) -> bool:
        if not self.terms:
            return True
        return all(not monomial for monomial in self.terms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.terms == other.terms

    def __str__(self) -> str:
        if not self.terms:
            return "0"

        def sort_key(item: Tuple[Monomial, Fraction]) -> Tuple[int, Tuple[str, ...]]:
            monomial, _ = item
            total_degree = sum(power for _, power in monomial)
            variables = tuple(name for name, _ in monomial)
            return (-total_degree, variables, monomial)

        parts = []
        for monomial, coeff in sorted(self.terms.items(), key=sort_key):
            coeff_abs = abs(coeff)
            monomial_str = self._format_monomial(monomial)
            coeff_part = self._format_coefficient(coeff_abs, bool(monomial_str))

            if monomial_str:
                term = coeff_part + (("*" if coeff_part else "") + monomial_str)
            else:
                term = coeff_part or "1"

            if coeff < 0:
                parts.append(f"- {term}" if parts else f"-{term}")
            else:
                parts.append(term if not parts else f"+ {term}")

        return " ".join(parts)

    @staticmethod
    def _format_monomial(monomial: Monomial) -> str:
        if not monomial:
            return ""
        factors = []
        for name, power in monomial:
            if power == 1:
                factors.append(name)
            else:
                factors.append(f"{name}^{power}")
        return "*".join(factors)

    @staticmethod
    def _format_coefficient(coeff: Fraction, has_variable: bool) -> str:
        if coeff == 0:
            return "0"
        if coeff == 1:
            return "" if has_variable else "1"
        if coeff.denominator == 1:
            return str(coeff.numerator)
        return f"{coeff.numerator}/{coeff.denominator}"


def _to_fraction(value: Union[Fraction, int, float]) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise TypeError(f"Unsupported numeric type for polynomial: {type(value)!r}")

