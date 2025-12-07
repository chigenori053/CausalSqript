"""UnitEngine module for handling unit conversions and dimensional analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Optional

from causalscript.core.symbolic_engine import SymbolicEngine

try:  # pragma: no cover - optional dependency.
    import sympy.physics.units as units
    from sympy import Expr, sympify
    from sympy.physics.units.util import check_dimensions as sympy_check_dimensions
    from sympy import simplify as sympy_simplify
except Exception:  # pragma: no cover - SymPy not available.
    units = None
    Expr = Any  # type: ignore[assignment]
    sympify = None
    sympy_check_dimensions = None
    sympy_simplify = None


class UnitEngine:
    """
    Engine for performing unit conversions and dimensional analysis.
    Wraps sympy.physics.units.
    """

    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic_engine = symbolic_engine

    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic_engine = symbolic_engine
        self._basic_parser: _BasicUnitParser | None = None
        if units is None:
            self._basic_parser = _BasicUnitParser()

    def _parse_expr(self, expr: str) -> Any:
        """Parse string expression to SymPy expression with units."""
        if units is None:
            assert self._basic_parser is not None
            return self._basic_parser.parse(expr)

        unit_locals = {
            name: getattr(units, name)
            for name in dir(units)
            if not name.startswith("_")
        }
        return sympify(expr, locals=unit_locals)

    def convert(self, expr: str, target_unit: str) -> str:
        """
        Convert an expression to a target unit.

        Args:
            expr: The expression to convert (e.g., "10 * meter")
            target_unit: The target unit (e.g., "centimeter")

        Returns:
            The converted expression as a string.
        """
        if units is None:
            assert self._basic_parser is not None
            source = self._basic_parser.parse(expr)
            target = self._basic_parser.parse(target_unit)
            if source.dimensions != target.dimensions:
                raise ValueError("Incompatible units for conversion.")
            ratio = source.magnitude / target.magnitude
            normalized = _normalize_unit_expression(target_unit)
            return f"{_format_fraction(ratio)}*{normalized}"

        sym_expr = self._parse_expr(expr)
        sym_target = self._parse_expr(target_unit)
        converted = units.convert_to(sym_expr, sym_target)
        return str(converted)

    def simplify(self, expr: str) -> str:
        """
        Simplify a unit expression.

        Args:
            expr: The expression to simplify.

        Returns:
            The simplified expression as a string.
        """
        if units is None:
            assert self._basic_parser is not None
            quantity = self._basic_parser.parse(expr)
            return _format_quantity(quantity)

        sym_expr = self._parse_expr(expr)
        simplified = sympy_simplify(sym_expr)
        return str(simplified)

    def check_consistency(self, expr: str) -> bool:
        """
        Check if the dimensions in the expression are consistent.
        For example, '1*meter + 1*second' is inconsistent.

        Args:
            expr: The expression to check.

        Returns:
            True if consistent, False otherwise.
        """
        if units is None:
            assert self._basic_parser is not None
            try:
                self._basic_parser.parse(expr)
                return True
            except ValueError:
                return False

        try:
            sym_expr = self._parse_expr(expr)
            sympy_check_dimensions(sym_expr)
            return True
        except ValueError:
            return False

    def get_si_units(self, expr: str) -> str:
        """
        Convert the expression to SI base units.

        Args:
            expr: The expression to convert.

        Returns:
            The expression in SI units.
        """
        if units is None:
            assert self._basic_parser is not None
            quantity = self._basic_parser.parse(expr)
            unit_str = _format_dimensions(quantity.dimensions)
            if not unit_str:
                return _format_fraction(quantity.magnitude)
            return f"{_format_fraction(quantity.magnitude)}*{unit_str}"

        sym_expr = self._parse_expr(expr)
        base_units = [
            units.meter,
            units.kilogram,
            units.second,
            units.ampere,
            units.kelvin,
            units.mole,
            units.candela,
        ]
        converted = units.convert_to(sym_expr, base_units)
        return str(converted)


@dataclass
class _Quantity:
    magnitude: Fraction
    dimensions: Dict[str, int]

    def copy(self) -> "_Quantity":
        return _Quantity(self.magnitude, dict(self.dimensions))


class _BasicUnitParser:
    """Very small arithmetic parser to keep tests working without SymPy."""

    def parse(self, expr: str) -> _Quantity:
        tree = ast.parse(expr, mode="eval")
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> _Quantity:
        if isinstance(node, ast.Constant):
            return _Quantity(Fraction(node.value), {})
        if isinstance(node, ast.Name):
            return self._unit_quantity(node.id)
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return self._add(left, right)
            if isinstance(node.op, ast.Sub):
                return self._add(left, self._negate(right))
            if isinstance(node.op, ast.Mult):
                return self._mul(left, right)
            if isinstance(node.op, ast.Div):
                return self._div(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval(node.operand)
            return self._negate(value)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _unit_quantity(self, name: str) -> _Quantity:
        if name not in _UNIT_DEFINITIONS:
            raise ValueError(f"Unknown unit: {name}")
        base_unit, factor = _UNIT_DEFINITIONS[name]
        dims = {base_unit: 1}
        return _Quantity(factor, dims)

    def _add(self, left: _Quantity, right: _Quantity) -> _Quantity:
        if left.dimensions != right.dimensions:
            raise ValueError("Incompatible units for addition.")
        return _Quantity(left.magnitude + right.magnitude, dict(left.dimensions))

    def _negate(self, qty: _Quantity) -> _Quantity:
        return _Quantity(-qty.magnitude, dict(qty.dimensions))

    def _mul(self, left: _Quantity, right: _Quantity) -> _Quantity:
        dims = dict(left.dimensions)
        for unit, power in right.dimensions.items():
            dims[unit] = dims.get(unit, 0) + power
            if dims[unit] == 0:
                del dims[unit]
        return _Quantity(left.magnitude * right.magnitude, dims)

    def _div(self, left: _Quantity, right: _Quantity) -> _Quantity:
        dims = dict(left.dimensions)
        for unit, power in right.dimensions.items():
            dims[unit] = dims.get(unit, 0) - power
            if dims[unit] == 0:
                del dims[unit]
        if right.magnitude == 0:
            raise ValueError("Division by zero.")
        return _Quantity(left.magnitude / right.magnitude, dims)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_dimensions(dimensions: Dict[str, int]) -> str:
    if not dimensions:
        return ""
    positives = []
    negatives = []
    for unit in sorted(dimensions):
        power = dimensions[unit]
        if power > 0:
            positives.append(_format_unit_power(unit, power))
        elif power < 0:
            negatives.append(_format_unit_power(unit, -power))
    numerator = "*".join(positives) if positives else "1"
    if not negatives:
        return numerator if numerator != "1" else ""
    denominator = "*".join(negatives)
    if numerator == "1":
        return f"1/{denominator}"
    return f"{numerator}/{denominator}"


def _format_unit_power(unit: str, power: int) -> str:
    if power == 1:
        return unit
    return f"{unit}**{power}"


def _normalize_unit_expression(expr: str) -> str:
    return "".join(expr.split())


def _format_quantity(qty: _Quantity) -> str:
    unit_str = _format_dimensions(qty.dimensions)
    if not unit_str:
        return _format_fraction(qty.magnitude)
    return f"{_format_fraction(qty.magnitude)}*{unit_str}"


_UNIT_DEFINITIONS: Dict[str, tuple[str, Fraction]] = {
    "meter": ("meter", Fraction(1)),
    "centimeter": ("meter", Fraction(1, 100)),
    "millimeter": ("meter", Fraction(1, 1000)),
    "kilometer": ("meter", Fraction(1000)),
    "second": ("second", Fraction(1)),
    "minute": ("second", Fraction(60)),
    "hour": ("second", Fraction(3600)),
    "kilogram": ("kilogram", Fraction(1)),
    "gram": ("kilogram", Fraction(1, 1000)),
}


def get_common_units() -> Dict[str, Any]:
    """
    Retrieve a dictionary of common units for context injection.
    
    Returns:
        Dict mapping unit names (e.g., 'cm', 'kg') to SymPy unit objects.
    """
    if units is None:
        return {}
        
    # List of common units to expose
    # Must match core.input_parser.CausalScriptInputParser._KNOWN_UNITS
    common_names = {
        "cm", "mm", "km", "kg", "g", "mg", "m", "s", "h", "Hz", "N", "J", "W", "Pa",
        "min", "hr", "deg", "rad", "liter", "L"
    }
    
    unit_map = {}
    for name in common_names:
        if hasattr(units, name):
            unit_map[name] = getattr(units, name)
            
    return unit_map
