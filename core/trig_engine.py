"""Trigonometric helpers (unit circle, identities, evaluation)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

try:  # pragma: no cover - optional dependency
    import sympy as _sympy
except Exception:  # pragma: no cover
    _sympy = None


class TrigHelper:
    """Provides small utilities for trigonometric reasoning."""

    _STANDARD_ANGLES: Dict[int, Tuple[float, float]] = {
        0: (0.0, 1.0),
        30: (0.5, math.sqrt(3) / 2),
        45: (math.sqrt(2) / 2, math.sqrt(2) / 2),
        60: (math.sqrt(3) / 2, 0.5),
        90: (1.0, 0.0),
        120: (math.sqrt(3) / 2, -0.5),
        135: (math.sqrt(2) / 2, -math.sqrt(2) / 2),
        150: (0.5, -math.sqrt(3) / 2),
        180: (0.0, -1.0),
        210: (-0.5, -math.sqrt(3) / 2),
        225: (-math.sqrt(2) / 2, -math.sqrt(2) / 2),
        240: (-math.sqrt(3) / 2, -0.5),
        270: (-1.0, 0.0),
        300: (-math.sqrt(3) / 2, 0.5),
        315: (-math.sqrt(2) / 2, math.sqrt(2) / 2),
        330: (-0.5, math.sqrt(3) / 2),
    }

    def to_radians(self, angle: float, unit: str = "radians") -> float:
        unit = unit.lower()
        if unit in ("degree", "degrees"):
            return math.radians(angle)
        return angle

    def unit_circle_point(self, angle: float, unit: str = "degrees") -> Dict[str, float]:
        radians = self.to_radians(angle, unit)
        sin_val = math.sin(radians)
        cos_val = math.cos(radians)
        return {
            "angle_degrees": math.degrees(radians),
            "angle_radians": radians,
            "sin": _round_if_close(sin_val),
            "cos": _round_if_close(cos_val),
        }

    def standard_angle(self, angle: int) -> Dict[str, float] | None:
        normalized = angle % 360
        cos_val, sin_val = self._STANDARD_ANGLES.get(normalized, (None, None))
        if cos_val is None:
            return None
        return {
            "angle_degrees": normalized,
            "sin": _round_if_close(sin_val),
            "cos": _round_if_close(cos_val),
        }

    def evaluate(self, func: str, angle: float, unit: str = "radians") -> float:
        radians = self.to_radians(angle, unit)
        name = func.lower()
        if _sympy is not None:
            sym_angle = _sympy.Float(radians)
            sym_func = getattr(_sympy, name, None)
            if sym_func is not None:
                return float(sym_func(sym_angle))
        trig_map = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sec": lambda x: 1 / math.cos(x),
            "csc": lambda x: 1 / math.sin(x),
            "cot": lambda x: 1 / math.tan(x),
        }
        if name not in trig_map:
            raise ValueError(f"Unsupported trig function: {func}")
        return trig_map[name](radians)

    def apply_identity(
        self,
        identity: str,
        angle: float,
        unit: str = "radians",
        other_angle: float | None = None,
        other_unit: str | None = None,
    ) -> float:
        radians = self.to_radians(angle, unit)
        secondary = None
        if other_angle is not None:
            secondary = self.to_radians(
                other_angle, other_unit or unit
            )
        if identity == "sin_double":
            return 2 * math.sin(radians) * math.cos(radians)
        if identity == "cos_double":
            return math.cos(2 * radians)
        if identity == "tan_double":
            denominator = 1 - math.tan(radians) ** 2
            if abs(denominator) < 1e-12:
                raise ValueError("tan double-angle undefined at this angle.")
            return (2 * math.tan(radians)) / denominator
        if identity == "sin_sum":
            if secondary is None:
                raise ValueError("sin_sum identity requires other_angle.")
            return math.sin(radians) * math.cos(secondary) + math.cos(radians) * math.sin(secondary)
        if identity == "cos_sum":
            if secondary is None:
                raise ValueError("cos_sum identity requires other_angle.")
            return math.cos(radians) * math.cos(secondary) - math.sin(radians) * math.sin(secondary)
        if identity == "pythagorean":
            return math.sin(radians) ** 2 + math.cos(radians) ** 2
        if identity == "tan_sum_zero":
            return math.tan(radians) + math.tan(math.pi - radians)
        raise ValueError(f"Unknown identity: {identity}")


def _round_if_close(value: float) -> float:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return float(rounded)
    return round(value, 6)
