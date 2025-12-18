"""Utility helpers for analyzing single-variable mathematical functions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .computation_engine import ComputationEngine
from .errors import EvaluationError, InvalidExprError

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore

NumericSample = Tuple[float, float]


@dataclass
class FunctionAnalysisResult:
    expression: str
    variable: str
    domain: Dict[str, Any]
    range: Dict[str, Any]
    intercepts: Dict[str, Any]
    behavior: Dict[str, Any]
    samples: int


class FunctionAnalyzer:
    """Provides lightweight function analytics and plotting helpers."""

    def __init__(
        self,
        computation_engine: ComputationEngine,
        sample_window: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        self.engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
        self.sample_window = sample_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        expr: str,
        variable: str = "x",
        sample_step: float = 1.0,
    ) -> FunctionAnalysisResult:
        """Analyze the supplied expression and return structured insights."""

        samples = self._sample_values(expr, variable, sample_step)
        domain = self._infer_domain(expr, variable, samples)
        intercepts = self._calculate_intercepts(expr, variable)
        range_info = self._calculate_range(samples)
        behavior = self._estimate_behavior(samples)

        return FunctionAnalysisResult(
            expression=expr,
            variable=variable,
            domain=domain,
            range=range_info,
            intercepts=intercepts,
            behavior=behavior,
            samples=len(samples),
        )

    def generate_plot_data(
        self,
        expr: str,
        variable: str = "x",
        start: float | None = None,
        end: float | None = None,
        num_points: int = 100,
    ) -> Dict[str, List[Optional[float]]]:
        """Return sample points suitable for plotting."""
        start = self.sample_window[0] if start is None else start
        end = self.sample_window[1] if end is None else end
        if num_points < 2:
            num_points = 2
        step = (end - start) / (num_points - 1)
        xs = [start + i * step for i in range(num_points)]
        ys: List[Optional[float]] = []
        for x_val in xs:
            ys.append(self._evaluate(expr, variable, x_val))
        return {"x": xs, "y": ys}

    def plot(
        self,
        expr: str,
        variable: str = "x",
        start: float | None = None,
        end: float | None = None,
        num_points: int = 100,
    ) -> Dict[str, Any]:
        """Generate plot data and optionally a matplotlib figure."""
        data = self.generate_plot_data(expr, variable, start, end, num_points)

        figure = None
        if plt is not None:
            valid_points = [
                (x, y) for x, y in zip(data["x"], data["y"]) if y is not None
            ]
            if len(valid_points) >= 2:
                xs, ys = zip(*valid_points)
                figure, ax = plt.subplots()  # type: ignore[assignment]
                ax.plot(xs, ys)
                ax.set_xlabel(variable)
                ax.set_ylabel(f"f({variable})")
                ax.set_title(f"{expr}")
        return {
            "x": data["x"],
            "y": data["y"],
            "figure": figure,
            "matplotlib": figure is not None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_values(
        self,
        expr: str,
        variable: str,
        sample_step: float,
    ) -> List[NumericSample]:
        start, end = self.sample_window
        total_steps = int(((end - start) / sample_step)) + 1
        if total_steps <= 1:
            total_steps = 2
        values: List[NumericSample] = []
        for idx in range(total_steps):
            x_val = start + idx * sample_step
            y_val = self._evaluate(expr, variable, x_val)
            if y_val is not None:
                values.append((x_val, y_val))
        return values

    def _evaluate(self, expr: str, variable: str, value: float) -> Optional[float]:
        try:
            result = self.engine.numeric_eval(expr, {variable: value})
        except EvaluationError:
            return None
        if isinstance(result, dict):
            return None
        return _to_float(result)

    def _infer_domain(
        self,
        expr: str,
        variable: str,
        samples: Sequence[NumericSample],
    ) -> Dict[str, Any]:
        denominators = self._collect_denominators(expr)
        restrictions: List[Dict[str, Any]] = []
        if denominators:
            candidate_values = self._candidate_points(samples, variable)
            for denom in denominators:
                zeros = self._find_zeros(denom, variable, candidate_values)
                if zeros:
                    restrictions.append(
                        {
                            "type": "denominator_zero",
                            "expression": denom,
                            "values": zeros,
                        }
                    )
        domain_type = "all_real" if not restrictions else "restricted"
        return {"type": domain_type, "restrictions": restrictions}

    def _candidate_points(
        self, samples: Sequence[NumericSample], variable: str
    ) -> Iterable[float]:
        seen = {self.sample_window[0], self.sample_window[1]}
        for x, _ in samples:
            seen.add(x)
        # add integers to improve coverage
        for val in range(-20, 21):
            seen.add(float(val))
        return sorted(seen)

    def _find_zeros(
        self, expr: str, variable: str, candidates: Iterable[float]
    ) -> List[float]:
        zeros: List[float] = []
        for value in candidates:
            y_val = self._evaluate(expr, variable, value)
            if y_val is None:
                continue
            if abs(y_val) < 1e-8:
                rounded = round(value, 8)
                if rounded not in zeros:
                    zeros.append(rounded)
        return zeros

    def _collect_denominators(self, expr: str) -> List[str]:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise InvalidExprError(str(exc)) from exc
        denominators: List[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_BinOp(self, node: ast.BinOp) -> None:
                if isinstance(node.op, ast.Div):
                    denominators.append(ast.unparse(node.right))
                self.generic_visit(node)

        Visitor().visit(tree)
        return denominators

    def _calculate_intercepts(
        self,
        expr: str,
        variable: str,
    ) -> Dict[str, Any]:
        y_intercept = self._evaluate(expr, variable, 0.0)
        x_intercepts: List[float] = []
        samples = self._sample_values(expr, variable, sample_step=0.5)
        for x, y in samples:
            if abs(y) < 1e-8:
                approx = round(x, 6)
                if approx not in x_intercepts:
                    x_intercepts.append(approx)
        # detect sign changes to capture zero crossings
        for (x1, y1), (x2, y2) in zip(samples, samples[1:]):
            if y1 is None or y2 is None:
                continue
            if y1 == 0 or y2 == 0:
                continue
            if y1 * y2 < 0:
                mid = round((x1 + x2) / 2.0, 6)
                if mid not in x_intercepts:
                    x_intercepts.append(mid)
        return {"x": sorted(x_intercepts), "y": y_intercept}

    def _calculate_range(self, samples: Sequence[NumericSample]) -> Dict[str, Any]:
        if not samples:
            return {"min": None, "max": None, "sample_size": 0}
        min_point = min(samples, key=lambda item: item[1])
        max_point = max(samples, key=lambda item: item[1])
        return {
            "min": {"x": min_point[0], "y": min_point[1]},
            "max": {"x": max_point[0], "y": max_point[1]},
            "sample_size": len(samples),
        }

    def _estimate_behavior(
        self, samples: Sequence[NumericSample]
    ) -> Dict[str, Any]:
        if len(samples) < 2:
            return {"trend": "unknown", "critical_points": []}
        slopes: List[float] = []
        critical_points: List[Dict[str, float]] = []
        previous_slope: Optional[float] = None

        for (x1, y1), (x2, y2) in zip(samples, samples[1:]):
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            slopes.append(slope)
            if previous_slope is not None:
                if previous_slope > 0 and slope < 0:
                    critical_points.append({"type": "local_max", "x": x1, "y": y1})
                elif previous_slope < 0 and slope > 0:
                    critical_points.append({"type": "local_min", "x": x1, "y": y1})
            previous_slope = slope

        increasing = any(s > 1e-6 for s in slopes)
        decreasing = any(s < -1e-6 for s in slopes)
        if increasing and not decreasing:
            trend = "increasing"
        elif decreasing and not increasing:
            trend = "decreasing"
        else:
            trend = "mixed"
        return {"trend": trend, "critical_points": critical_points}


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
