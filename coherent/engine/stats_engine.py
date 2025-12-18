"""Statistics and probability utilities for Coherent Core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


@dataclass
class DistributionResult:
    distribution: str
    value: float
    params: Dict[str, Any]
    pdf: Optional[float]
    cdf: Optional[float]


class StatsEngine:
    """Provides descriptive statistics, probability distribution helpers, and basic visualization."""

    def describe(self, data: Sequence[float]) -> Dict[str, Any]:
        if not data:
            raise ValueError("Data sequence must not be empty.")
        values = [float(x) for x in data]
        sorted_values = sorted(values)
        count = len(values)
        mean = sum(values) / count

        median = self._median(sorted_values)
        variance = (
            sum((x - mean) ** 2 for x in values) / (count - 1) if count > 1 else 0.0
        )
        std_dev = math.sqrt(variance)

        q1 = self._quantile(sorted_values, 0.25)
        q3 = self._quantile(sorted_values, 0.75)

        return {
            "count": count,
            "mean": mean,
            "median": median,
            "variance": variance,
            "std_dev": std_dev,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "quartiles": {"q1": q1, "q3": q3},
        }

    def distribution_info(
        self,
        distribution: str,
        value: float,
        *,
        kind: str = "continuous",
        params: Optional[Dict[str, Any]] = None,
    ) -> DistributionResult:
        params = params or {}
        distribution = distribution.lower()
        pdf = self._distribution_pdf(distribution, value, params)
        cdf = self._distribution_cdf(distribution, value, params)
        if kind == "discrete":
            cdf = self._distribution_cdf(distribution, math.floor(value), params)
        return DistributionResult(
            distribution=distribution,
            value=value,
            params=params,
            pdf=pdf,
            cdf=cdf,
        )

    def visualize(
        self,
        data: Sequence[float],
        bins: int = 10,
    ) -> Dict[str, Any]:
        if not data:
            raise ValueError("Data sequence must not be empty.")
        values = [float(x) for x in data]
        if bins < 1:
            bins = 1

        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            bin_edges = [min_val + i for i in range(bins + 1)]
        else:
            width = (max_val - min_val) / bins
            bin_edges = [min_val + i * width for i in range(bins + 1)]

        counts = [0] * bins
        for value in values:
            if value == max_val:
                counts[-1] += 1
                continue
            idx = int((value - min_val) / (bin_edges[1] - bin_edges[0]))
            counts[idx] += 1

        bins_data = []
        for i in range(bins):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            bins_data.append(
                {
                    "range": (start, end),
                    "count": counts[i],
                }
            )

        figure = None
        if plt is not None:
            figure, ax = plt.subplots()  # type: ignore[assignment]
            ax.hist(values, bins=bins, edgecolor="black")
            ax.set_title("Data distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        return {
            "bins": bins_data,
            "figure": figure,
            "matplotlib": figure is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _median(self, sorted_values: List[float]) -> float:
        count = len(sorted_values)
        mid = count // 2
        if count % 2 == 1:
            return sorted_values[mid]
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

    def _quantile(self, sorted_values: List[float], quantile: float) -> float:
        if not sorted_values:
            raise ValueError("No data points provided.")
        pos = (len(sorted_values) - 1) * quantile
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return sorted_values[int(pos)]
        lower_value = sorted_values[lower]
        upper_value = sorted_values[upper]
        return lower_value + (upper_value - lower_value) * (pos - lower)

    def _distribution_pdf(
        self, distribution: str, value: float, params: Dict[str, Any]
    ) -> Optional[float]:
        if distribution == "normal":
            mu = params.get("mean", 0.0)
            sigma = params.get("std", 1.0)
            if sigma <= 0:
                raise ValueError("Standard deviation must be positive.")
            coeff = 1 / (sigma * math.sqrt(2 * math.pi))
            exponent = -0.5 * ((value - mu) / sigma) ** 2
            return coeff * math.exp(exponent)
        if distribution == "uniform":
            a = params.get("a", 0.0)
            b = params.get("b", 1.0)
            if a >= b:
                raise ValueError("Uniform distribution requires a < b.")
            return 1 / (b - a) if a <= value <= b else 0.0
        if distribution == "exponential":
            lam = params.get("lambda", 1.0)
            if lam <= 0:
                raise ValueError("Lambda must be positive.")
            return lam * math.exp(-lam * value) if value >= 0 else 0.0
        if distribution == "bernoulli":
            p = params.get("p", 0.5)
            if not 0 <= p <= 1:
                raise ValueError("Probability must be between 0 and 1.")
            if value not in (0, 1):
                return 0.0
            return p if value == 1 else 1 - p
        if distribution == "binomial":
            n = int(params.get("n", 1))
            p = params.get("p", 0.5)
            k = int(value)
            if k < 0 or k > n:
                return 0.0
            return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return None

    def _distribution_cdf(
        self, distribution: str, value: float, params: Dict[str, Any]
    ) -> Optional[float]:
        if distribution == "normal":
            mu = params.get("mean", 0.0)
            sigma = params.get("std", 1.0)
            if sigma <= 0:
                raise ValueError("Standard deviation must be positive.")
            z = (value - mu) / (sigma * math.sqrt(2))
            return 0.5 * (1 + math.erf(z))
        if distribution == "uniform":
            a = params.get("a", 0.0)
            b = params.get("b", 1.0)
            if a >= b:
                raise ValueError("Uniform distribution requires a < b.")
            if value < a:
                return 0.0
            if value > b:
                return 1.0
            return (value - a) / (b - a)
        if distribution == "exponential":
            lam = params.get("lambda", 1.0)
            if lam <= 0:
                raise ValueError("Lambda must be positive.")
            if value < 0:
                return 0.0
            return 1 - math.exp(-lam * value)
        if distribution == "bernoulli":
            p = params.get("p", 0.5)
            if value < 0:
                return 0.0
            if value < 1:
                return 1 - p
            return 1.0
        if distribution == "binomial":
            n = int(params.get("n", 1))
            p = params.get("p", 0.5)
            cumulative = 0.0
            k = int(value)
            for i in range(0, k + 1):
                cumulative += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
            return min(cumulative, 1.0)
        return None
