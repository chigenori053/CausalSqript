"""Threshold configuration for fuzzy judgement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuzzyThresholdConfig:
    exact: float = 0.99
    equivalent: float = 0.95
    approx_eq: float = 0.80
    analogous: float = 0.60
    contradict: float = 0.20
