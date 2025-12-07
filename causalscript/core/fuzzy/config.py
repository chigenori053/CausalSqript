"""Configuration for fuzzy judge."""

from __future__ import annotations

from dataclasses import dataclass
from causalscript.core.decision_theory import DecisionConfig


@dataclass
class FuzzyThresholdConfig:
    """
    Thresholds for fuzzy matching labels.
    
    Legacy configuration for backward compatibility or direct threshold usage.
    """
    exact: float = 1.0
    equivalent: float = 0.9
    approx_eq: float = 0.8
    analogous: float = 0.6
    contradict: float = 0.2
