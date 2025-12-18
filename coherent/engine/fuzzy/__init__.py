"""Fuzzy reasoning subsystem."""

from .config import FuzzyThresholdConfig
from .encoder import ExpressionEncoder, MLVector
from .judge import FuzzyJudge
from .metric import SimilarityMetric
from .types import (
    FuzzyLabel,
    FuzzyResult,
    FuzzyScore,
    NormalizedExpr,
)

__all__ = [
    "ExpressionEncoder",
    "FuzzyJudge",
    "FuzzyLabel",
    "FuzzyResult",
    "FuzzyScore",
    "FuzzyThresholdConfig",
    "MLVector",
    "NormalizedExpr",
    "SimilarityMetric",
]
