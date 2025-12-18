"""Typed structures for the fuzzy reasoning engine."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, TypedDict


class NormalizedExpr(TypedDict):
    raw: str
    sympy: str
    tokens: List[str]


class FuzzyScore(TypedDict):
    expr_similarity: float
    rule_similarity: float
    text_similarity: float
    combined_score: float


class FuzzyLabel(str, Enum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    APPROX_EQ = "approx_eq"
    ANALOGOUS = "analogous"
    CONTRADICT = "contradict"
    UNKNOWN = "unknown"


class FuzzyResult(TypedDict):
    label: FuzzyLabel
    score: FuzzyScore
    reason: str
    debug: dict[str, Any]
