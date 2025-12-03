from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional

class MathCategory(Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    GEOMETRY = "geometry"
    UNKNOWN = "unknown"

@dataclass
class CategoryResult:
    primary_category: MathCategory
    confidence: float = 1.0
    related_categories: List[MathCategory] = field(default_factory=list)
    details: dict = field(default_factory=dict)
