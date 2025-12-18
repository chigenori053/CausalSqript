from enum import Enum
from typing import Set
from causalscript.core.input_parser import CausalScriptInputParser
from causalscript.core.math_category import MathCategory

class CategoryAnalyzer:
    """Analyzes expression strings to determine their mathematical category."""

    # Keywords mapping for specialized domains
    _KEYWORDS = {
        MathCategory.CALCULUS: {"diff", "integrate", "limit", "d/dx", "∫", "Subs", "Derivative", "Integral"},
        MathCategory.GEOMETRY: {"Point", "Line", "Circle", "Triangle", "Segment", "Ray", "Polygon", "area", "volume"},
        MathCategory.LINEAR_ALGEBRA: {"Matrix", "Vector", "dot", "cross", "det", "eigenvals", "eigenvects", "inverse", "transpose"},
        MathCategory.STATISTICS: {"mean", "median", "mode", "variance", "std_dev", "normal", "uniform", "binomial", "pdf", "cdf"},
    }

    @staticmethod
    def detect(expr: str) -> MathCategory:
        try:
            # Reuse existing tokenizer for consistency
            tokens = set(CausalScriptInputParser.tokenize(expr))
        except Exception:
            # Fallback for simple strings
            tokens = set(expr.replace("(", " ").replace(")", " ").replace(",", " ").split())

        # Check specific keywords first
        for category, keywords in CategoryAnalyzer._KEYWORDS.items():
            if not tokens.isdisjoint(keywords):
                return category
            
            # Check for non-alphanumeric keywords (like d/dx, ∫) in the raw string
            # because tokenizer might split them
            for kw in keywords:
                if not kw.isalnum() and kw in expr:
                    return category
        
        # Check for variables (Algebra vs Arithmetic)
        # Exclude constants like pi, e from variable detection
        # Simple heuristic: if there are alpha characters that are not keywords/constants
        if any(t.isidentifier() and t not in {"pi", "e", "done", "true", "false"} for t in tokens):
            return MathCategory.ALGEBRA
            
        return MathCategory.ARITHMETIC
