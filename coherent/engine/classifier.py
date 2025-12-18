from typing import List, Set
from .symbolic_engine import SymbolicEngine

class ExpressionClassifier:
    """
    Classifies mathematical expressions into domains (Arithmetic, Algebra, Calculus, etc.)
    to guide rule selection and engine configuration.
    """

    def __init__(self, engine: SymbolicEngine):
        self.engine = engine

    def classify(self, expr: str) -> List[str]:
        """
        Determines the applicable domains for the given expression.
        Returns a list of domain IDs (e.g., ["arithmetic"], ["algebra", "arithmetic"]).
        The order indicates priority (primary domain first).
        """
        domains: List[str] = []
        
        # 1. Check for Calculus features
        if self._is_calculus(expr):
            domains.append("calculus")
        
        # 2. Check for Geometry/Vector features (Placeholder for now)
        # if self._is_geometry(expr):

        # 2. Check for Linear Algebra features
        if self._is_linear_algebra(expr):
            domains.append("linear_algebra")

        # 3. Check for Statistics features
        if self._is_statistics(expr):
            domains.append("statistics")
        
        # 4. Check for Geometry features
        if self._is_geometry(expr):
            domains.append("geometry")
            
        # 5. Check for Algebra vs Arithmetic
        # If it has free symbols, it's Algebra.
        # If it's purely numeric, it's Arithmetic.
        # Note: Calculus/Geometry expressions usually imply Algebra too, but 
        # we might want specific rules.
        
        if self.engine.is_numeric(expr):
            domains.append("arithmetic")
        else:
            domains.append("algebra")
            # Algebra usually includes Arithmetic rules (e.g. 1+1 in an algebraic expr)
            # But strictly speaking, the *problem* is algebra.
            # We can append arithmetic as a fallback or secondary context.
            domains.append("arithmetic")
            
        return domains

    def _is_calculus(self, expr: str) -> bool:
        """Checks for calculus-specific notation or functions."""
        # Check for integral/derivative keywords or symbols
        # This depends on how they are represented in the string.
        # "Integral", "Derivative", "Diff", "int", "d/dx"
        # Also the bracket notation [expr]_a^b which becomes Subs
        
        keywords = {"Integral", "Derivative", "Diff", "Subs", "Limit", "integrate", "diff"}
        # Simple string check might be enough for now, or check AST/SymPy structure
        # But expr is a string here.
        
        # Check for "int " or "diff " or "Integral(" etc.
        # Also check for normalized forms.
        
        # Heuristic:
        for kw in keywords:
            if kw in expr:
                return True
            
        # Check for "int" or "diff" tokens if not fully normalized to SymPy class names
        # But usually we classify *before* or *during* evaluation.
        # If expr is "int(x^2, x)", it contains "int".
        
        return False

    def _is_linear_algebra(self, expr: str) -> bool:
        """Checks for linear algebra-specific notation or functions."""
        keywords = {"Vector", "Matrix", "dot", "cross", "det", "inverse", "transpose", "eigenvals", "eigenvects"}
        for kw in keywords:
            if kw in expr:
                return True
        return False

    def _is_statistics(self, expr: str) -> bool:
        """Checks for statistics-specific notation or functions."""
        keywords = {"mean", "median", "variance", "std", "pdf", "cdf", "normal", "binomial", "bernoulli", "uniform", "exponential"}
        for kw in keywords:
            if kw in expr:
                return True
        return False

    def _is_geometry(self, expr: str) -> bool:
        """Checks for geometry-specific notation or functions."""
        keywords = {"Point", "Line", "Circle", "Triangle", "area", "perimeter", "distance"}
        for kw in keywords:
            if kw in expr:
                return True
        return False
