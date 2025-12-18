from dataclasses import dataclass
from typing import Any

from coherent.engine.symbolic_engine import SymbolicEngine

@dataclass
class GoalState:
    is_solved: bool
    complexity_score: float
    distance_to_goal: float

class GoalScanner:
    """Analyzes expressions to determine goal status and complexity."""

    def __init__(self, engine: SymbolicEngine):
        self.engine = engine

    def scan(self, expr: str) -> GoalState:
        """
        Scans the expression to evaluate its state.
        For now, assumes the goal is 'variable isolation' (x = constant).
        """
        is_solved = self._is_solved_form(expr)
        complexity = self._calculate_complexity(expr)
        
        # Distance heuristic: Complexity is a proxy for distance
        # Ideally 0 complexity means solved (not really, x=3 has complexity)
        # We start with simple complexity score.
        distance = complexity 
        if is_solved:
            distance = 0.0

        return GoalState(
            is_solved=is_solved,
            complexity_score=complexity,
            distance_to_goal=distance
        )

    def _is_solved_form(self, expr: str) -> bool:
        """Checks if expr is in the form 'x = number'."""
        # This is a naive check.
        # Should check if it's an Equation first.
        if "=" not in expr:
            return False
            
        parts = expr.split("=")
        if len(parts) != 2:
            return False
            
        left = parts[0].strip()
        right = parts[1].strip()
        
        # Check if left is a single symbol and right is numeric
        # Use Simple check based on string or engine
        # Avoid engine overhead if possible, but safely we should use it.
        
        # Check if left is pure symbol
        if not left.isidentifier() and not self._is_symbol_only(left):
            return False
            
        # Check if right is numeric
        if not self.engine.is_numeric(right):
            return False
            
        return True

    def _is_symbol_only(self, s: str) -> bool:
        # Remove potential whitespace or parens if simple
        s = s.strip()
        return s.replace("_", "").isalnum() and not s.isdigit()

    def _calculate_complexity(self, expr: str) -> float:
        """
        Calculates a complexity score based on string length and structure.
        Lower is usually better (simpler).
        """
        # 1. String length (weighted small)
        score = len(expr) * 0.1
        
        # 2. Number of operations (heuristic)
        ops = ["+", "-", "*", "/", "^", "(", ")"]
        for char in expr:
            if char in ops:
                score += 1.0
                
        return score
