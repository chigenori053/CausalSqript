from typing import List
import uuid

from ..knowledge_registry import KnowledgeRegistry
from ..symbolic_engine import SymbolicEngine
from .types import Hypothesis

class HypothesisGenerator:
    """Generates candidate next steps (Hypotheses) by searching the KnowledgeRegistry."""

    def __init__(self, registry: KnowledgeRegistry, engine: SymbolicEngine):
        self.registry = registry
        self.engine = engine

    def generate(self, expr: str) -> List[Hypothesis]:
        """
        Generates all valid hypotheses for the given expression.
        """
        # Normalize equation syntax (a = b -> Eq(a, b))
        # This allows SymbolicEngine to parse it correctly
        normalized_expr = self._normalize_input(expr)
        
        matches = self.registry.match_rules(normalized_expr)
        candidates = []
        
        for rule, next_expr in matches:
            # Post-process next_expr to be user-friendly if it's Eq
            # But for now, we keep internal format or try to convert back
            # Converting back to '=' for display is better
            display_next = self._format_output(next_expr)

            # Create a hypothesis
            # ID is purely internal/ephemeral for now
            h_id = str(uuid.uuid4())[:8]
            
            hyp = Hypothesis(
                id=h_id,
                rule_id=rule.id,
                current_expr=expr, # Use original user input as current
                next_expr=display_next,
                metadata={
                    "rule_description": rule.description,
                    "rule_category": rule.category,
                    "rule_priority": rule.priority
                }
            )
            candidates.append(hyp)
            
        return candidates

    def _normalize_input(self, expr: str) -> str:
        """Converts user-friendly equation syntax to internal Eq() format if needed."""
        if "=" in expr and "Eq(" not in expr:
            parts = expr.split("=")
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                return f"Eq({lhs}, {rhs})"
        return expr

    def _format_output(self, expr: str) -> str:
        """Converts internal Eq() format back to user-friendly '=' syntax."""
        if expr.startswith("Eq(") and expr.endswith(")"):
            # Simple parser for string manipulation
            # A bit risky if nested, but works for simple cases
            inner = expr[3:-1]
            # Split by comma respecting parenthesis?
            # For now, simplistic split
            if "," in inner:
                # Find the top-level comma
                depth = 0
                split_idx = -1
                for i, char in enumerate(inner):
                    if char in "([{":
                        depth += 1
                    elif char in ")]}":
                        depth -= 1
                    elif char == "," and depth == 0:
                        split_idx = i
                        break
                
                if split_idx != -1:
                    lhs = inner[:split_idx].strip()
                    rhs = inner[split_idx+1:].strip()
                    return f"{lhs} = {rhs}"
        return expr
