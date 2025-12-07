from dataclasses import dataclass
from typing import Any, Dict, Optional
from causalscript.core.math_category import MathCategory
from causalscript.core.symbolic_engine import SymbolicEngine

@dataclass
class RenderContext:
    expression: str
    category: str
    metadata: Dict[str, Any]

class RenderingEngine:
    """
    Handles formatting of mathematical results into presentation-ready logic 
    (e.g. LaTeX, Markdown).
    """
    
    def __init__(self, symbolic_engine: Optional[SymbolicEngine] = None):
        self.symbolic_engine = symbolic_engine

    def render_result(self, result: Dict[str, Any]) -> None:
        """
        Enhances the result dictionary with a 'rendered' field in the details.
        
        Args:
            result: The step check result dictionary to modify in-place.
        """
        if "details" not in result:
            result["details"] = {}
            
        before = result.get("before", "")
        after = result.get("after", "")
        # Use the detected category from the result details if available
        category_val = result["details"].get("category", MathCategory.ALGEBRA.value)
        metadata = result["details"]
        
        result["details"]["rendered"] = {
            "before": self._render_expression(before, category_val, metadata),
            "after": self._render_expression(after, category_val, metadata)
        }

    def _render_expression(self, expr: str, category: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Renders a single expression based on its category.
        """
        if not expr:
            return ""

        # Delegate to SymbolicEngine for LaTeX conversion if available
        # This handles Algebra, Calculus, etc. properly via strategies.
        if self.symbolic_engine:
            rendered = self.symbolic_engine.to_latex(expr, context_domains=[category])
            # If to_latex didn't change anything, try our heuristics
            if rendered != expr:
                return rendered
        
        # Fallback manual rendering if no engine or if engine failed to render
        if category == "geometry":
             return self._render_geometry(expr, metadata)
        elif category == "calculus":
             return self._render_calculus(expr)
        elif category == "statistics":
             return self._render_statistics(expr)
             
        return expr

    def _render_geometry(self, expr: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if metadata and "description" in metadata:
            return f"{metadata['description']} ({expr})"
        return expr

    def _render_calculus(self, expr: str) -> str:
        # Simple fallback for standard calculus notation
        result = expr
        if "diff(" in result:
            result = result.replace("diff(", "\\frac{d}{dx}(")
        if "integrate(" in result:
             result = result.replace("integrate(", "\\int(")
        return result

    def _render_statistics(self, expr: str) -> str:
        return f"📊 {expr}"


# ==============================================================================
# Backward Compatibility - Legacy Renderer
# ==============================================================================

@dataclass
class RenderContext:
    expression: str
    category: str
    metadata: Dict[str, Any]


class ContentRenderer:
    """
    Legacy static renderer adapter.
    Please use RenderingEngine for new code.
    """

    @staticmethod
    def render_step(context: RenderContext) -> str:
        # Create a temporary engine instance (without symbolic engine)
        # to reuse the rendering logic.
        engine = RenderingEngine()
        return engine._render_expression(context.expression, context.category, context.metadata)
