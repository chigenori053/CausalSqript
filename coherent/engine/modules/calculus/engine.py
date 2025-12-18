from typing import Any, Dict
from coherent.engine.interfaces import BaseEngine
from coherent.engine.symbolic_engine import SymbolicEngine

class CalculusEngine(BaseEngine):
    """
    Engine for Calculus mode.
    Wraps the unified SymbolicEngine to perform evaluations.
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()
        # Ensure Calculus strategy is prioritized (though SymbolicEngine structure handles it via categories)
        # Note: SymbolicEngine's active_categories are per-instance or global?
        # check symbolic_engine.py: set_context(categories) sets self.active_categories.
        # So we should set context on our internal engine instance.
        from coherent.engine.math_category import MathCategory
        self.symbolic_engine.set_context([MathCategory.CALCULUS, MathCategory.ALGEBRA])

    def evaluate(self, node: Any, context: Dict[str, Any]) -> Any:
        """
        Evaluate the AST node/SymPy object within the given context.
        """
        # node is already a SymPy object or compatible from Parser
        return self.symbolic_engine.evaluate(node, context)
