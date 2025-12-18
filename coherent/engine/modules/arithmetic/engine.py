import ast
import operator
from typing import Any, Dict
from coherent.engine.interfaces import BaseEngine
from coherent.engine.errors import EvaluationError

class FastMathEngine(BaseEngine):
    """
    A lightweight evaluation engine for arithmetic expressions.
    Uses Python's native operators and does NOT depend on SymPy.
    """
    
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(self, node: Any, context: Dict[str, Any] = None) -> Any:
        # Context is ignored in Arithmetic mode as variables are not allowed
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body, context)
            
        if isinstance(node, ast.Constant):
            return node.value
            
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left, context)
            right = self.evaluate(node.right, context)
            op_func = self._OPERATORS.get(type(node.op))
            if op_func:
                return op_func(left, right)
            raise EvaluationError(f"Unsupported operator: {type(node.op).__name__}")
            
        if isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand, context)
            op_func = self._OPERATORS.get(type(node.op))
            if op_func:
                return op_func(operand)
            raise EvaluationError(f"Unsupported unary operator: {type(node.op).__name__}")
            
        raise EvaluationError(f"Unsupported node type: {type(node).__name__}")
