import numpy as np
from ..ast_nodes import Expr, Add, Mul, Pow, Neg, Call, Div, RationalNode, Int, Rat, Sym
from typing import Union

class FeatureExtractor:
    """
    Extracts numerical features from AST nodes for use in the Optical Scoring Layer.
    """

    # Feature indices mapping
    # Node Types
    NODE_TYPES = [
        Add, Mul, Pow, Neg, Call, Div, RationalNode, Int, Rat, Sym
    ]
    
    # Feature Vector Layout:
    # [0..N-1]: Node Type Counts (One-hot-like, but we sum counts for the whole tree)
    # [N]: Max Depth
    # [N+1]: Number of Variables
    # [N+2]: Number of Constants
    
    def __init__(self, vector_size: int = 64):
        self.vector_size = vector_size
        self.node_type_map = {cls: i for i, cls in enumerate(self.NODE_TYPES)}

    def vectorize(self, ast_node: Expr) -> np.ndarray:
        """
        Converts an AST node into a fixed-size feature vector.
        """
        vector = np.zeros(self.vector_size, dtype=np.float32)
        
        self._traverse(ast_node, vector, depth=0)
        
        return vector

    def _traverse(self, node: Expr, vector: np.ndarray, depth: int):
        # Update Max Depth (Feature N)
        n_types = len(self.NODE_TYPES)
        if depth > vector[n_types]:
            vector[n_types] = float(depth)

        # Node Type Count
        if type(node) in self.node_type_map:
            idx = self.node_type_map[type(node)]
            vector[idx] += 1.0

        # Specific Features
        if isinstance(node, Sym):
            # Number of Variables (Feature N+1)
            vector[n_types + 1] += 1.0
        elif isinstance(node, (Int, Rat)):
            # Number of Constants (Feature N+2)
            vector[n_types + 2] += 1.0
            
        # Recursive Traversal
        # Using iter_child_nodes logic manually or simplified since we are in core
        # We can implement a simple dispatcher
        if isinstance(node, Add):
            for term in node.terms:
                self._traverse(term, vector, depth + 1)
        elif isinstance(node, Mul):
            for factor in node.factors:
                self._traverse(factor, vector, depth + 1)
        elif isinstance(node, Pow):
            self._traverse(node.base, vector, depth + 1)
            self._traverse(node.exp, vector, depth + 1)
        elif isinstance(node, Neg):
            self._traverse(node.expr, vector, depth + 1)
        elif isinstance(node, Call):
            for arg in node.args:
                self._traverse(arg, vector, depth + 1)
        elif isinstance(node, Div):
            self._traverse(node.left, vector, depth + 1)
            self._traverse(node.right, vector, depth + 1)
        elif isinstance(node, RationalNode):
            self._traverse(node.numerator, vector, depth + 1)
            self._traverse(node.denominator, vector, depth + 1)

