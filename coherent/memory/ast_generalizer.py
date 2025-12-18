
import re

class ASTGeneralizer:
    """
    Abstracs concrete AST expressions into generalized forms for the Vector Network.
    Example: "x^2 + 2*x + 1" -> "_v0^2 + 2*_v0 + 1"
    """
    def __init__(self):
        # Regex to identify variables (simple heuristic: single letters or words not in keywords)
        # This is strictly a heuristic for the prototype.
        # Ideally we use the SymbolicEngine's parser to identify specialized symbols.
        self.var_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
        self.keywords = {
            "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "diff", "int", 
            "Integral", "Derivative", "Add", "Mul", "Pow", "Symbol", "Integer"
        }

    def generalize(self, expr: str) -> str:
        """
        Replaces variables with generic placeholders to allow structural matching.
        """
        # Naive implementation: Find all identifiers. If not keyword, map to _v0, _v1...
        # For simplicity in this phase, we map ALL variables to a single generic token "_v"
        # to enforce "structure over identity".
        
        def replace(match):
            token = match.group(0)
            if token in self.keywords:
                return token
            return "_v"

        return self.var_pattern.sub(replace, expr)
