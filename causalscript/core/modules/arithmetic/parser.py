import ast
import re
from typing import Any
from causalscript.core.interfaces import BaseParser
from causalscript.core.errors import InvalidExprError

class ArithmeticParser(BaseParser):
    """
    A strict parser for arithmetic expressions.
    Allowed: Numbers, +, -, *, /, (), ., whitespace.
    Disallowed: Variables, functions, other operators.
    """
    
    # Regex for validation: only allows digits, dots, operators, parens, whitespace
    VALID_PATTERN = re.compile(r'^[0-9\+\-\*\/\(\)\.\s]+$')

    def validate(self, text: str) -> bool:
        if not text.strip():
            return False
        return bool(self.VALID_PATTERN.match(text))

    def parse(self, text: str) -> Any:
        if not self.validate(text):
            raise InvalidExprError(f"Invalid arithmetic expression: {text}")
        
        try:
            # Parse using Python's AST
            node = ast.parse(text, mode='eval')
            
            # Additional structural check to ensure only allowed nodes are present
            for child in ast.walk(node):
                if not isinstance(child, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                        ast.Constant, ast.Add, ast.Sub, 
                                        ast.Mult, ast.Div, ast.USub, ast.UAdd)):
                    raise InvalidExprError(f"Forbidden AST node: {type(child).__name__}")
            
            return node
        except SyntaxError as e:
            raise InvalidExprError(f"Syntax error: {e}")
        except Exception as e:
            if isinstance(e, InvalidExprError):
                raise e
            raise InvalidExprError(f"Parsing failed: {e}")
