import ast
import re
from typing import Any, List
from causalscript.core.interfaces import BaseParser
from causalscript.core.errors import InvalidExprError

class ArithmeticParser(BaseParser):
    """
    Parser for arithmetic and linear algebra expressions.
    Handles matrix normalization and strict arithmetic validation.
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

    @staticmethod
    def normalize_matrices(tokens: List[str]) -> List[str]:
        """
        Convert matrix syntax [a, b; c, d] to Matrix([[a, b], [c, d]]).
        Supports nested lists if commas are used, but specifically looks for semicolons 
        as row separators to identify implicit matrices.
        """
        while True:
            changed = False
            i = 0
            while i < len(tokens):
                if tokens[i] == '[':
                    # Find matching closing bracket
                    depth = 1
                    j = i + 1
                    has_semicolon = False
                    while j < len(tokens):
                        if tokens[j] == '[':
                            depth += 1
                        elif tokens[j] == ']':
                            depth -= 1
                            if depth == 0:
                                break
                        elif tokens[j] == ';' and depth == 1:
                            has_semicolon = True
                        j += 1
                    
                    if j < len(tokens) and has_semicolon:
                        # Process matrix content
                        content = tokens[i+1 : j]
                        rows = []
                        current_row = []
                        
                        # Split content by ';' at depth 0 (relative to content)
                        k = 0
                        row_start = 0
                        local_depth = 0
                        while k < len(content):
                            if content[k] == '[' or content[k] == '(':
                                local_depth += 1
                            elif content[k] == ']' or content[k] == ')':
                                local_depth -= 1
                            elif content[k] == ';' and local_depth == 0:
                                # Row terminator
                                row_tokens = content[row_start:k]
                                if any(t.strip() for t in row_tokens):
                                     rows.append(row_tokens)
                                row_start = k + 1
                            k += 1
                        # Add last row
                        last_row = content[row_start:]
                        if any(t.strip() for t in last_row):
                             rows.append(last_row)
                        
                        # Construct Matrix string replacement
                        # Matrix([[r1], [r2]])
                        
                        repl = ['Matrix', '(', '[']
                        for idx, row in enumerate(rows):
                            if idx > 0:
                                repl.append(',')
                            repl.append('[')
                            repl.extend(row)
                            repl.append(']')
                        repl.append(']')
                        repl.append(')')
                        
                        tokens[i : j+1] = repl
                        changed = True
                        break
                i += 1
            if not changed:
                break
        return tokens
