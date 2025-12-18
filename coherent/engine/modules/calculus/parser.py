from typing import Any, List
from coherent.engine.interfaces import BaseParser
from coherent.engine.input_parser import CoherentInputParser
from coherent.engine.symbolic_engine import SymbolicEngine

class CalculusParser(BaseParser):
    """
    Parser for Calculus mode.
    Handles normalization of derivatives (Lagrange/Leibniz) and integrals (definite/indefinite).
    """
    
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()

    def parse(self, text: str) -> Any:
        """
        Parse the text into a SymPy expression (or fallback AST).
        """
        # 1. Normalize
        normalized = CoherentInputParser.normalize(text)
        
        # 2. Convert to Internal Representation (SymPy object)
        return self.symbolic_engine.to_internal(normalized)

    def validate(self, text: str) -> bool:
        try:
            self.parse(text)
            return True
        except Exception:
            return False

    @staticmethod
    def normalize_derivatives(tokens: List[str]) -> List[str]:
        """
        Convert derivative syntax:
        1. Leibniz: d/dx f(x) -> diff(f(x), x)
        2. Lagrange: f(x)' -> diff(f(x), x) (or inferred var)
        """
        while True:
            changed = False
            
            # 1. Handle Lagrange notation (') first (postfix)
            i = 0
            while i < len(tokens):
                if tokens[i] == "'":
                    # Found prime. Identify operand.
                    if i == 0:
                        i += 1
                        continue
                    
                    operand_end = i
                    operand_start = i - 1
                    
                    if tokens[i-1] == ')':
                        depth = 1
                        j = i - 2
                        while j >= 0:
                            if tokens[j] == ')':
                                depth += 1
                            elif tokens[j] == '(':
                                depth -= 1
                                if depth == 0:
                                    operand_start = j
                                    if j > 0 and CalculusParser._is_identifier(tokens[j-1]):
                                        operand_start = j - 1
                                    break
                            j -= 1
                    
                    operand_tokens = tokens[operand_start:operand_end]
                    
                    # Infer variable
                    var = 'x'
                    if '(' in operand_tokens and operand_tokens[-1] == ')':
                         depth = 0
                         arg_start = -1
                         for k in range(len(operand_tokens)-1, -1, -1):
                             if operand_tokens[k] == ')':
                                 depth += 1
                             elif operand_tokens[k] == '(':
                                 depth -= 1
                                 if depth == 0:
                                     arg_start = k
                                     break
                         
                         if arg_start != -1:
                             args = operand_tokens[arg_start+1 : -1]
                             if len(args) == 1 and CalculusParser._is_identifier(args[0]):
                                 var = args[0]
                    
                    replacement = ['diff', '(',] + operand_tokens + [',', var, ')']
                    tokens[operand_start : i+1] = replacement
                    changed = True
                    break # Restart loop to handle nesting safely
                i += 1
            
            if changed:
                continue

            # 2. Handle Leibniz notation (d/dx) (prefix)
            i = 0
            while i < len(tokens):
                if tokens[i] == 'd' and i + 2 < len(tokens) and tokens[i+1] == '/':
                    var_name = None
                    consumed = 0
                    
                    if tokens[i+2].startswith('d') and len(tokens[i+2]) > 1:
                        var_name = tokens[i+2][1:]
                        consumed = 3
                    elif i + 3 < len(tokens) and tokens[i+2] == 'd':
                        var_name = tokens[i+3]
                        consumed = 4
                    
                    if var_name:
                        target_start = i + consumed
                        target_tokens, next_idx = CoherentInputParser._consume_term(tokens, target_start)
                        
                        replacement = ['diff', '('] + target_tokens + [',', var_name, ')']
                        tokens[i : next_idx] = replacement
                        changed = True
                        break # Restart loop
                i += 1
            
            if not changed:
                break
                
        return tokens

    @staticmethod
    def normalize_integrals(tokens: List[str]) -> List[str]:
        """
        Convert integral syntax (∫ f(x) dx) to integrate(f(x), x) or integrate(f(x), (x, a, b)).
        Handles nested integrals by processing innermost first (last '∫' in tokens).
        """
        while '∫' in tokens:
            try:
                # Find the last '∫' to handle innermost first
                indices = [i for i, t in enumerate(tokens) if t == '∫']
                if not indices:
                    break
                start_idx = indices[-1]
                
                current = start_idx + 1
                lower_bound = None
                upper_bound = None
                
                # Parse bounds (allow any order)
                found_bound = True
                while found_bound:
                    found_bound = False
                    if current < len(tokens) and tokens[current] == '_':
                         current += 1
                         bound_tokens, current = CoherentInputParser._consume_function_operand(tokens, current)
                         lower_bound = CoherentInputParser.to_string(bound_tokens)
                         found_bound = True
                    elif current < len(tokens) and tokens[current].startswith('_') and len(tokens[current]) > 1:
                         lower_bound = tokens[current][1:]
                         current += 1
                         found_bound = True
                    elif current < len(tokens) and tokens[current] == '**':
                         current += 1
                         bound_tokens, current = CoherentInputParser._consume_function_operand(tokens, current)
                         upper_bound = CoherentInputParser.to_string(bound_tokens)
                         found_bound = True
                
                # Scan for differential (d<var> or d <var>)
                diff_idx = -1
                var_name = None
                consumed_tokens_for_diff = 0
                
                for k in range(current, len(tokens)):
                    t = tokens[k]
                    # Check for combined differential "dx"
                    if t.startswith('d') and len(t) > 1 and t[1:].isidentifier():
                        diff_idx = k
                        var_name = t[1:]
                        consumed_tokens_for_diff = 1
                        break
                    # Check for separated differential "d" "x"
                    if t == 'd' and k + 1 < len(tokens) and tokens[k+1].isidentifier():
                        diff_idx = k
                        var_name = tokens[k+1]
                        consumed_tokens_for_diff = 2
                        break
                
                if diff_idx == -1:
                    tokens[start_idx] = 'INTEGRAL_ERROR'
                    continue
                
                # Extract integrand
                integrand_tokens = tokens[current : diff_idx]
                
                # Construct integrand replacement
                replacement = ['integrate', '(']
                replacement.extend(integrand_tokens)
                replacement.append(',')
                
                if lower_bound is not None and upper_bound is not None:
                    replacement.append('(')
                    replacement.append(var_name)
                    replacement.append(',')
                    replacement.append(lower_bound)
                    replacement.append(',')
                    replacement.append(upper_bound)
                    replacement.append(')')
                else:
                    replacement.append(var_name)
                
                replacement.append(')')
                
                # Replace tokens
                end_replace_idx = diff_idx + consumed_tokens_for_diff
                tokens[start_idx : end_replace_idx] = replacement
                
            except Exception:
                if '∫' in tokens:
                    idx = tokens.index('∫')
                    tokens[idx] = 'INTEGRAL_ERROR'
                break
                
        return tokens

    @staticmethod
    def _is_identifier(token: str) -> bool:
        # Helper duplicated from InputParser or imported?
        # Better to use CoherentInputParser._is_identifier if possible, but it's protected.
        # Let's duplicate or make public in InputParser later.
        import re
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token))

