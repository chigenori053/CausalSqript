"""Normalize human-friendly math expressions into strict CausalScript syntax."""

from __future__ import annotations

import re
from typing import List


class CausalScriptInputParser:
    """Utility that converts notebook-style math into SymPy-friendly strings."""

    _FUNCTION_TOKENS = {"SQRT"}
    _KNOWN_FUNCTIONS = {
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "exp",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "integrate",
        "diff",
        "Subs",
        "Matrix",
    }
    _KNOWN_CONSTANTS = {"pi", "e"}
    _UNARY_PRECEDERS = {"(", "+", "-", "*", "/", "**"}
    _OP_TOKENS = {"+", "-", "*", "/", "**"}

    @staticmethod
    def normalize(expr: str) -> str:
        text = expr.strip()
        if not text:
            return ""
        tokens = CausalScriptInputParser.tokenize(text)
        tokens = CausalScriptInputParser.normalize_unicode(tokens)
        tokens = CausalScriptInputParser.normalize_power(tokens)
        tokens = CausalScriptInputParser.expand_mixed_numbers(tokens)
        tokens = CausalScriptInputParser.normalize_matrices(tokens)
        tokens = CausalScriptInputParser.normalize_brackets(tokens)
        tokens = CausalScriptInputParser.normalize_derivatives(tokens)
        tokens = CausalScriptInputParser.normalize_integrals(tokens)
        tokens = CausalScriptInputParser.split_concatenated_identifiers(tokens)
        tokens = CausalScriptInputParser.insert_implicit_multiplication(tokens)
        tokens = CausalScriptInputParser.normalize_functions(tokens)
        return CausalScriptInputParser.to_string(tokens)

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
                                    if j > 0 and CausalScriptInputParser._is_identifier(tokens[j-1]):
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
                             if len(args) == 1 and CausalScriptInputParser._is_identifier(args[0]):
                                 var = args[0]
                    
                    replacement = ['diff', '(',] + operand_tokens + [',', var, ')']
                    tokens[operand_start : i+1] = replacement
                    changed = True
                    break # Restart loop to handle nesting safely
                i += 1
            
            if changed:
                continue

            # 2. Handle Leibniz notation (d/dx) (prefix)
            # Scan right-to-left to handle nesting naturally? 
            # Or just use the 'changed' loop.
            # Let's use the 'changed' loop and scan left-to-right, breaking on change.
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
                        target_tokens, next_idx = CausalScriptInputParser._consume_term(tokens, target_start)
                        
                        replacement = ['diff', '('] + target_tokens + [',', var_name, ')']
                        tokens[i : next_idx] = replacement
                        changed = True
                        break # Restart loop
                i += 1
            
            if not changed:
                break
                
        return tokens

    @staticmethod
    def normalize_brackets(tokens: List[str]) -> List[str]:
        """
        Convert bracket notation for definite integrals:
        [expr]_a^b -> (expr).subs(var, b) - (expr).subs(var, a)
        """
        while True:
            changed = False
            i = 0
            while i < len(tokens):
                if tokens[i] == '[':
                    # Find matching closing bracket
                    depth = 1
                    j = i + 1
                    while j < len(tokens):
                        if tokens[j] == '[':
                            depth += 1
                        elif tokens[j] == ']':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    
                    if j < len(tokens):
                        # Found matching ']'. Check for bounds _a^b
                        expr_tokens = tokens[i+1 : j]
                        
                        current = j + 1
                        lower_bound = None
                        upper_bound = None
                        
                        # Parse bounds (similar to integrals)
                        found_bound = True
                        while found_bound:
                            found_bound = False
                            if current < len(tokens) and tokens[current] == '_':
                                 current += 1
                                 bound_tokens, current = CausalScriptInputParser._consume_function_operand(tokens, current)
                                 lower_bound = CausalScriptInputParser.to_string(bound_tokens)
                                 found_bound = True
                            elif current < len(tokens) and tokens[current].startswith('_') and len(tokens[current]) > 1:
                                 lower_bound = tokens[current][1:]
                                 current += 1
                                 found_bound = True
                            elif current < len(tokens) and tokens[current] == '**':
                                 current += 1
                                 bound_tokens, current = CausalScriptInputParser._consume_function_operand(tokens, current)
                                 upper_bound = CausalScriptInputParser.to_string(bound_tokens)
                                 found_bound = True
                            elif current < len(tokens) and tokens[current] == '^': # Handle ^ as well if normalized
                                 current += 1
                                 bound_tokens, current = CausalScriptInputParser._consume_function_operand(tokens, current)
                                 upper_bound = CausalScriptInputParser.to_string(bound_tokens)
                                 found_bound = True

                        if lower_bound is not None and upper_bound is not None:
                            # Infer variable from expr_tokens
                            var = CausalScriptInputParser._infer_variable(expr_tokens)
                            expr_str = CausalScriptInputParser.to_string(expr_tokens)
                            
                            # Construct replacement: Subs(expr, var, upper) - Subs(expr, var, lower)
                            # We return tokens.
                            # Note: We need to be careful with tokenization of the replacement string.
                            # It's safer to construct the string and tokenize it, or build tokens manually.
                            # Let's build a string and tokenize it, as it's complex.
                            
                            replacement_str = f"(Subs({expr_str}, {var}, {upper_bound}) - Subs({expr_str}, {var}, {lower_bound}))"
                            replacement_tokens = CausalScriptInputParser.tokenize(replacement_str)
                            
                            tokens[i : current] = replacement_tokens
                            changed = True
                            break
                i += 1
            
            if not changed:
                break
            if not changed:
                break
        return tokens

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
                        # content tokens don't include outer []
                        
                        # Helper for simple splitting
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
                        
                        # We need to preserve comma separation within rows.
                        # Assuming rows are "1, 0, 1".
                        # If a row is "1 0 1" (space separated), we might need to verify comma insertion?
                        # insert_implicit_multiplication runs AFTER this? 
                        # Order in normalize: matrices -> implicit mult.
                        # So "1 2" might look like implicit mult unless we handle it?
                        # But standard list syntax is comma separated. 
                        # We assume the user provides commas or we rely on some other normalization?
                        # The user example has commas: "1, 0, 1".
                        
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

    @staticmethod
    def _infer_variable(tokens: List[str]) -> str:
        """Heuristic to infer the variable of integration/differentiation."""
        identifiers = set()
        for t in tokens:
            if CausalScriptInputParser._is_identifier(t) and t not in CausalScriptInputParser._KNOWN_FUNCTIONS and t not in CausalScriptInputParser._KNOWN_CONSTANTS:
                identifiers.add(t)
        
        if len(identifiers) == 1:
            return list(identifiers)[0]
        if 'x' in identifiers:
            return 'x'
        if 't' in identifiers:
            return 't'
        return 'x' # Default

    @staticmethod
    def _consume_term(tokens: List[str], start: int) -> tuple[List[str], int]:
        """
        Consume a mathematical term:
        - Single token (x, 1)
        - Parenthesized group ((...))
        - Function call (f(...))
        - Power (term^exp)
        """
        if start >= len(tokens):
            return ["0"], start
            
        # 1. Consume base
        is_parens = tokens[start] == '('
        base, index = CausalScriptInputParser._consume_function_operand(tokens, start)
        
        if is_parens:
            # Restore parens if it was a group
            base = ['('] + base + [')']
        
        # 2. Check for function call: Identifier followed by '('
        # _consume_function_operand consumes 'sin' as a single token.
        if len(base) == 1 and CausalScriptInputParser._is_identifier(base[0]):
             if index < len(tokens) and tokens[index] == '(':
                 args, index = CausalScriptInputParser._consume_function_operand(tokens, index)
                 # Restore parens for function call
                 base.append('(')
                 base.extend(args)
                 base.append(')')
        
        # 3. Check for power: '^' or '**'
        if index < len(tokens):
            if tokens[index] == '^' or tokens[index] == '**':
                op = tokens[index]
                exponent, index = CausalScriptInputParser._consume_function_operand(tokens, index + 1)
                base.append(op)
                base.extend(exponent)
                
        return base, index

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
                         bound_tokens, current = CausalScriptInputParser._consume_function_operand(tokens, current)
                         lower_bound = CausalScriptInputParser.to_string(bound_tokens)
                         found_bound = True
                    elif current < len(tokens) and tokens[current].startswith('_') and len(tokens[current]) > 1:
                         lower_bound = tokens[current][1:]
                         current += 1
                         found_bound = True
                    elif current < len(tokens) and tokens[current] == '**':
                         current += 1
                         bound_tokens, current = CausalScriptInputParser._consume_function_operand(tokens, current)
                         upper_bound = CausalScriptInputParser.to_string(bound_tokens)
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
                    # Error: No differential found for this integral.
                    # Replace '∫' to avoid infinite loop.
                    tokens[start_idx] = 'INTEGRAL_ERROR'
                    continue
                
                # Extract integrand
                integrand_tokens = tokens[current : diff_idx]
                
                # Construct replacement: integrate(integrand, var) or integrate(integrand, (var, lower, upper))
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
                # Safety break to prevent infinite loops on error
                if '∫' in tokens:
                    idx = tokens.index('∫')
                    tokens[idx] = 'INTEGRAL_ERROR'
                break
                
        return tokens

    @staticmethod
    def tokenize(expr: str) -> List[str]:
        tokens: List[str] = []
        index = 0
        length = len(expr)
        while index < length:
            char = expr[index]
            if char.isspace():
                index += 1
                continue
            if char.isdigit() or (char == "." and index + 1 < length and expr[index + 1].isdigit()):
                end = index + 1
                while end < length and (expr[end].isdigit() or expr[end] == "."):
                    end += 1
                tokens.append(expr[index:end])
                index = end
                continue
            if char.isalpha() or char == "_":
                end = index + 1
                while end < length and (expr[end].isalnum() or expr[end] == "_"):
                    end += 1
                tokens.append(expr[index:end])
                index = end
                continue
            if char == "*":
                if index + 1 < length and expr[index + 1] == "*":
                    tokens.append("**")
                    index += 2
                else:
                    tokens.append("*")
                    index += 1
                continue
            if char in "+-/^(),'[];":
                tokens.append(char)
                index += 1
                continue
            if char in {"²", "³", "√"}:
                tokens.append(char)
                index += 1
                continue
            tokens.append(char)
            index += 1
        return tokens

    @staticmethod
    def normalize_unicode(tokens: List[str]) -> List[str]:
        result: List[str] = []
        for token in tokens:
            if token == "²":
                result.extend(["**", "2"])
            elif token == "³":
                result.extend(["**", "3"])
            elif token == "√":
                result.append("SQRT")
            elif token == "π":
                result.append("pi")
            elif token == "∫":
                result.append("∫")
            else:
                result.append(token)
        return result

    @staticmethod
    def normalize_power(tokens: List[str]) -> List[str]:
        return ["**" if token == "^" else token for token in tokens]

    @staticmethod
    def expand_mixed_numbers(tokens: List[str]) -> List[str]:
        """Convert mixed numbers like 1(3/4) or 1 3/4 into addition form."""
        result: List[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if CausalScriptInputParser._is_integer(token):
                if (
                    i + 5 < len(tokens)
                    and tokens[i + 1] == "("
                    and CausalScriptInputParser._is_number(tokens[i + 2])
                    and tokens[i + 3] == "/"
                    and CausalScriptInputParser._is_number(tokens[i + 4])
                    and tokens[i + 5] == ")"
                ):
                    result.extend(["(", token, "+", tokens[i + 2], "/", tokens[i + 4], ")"])
                    i += 6
                    continue
                if (
                    i + 3 < len(tokens)
                    and CausalScriptInputParser._is_number(tokens[i + 1])
                    and tokens[i + 2] == "/"
                    and CausalScriptInputParser._is_number(tokens[i + 3])
                ):
                    result.extend(["(", token, "+", tokens[i + 1], "/", tokens[i + 3], ")"])
                    i += 4
                    continue
            result.append(token)
            i += 1
        return result

    @staticmethod
    def split_concatenated_identifiers(tokens: List[str]) -> List[str]:
        result: List[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # If we see a known function, we assume the following (...) block contains non-splittable identifiers.
            if token in CausalScriptInputParser._KNOWN_FUNCTIONS and i + 1 < len(tokens) and tokens[i+1] == '(':
                result.append(token) # func name
                result.append('(')   # open paren
                i += 2
                
                paren_depth = 1
                # We need to handle the case of no arguments, e.g. func()
                if i < len(tokens) and tokens[i] == ')':
                    result.append(')')
                    i += 1
                    continue

                while i < len(tokens):
                    inner_token = tokens[i]
                    if inner_token == '(':
                        paren_depth += 1
                    elif inner_token == ')':
                        paren_depth -= 1

                    # Add all tokens inside parens without splitting
                    result.append(inner_token)
                    i += 1
                    if paren_depth == 0:
                        break
                continue

            if CausalScriptInputParser._should_split_identifier(token):
                result.extend(list(token))
            else:
                result.append(token)
            i += 1
        return result

    @staticmethod
    def insert_implicit_multiplication(tokens: List[str]) -> List[str]:
        result: List[str] = []
        prev_token: str | None = None
        for token in tokens:
            if prev_token is not None and CausalScriptInputParser._needs_multiplication(prev_token, token):
                result.append("*")
            result.append(token)
            prev_token = token
        return result

    @staticmethod
    def _needs_multiplication(prev: str, current: str) -> bool:
        if prev in CausalScriptInputParser._OP_TOKENS or prev == "(":
            return False
        if current in CausalScriptInputParser._OP_TOKENS or current in {")", ",", "**"}:
            return False
        if CausalScriptInputParser._is_function_token(prev):
            return False
        prev_is_term = (
            CausalScriptInputParser._is_number(prev)
            or (CausalScriptInputParser._is_identifier(prev) and not CausalScriptInputParser._is_known_function(prev))
            or prev == ")"
        )
        current_is_term = (
            CausalScriptInputParser._is_number(current)
            or CausalScriptInputParser._is_identifier(current)
            or CausalScriptInputParser._is_function_token(current)
            or current == "("
        )
        if not prev_is_term or not current_is_term:
            return False
        if CausalScriptInputParser._is_known_function(current):
            return True
        if CausalScriptInputParser._is_function_token(current):
            return True
        if current == "(":
            return True
        if CausalScriptInputParser._is_identifier(current):
            return not CausalScriptInputParser._is_known_function(current)
        return True

    @staticmethod
    def normalize_functions(tokens: List[str]) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "SQRT":
                result.extend(["sqrt", "("])
                operand, index = CausalScriptInputParser._consume_function_operand(tokens, index + 1)
                result.extend(operand)
                result.append(")")
                continue
            result.append(token)
            index += 1
        return result

    @staticmethod
    def _consume_function_operand(tokens: List[str], start: int) -> tuple[List[str], int]:
        if start >= len(tokens):
            return ["0"], start
        token = tokens[start]
        if token == "(":
            depth = 1
            index = start + 1
            captured: List[str] = []
            while index < len(tokens):
                current = tokens[index]
                if current == "(":
                    depth += 1
                elif current == ")":
                    depth -= 1
                    if depth == 0:
                        return captured, index + 1
                captured.append(current)
                index += 1
            return captured, index
        return [token], start + 1

    @staticmethod
    def to_string(tokens: List[str]) -> str:
        pieces: List[str] = []
        prev: str | None = None
        for token in tokens:
            if token in {"+", "-"}:
                if token == "-" and (prev is None or prev in CausalScriptInputParser._UNARY_PRECEDERS):
                    pieces.append("-")
                else:
                    pieces.append(" ")
                    pieces.append(token)
                    pieces.append(" ")
            else:
                pieces.append(token)
            prev = token
        text = "".join(pieces)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("( ", "(").replace(" )", ")")
        return text.strip()

    @staticmethod
    def _is_identifier(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token))

    @staticmethod
    def _is_integer(token: str) -> bool:
        return bool(re.fullmatch(r"\d+", token))

    @staticmethod
    def _is_number(token: str) -> bool:
        return bool(re.fullmatch(r"(?:\d+(?:\.\d+)?)|(?:\d*\.\d+)", token))

    @staticmethod
    def _is_known_function(token: str) -> bool:
        return token in CausalScriptInputParser._KNOWN_FUNCTIONS

    @staticmethod
    def _is_function_token(token: str) -> bool:
        return token in CausalScriptInputParser._FUNCTION_TOKENS

    _KNOWN_UNITS = {
        "cm", "mm", "km", "kg", "g", "mg", "m", "s", "h", "Hz", "N", "J", "W", "Pa",
        "min", "hr", "deg", "rad", "liter", "L"
    }

    @staticmethod
    def _should_split_identifier(token: str) -> bool:
        return (
            len(token) > 1
            and token.isalpha()
            and token.islower()
            and token not in CausalScriptInputParser._KNOWN_FUNCTIONS
            and token not in CausalScriptInputParser._KNOWN_CONSTANTS
            and token not in CausalScriptInputParser._KNOWN_UNITS
        )
