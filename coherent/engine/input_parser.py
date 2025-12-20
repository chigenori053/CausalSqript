"""Normalize human-friendly math expressions into strict Coherent syntax."""

from __future__ import annotations

import re
from typing import List


class CoherentInputParser:
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
        "Point", "Line", "Circle", "Polygon", "Segment", "Ray", "Triangle"
    }

    @staticmethod
    def split_concatenated_identifiers(tokens: List[str]) -> List[str]:
        result: List[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # SPECIAL CASE: Dot notation (e.g. .distance)
            # If previous token was '.', treat this token as a property/method, do NOT split.
            if result and result[-1] == '.':
                 result.append(token)
                 i += 1
                 continue

            # If we see a known function, we assume the following (...) block contains non-splittable identifiers.
            if token in CoherentInputParser._KNOWN_FUNCTIONS and i + 1 < len(tokens) and tokens[i+1] == '(':
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

            if CoherentInputParser._should_split_identifier(token):
                result.extend(list(token))
            else:
                result.append(token)
            i += 1
        return result

    @staticmethod
    def insert_implicit_multiplication(tokens: List[str]) -> List[str]:
        result: List[str] = []
        for i, token in enumerate(tokens):
            if i > 0:
                prev = tokens[i-1]
                if CoherentInputParser._needs_multiplication(prev, token):
                    # Check for .method(...) exception to prevent .distance * (
                    is_method_call = False
                    if token == '(':
                        # If prev is identifier and preceded by .
                        if i > 1 and tokens[i-2] == '.':
                            is_method_call = True
                    
                    if not is_method_call:
                        result.append("*")
            result.append(token)
        return result

    _KNOWN_CONSTANTS = {"pi", "e"}
    _UNARY_PRECEDERS = {"(", "+", "-", "*", "/", "**"}
    _OP_TOKENS = {"+", "-", "*", "/", "**"}

    @staticmethod
    def normalize(expr: str) -> str:
        from coherent.engine.modules.arithmetic.parser import ArithmeticParser
        from coherent.engine.modules.calculus.parser import CalculusParser

        text = expr.strip()
        if not text:
            return ""
        tokens = CoherentInputParser.tokenize(text)
        tokens = CoherentInputParser.normalize_unicode(tokens)
        tokens = CoherentInputParser.normalize_power(tokens)
        tokens = CoherentInputParser.expand_mixed_numbers(tokens)
        
        # Delegate to ArithmeticParser for matrices
        tokens = ArithmeticParser.normalize_matrices(tokens)
        
        tokens = CoherentInputParser.normalize_brackets(tokens)
        
        # Delegate to CalculusParser for derivatives and integrals
        tokens = CalculusParser.normalize_derivatives(tokens)
        tokens = CalculusParser.normalize_integrals(tokens)
        
        tokens = CoherentInputParser.split_identifiers_patched(tokens)
        tokens = CoherentInputParser.insert_implicit_mult_patched(tokens)
        tokens = CoherentInputParser.normalize_functions(tokens)
        return CoherentInputParser.to_string(tokens)

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
                        # Note: This logic might conflict with matrices if not careful.
                        # But matrices are handled BEFORE this (normalize_matrices).
                        # And normalize_matrices consumes [ ... ; ... ] or [ [ ... ] ].
                        # If normalize_matrices didn't trigger, this might be a bracketed expr.
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
                            elif current < len(tokens) and tokens[current] == '^': # Handle ^ as well if normalized
                                 current += 1
                                 bound_tokens, current = CoherentInputParser._consume_function_operand(tokens, current)
                                 upper_bound = CoherentInputParser.to_string(bound_tokens)
                                 found_bound = True

                        if lower_bound is not None and upper_bound is not None:
                            # Infer variable from expr_tokens
                            var = CoherentInputParser._infer_variable(expr_tokens)
                            expr_str = CoherentInputParser.to_string(expr_tokens)
                            
                            # Construct replacement: Subs(expr, var, upper) - Subs(expr, var, lower)
                            # We return tokens.
                            # Note: We need to be careful with tokenization of the replacement string.
                            # It's safer to construct the string and tokenize it, or build tokens manually.
                            # Let's build a string and tokenize it, as it's complex.
                            
                            replacement_str = f"(Subs({expr_str}, {var}, {upper_bound}) - Subs({expr_str}, {var}, {lower_bound}))"
                            replacement_tokens = CoherentInputParser.tokenize(replacement_str)
                            
                            tokens[i : current] = replacement_tokens
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
            if CoherentInputParser._is_identifier(t) and t not in CoherentInputParser._KNOWN_FUNCTIONS and t not in CoherentInputParser._KNOWN_CONSTANTS:
                identifiers.add(t)
        
        if len(identifiers) == 1:
            return list(identifiers)[0]
        if 'x' in identifiers:
            return 'x'
        if 't' in identifiers:
            return 't'
        return 'x' # Default

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
            elif token == "·":
                result.append("*")
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
            if CoherentInputParser._is_integer(token):
                if (
                    i + 5 < len(tokens)
                    and tokens[i + 1] == "("
                    and CoherentInputParser._is_number(tokens[i + 2])
                    and tokens[i + 3] == "/"
                    and CoherentInputParser._is_number(tokens[i + 4])
                    and tokens[i + 5] == ")"
                ):
                    result.extend(["(", token, "+", tokens[i + 2], "/", tokens[i + 4], ")"])
                    i += 6
                    continue
                if (
                    i + 3 < len(tokens)
                    and CoherentInputParser._is_number(tokens[i + 1])
                    and tokens[i + 2] == "/"
                    and CoherentInputParser._is_number(tokens[i + 3])
                ):
                    result.extend(["(", token, "+", tokens[i + 1], "/", tokens[i + 3], ")"])
                    i += 4
                    continue
            result.append(token)
            i += 1
        return result

    @staticmethod
    def split_identifiers_patched(tokens: List[str]) -> List[str]:
        result: List[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # SPECIAL CASE: Dot notation (e.g. .distance)
            # If previous token was '.', treat this token as a property/method, do NOT split.
            if result and result[-1] == '.':
                 result.append(token)
                 i += 1
                 continue

            # If we see a known function, we assume the following (...) block contains non-splittable identifiers.
            if token in CoherentInputParser._KNOWN_FUNCTIONS and i + 1 < len(tokens) and tokens[i+1] == '(':
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

            if CoherentInputParser._should_split_identifier(token):
                result.extend(list(token))
            else:
                result.append(token)
            i += 1
        return result

    @staticmethod
    def insert_implicit_mult_patched(tokens: List[str]) -> List[str]:
        result: List[str] = []
        for i, token in enumerate(tokens):
            if i > 0:
                prev = tokens[i-1]
                if CoherentInputParser._needs_multiplication(prev, token):
                    # Check for .method(...) exception to prevent .distance * (
                    is_method_call = False
                    if token == '(':
                        # If prev is identifier and preceded by .
                        if i > 1 and tokens[i-2] == '.':
                            is_method_call = True
                    
                    if not is_method_call:
                        result.append("*")
            result.append(token)
        return result

    @staticmethod
    def _needs_multiplication(prev: str, current: str) -> bool:
        if prev in CoherentInputParser._OP_TOKENS or prev == "(":
            return False
        if current in CoherentInputParser._OP_TOKENS or current in {")", ",", "**"}:
            return False
        if CoherentInputParser._is_function_token(prev):
            return False
        prev_is_term = (
            CoherentInputParser._is_number(prev)
            or (CoherentInputParser._is_identifier(prev) and not CoherentInputParser._is_known_function(prev))
            or prev == ")"
        )
        current_is_term = (
            CoherentInputParser._is_number(current)
            or CoherentInputParser._is_identifier(current)
            or CoherentInputParser._is_function_token(current)
            or current == "("
        )
        if not prev_is_term or not current_is_term:
            return False
        if CoherentInputParser._is_known_function(current):
            return True
        if CoherentInputParser._is_function_token(current):
            return True
        if current == "(":
            return True
        if CoherentInputParser._is_identifier(current):
            return not CoherentInputParser._is_known_function(current)
        return True

    @staticmethod
    def normalize_functions(tokens: List[str]) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "SQRT":
                result.extend(["sqrt", "("])
                operand, index = CoherentInputParser._consume_function_operand(tokens, index + 1)
                result.extend(operand)
                result.append(")")
                continue
            result.append(token)
            index += 1
        return result

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
        base, index = CoherentInputParser._consume_function_operand(tokens, start)
        
        if is_parens:
            # Restore parens if it was a group
            base = ['('] + base + [')']
        
        # 2. Check for function call: Identifier followed by '('
        # _consume_function_operand consumes 'sin' as a single token.
        if len(base) == 1 and CoherentInputParser._is_identifier(base[0]):
             if index < len(tokens) and tokens[index] == '(':
                 args, index = CoherentInputParser._consume_function_operand(tokens, index)
                 # Restore parens for function call
                 base.append('(')
                 base.extend(args)
                 base.append(')')
        
        # 3. Check for Power: '^' or '**'
        if index < len(tokens):
            if tokens[index] == '^' or tokens[index] == '**':
                op = tokens[index]
                exponent, index = CoherentInputParser._consume_function_operand(tokens, index + 1)
                base.append(op)
                base.extend(exponent)
                
        return base, index

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
                if token == "-" and (prev is None or prev in CoherentInputParser._UNARY_PRECEDERS):
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
        return token in CoherentInputParser._KNOWN_FUNCTIONS

    @staticmethod
    def _is_function_token(token: str) -> bool:
        return token in CoherentInputParser._FUNCTION_TOKENS

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
            and token not in CoherentInputParser._KNOWN_FUNCTIONS
            and token not in CoherentInputParser._KNOWN_CONSTANTS
            and token not in CoherentInputParser._KNOWN_UNITS
        )
