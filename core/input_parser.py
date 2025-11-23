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
        tokens = CausalScriptInputParser.split_concatenated_identifiers(tokens)
        tokens = CausalScriptInputParser.insert_implicit_multiplication(tokens)
        tokens = CausalScriptInputParser.normalize_functions(tokens)
        return CausalScriptInputParser.to_string(tokens)

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
            if char in "+-/^(),":
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

    @staticmethod
    def _should_split_identifier(token: str) -> bool:
        return (
            len(token) > 1
            and token.isalpha()
            and token.islower()
            and token not in CausalScriptInputParser._KNOWN_FUNCTIONS
            and token not in CausalScriptInputParser._KNOWN_CONSTANTS
        )
