"""Line-oriented parser for the MathLang Core DSL."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Iterable, List
import re
import difflib

from . import ast_nodes as ast
from .errors import SyntaxError
from .input_parser import CausalScriptInputParser
from .i18n import get_language_pack


@dataclass
class ParsedLine:
    number: int
    content: str


class Parser:
    """Parse MathLang DSL source text into a ProgramNode."""

    def __init__(self, source: str, *, language: dict | None = None) -> None:
        normalized_source = dedent(source)
        self._lines = [
            ParsedLine(idx, line.rstrip("\n"))
            for idx, line in enumerate(normalized_source.splitlines(), start=1)
        ]
        self._language = language

    def parse(self) -> ast.ProgramNode:
        nodes: list[ast.Node] = []
        index = 0
        problem_seen = False
        prepare_seen = False
        step_seen = False
        while index < len(self._lines):
            parsed = self._lines[index]
            raw = parsed.content
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            keyword, tail, rest, has_colon = self._extract_keyword_parts(stripped)
            if keyword == "problem" and has_colon:
                content = rest.strip()
                if not content:
                    block_lines, index = self._collect_block(index + 1)
                    nodes.append(self._parse_problem_block(block_lines, parsed.number))
                else:
                    nodes.append(self._parse_problem(content, parsed.number))
                    index += 1
                problem_seen = True
            elif keyword == "step" and has_colon:
                content = rest.strip()
                current_indent = len(raw) - len(raw.lstrip(" "))
                
                # Check for following block regardless of inline content
                next_line_idx = index + 1
                is_block = False
                if next_line_idx < len(self._lines):
                    next_line = self._lines[next_line_idx].content
                    next_indent = len(next_line) - len(next_line.lstrip(" "))
                    if next_line.strip() and next_indent > current_indent:
                        is_block = True
                
                if is_block:
                     block_lines, index = self._collect_block(next_line_idx)
                     if content:
                         block_lines.insert(0, content)
                     
                     if self._is_mapping_block(block_lines):
                        nodes.append(self._parse_step_block(block_lines, parsed.number))
                     else:
                        nodes.append(self._parse_step_multiline(block_lines, parsed.number))
                elif not content:
                    # No inline content and no block -> empty step? Or maybe block starts with empty line?
                    # _collect_block handles empty lines if they are indented or if we are strict.
                    # But here we already checked next line indentation.
                    # If not content and not indented block, it's an error or empty block.
                    # Let's try collecting block anyway to be safe (e.g. if next line is empty but line after is indented? _collect_block stops at empty line usually)
                    # But we must be careful not to consume sibling steps.
                    # If next line is not indented deeper, we shouldn't consume it.
                    nodes.append(self._parse_step_legacy("", None, parsed.number))
                    index += 1
                else:
                    nodes.append(self._parse_step_legacy(content, tail or None, parsed.number))
                    index += 1
                step_seen = True


            elif keyword == "end" and has_colon:
                content = rest.strip()
                if not content:
                     # Check if there is a block (e.g. multi-line end condition?)
                     # For now, end usually is single line or 'done'.
                     # But if user writes:
                     # end:
                     #  x = 3
                     #  y = 7
                     # We should support it.
                     block_lines, index = self._collect_block(index + 1)
                     if block_lines:
                         nodes.append(self._parse_end_block(block_lines, parsed.number))
                     else:
                         # Just 'end:' with nothing? Assume done?
                         nodes.append(self._parse_end("done", parsed.number))
                else:
                    nodes.append(self._parse_end(content, parsed.number))
                    index += 1
            elif keyword == "explain" and has_colon:
                nodes.append(self._parse_explain(rest.strip(), parsed.number))
                index += 1
            elif keyword == "meta" and has_colon:
                block_lines, index = self._collect_block(index + 1)
                nodes.append(
                    ast.MetaNode(line=parsed.number, data=self._parse_mapping(block_lines))
                )
            elif keyword == "config" and has_colon:
                block_lines, index = self._collect_block(index + 1)
                nodes.append(
                    ast.ConfigNode(
                        line=parsed.number,
                        options=self._parse_config_options(self._parse_mapping(block_lines)),
                    )
                )
            elif keyword == "mode" and has_colon:
                mode_value = rest.strip() or "strict"
                nodes.append(ast.ModeNode(line=parsed.number, mode=mode_value))
                index += 1
            elif keyword == "prepare" and has_colon:
                if prepare_seen:
                    raise SyntaxError("Multiple prepare statements are not allowed.")
                if step_seen:
                    raise SyntaxError("prepare must appear before steps.")
                content = rest.strip()
                if content:
                    nodes.append(self._parse_prepare_inline(content, parsed.number))
                    index += 1
                else:
                    block_lines, index = self._collect_block(index + 1)
                    nodes.append(self._parse_prepare_block(block_lines, parsed.number))
                prepare_seen = True
            elif keyword == "counterfactual" and has_colon:
                block_lines, index = self._collect_block(index + 1)
                cf_data = self._parse_mapping(block_lines)
                assume = {}
                for key, value in (cf_data.get("assume") or {}).items():
                    text = str(value).strip()
                    if not text:
                        continue
                    assume[key] = self._normalize_expr(text)
                expect_value = cf_data.get("expect")
                expect = self._normalize_expr(expect_value) if isinstance(expect_value, str) else None
                nodes.append(
                    ast.CounterfactualNode(
                        line=parsed.number,
                        assume=assume,
                        expect=expect,
                    )
                )
            elif keyword == "scenario" and has_colon:
                # scenario "Name":
                #   var = value
                name = self._strip_string_literal(tail.strip(), parsed.number)
                block_lines, index = self._collect_block(index + 1)
                assignments = {}
                # Parse block lines as assignments
                for raw in block_lines:
                    stripped = raw.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" in stripped:
                        key, value = stripped.split("=", 1)
                        assignments[key.strip()] = self._normalize_expr(value.strip())
                
                nodes.append(
                    ast.ScenarioNode(
                        line=parsed.number,
                        name=name,
                        assignments=assignments
                    )
                )
            elif keyword == "sub_problem" and has_colon:
                nodes.append(self._parse_sub_problem(rest.strip(), parsed.number))
                index += 1
            else:
                # Fuzzy matching for better error messages
                known_keywords = [
                    "problem", "step", "end", "explain", "meta", "config", 
                    "mode", "prepare", "counterfactual", "scenario", "sub_problem"
                ]
                # Extract the first word as the potential keyword
                potential_keyword = raw.strip().split(":")[0].strip().lower()
                matches = difflib.get_close_matches(potential_keyword, known_keywords, n=1, cutoff=0.6)
                
                msg = f"Unsupported statement on line {parsed.number}: {raw.strip()}"
                if matches:
                    msg += f" Did you mean '{matches[0]}'?"
                
                raise SyntaxError(msg)

        program = ast.ProgramNode(line=None, body=nodes)
        i18n = get_language_pack()

        if not any(isinstance(node, ast.ProblemNode) for node in nodes):
            raise SyntaxError(i18n.text("parser.problem_required"))
        
        # Allow implicit end if missing
        if not any(isinstance(node, ast.EndNode) for node in nodes):
            # Append an implicit 'end: done'
            nodes.append(ast.EndNode(expr=None, is_done=True, line=None))
            
        if not any(isinstance(node, ast.StepNode) for node in nodes):
            raise SyntaxError(i18n.text("parser.step_required"))
        return program

    def _parse_problem(self, content: str, number: int) -> ast.ProblemNode:
        # Handle potential double prefixing (e.g. "problem: problem: ...")
        if content.strip().lower().startswith("problem:"):
            content = content.strip()[8:].strip()
            
        # Check for "Name = Expression" syntax (top-level '=' only)
        name = None
        assignment = self._split_top_level_assignment(content)
        if assignment:
            possible_name, rhs = assignment
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", possible_name):
                name = possible_name
                content = rhs

        expr = self._normalize_expr(content)
        # If content still has '=', it's an equation, not an assignment
        if "=" in expr:
             expr = self._to_equation(expr)
             
        if not expr:
            raise SyntaxError(f"Problem expression required on line {number}.")
        return ast.ProblemNode(expr=expr, name=name, line=number)

    def _parse_sub_problem(self, content: str, number: int) -> ast.SubProblemNode:
        raw_expr = content.strip()
        
        # Check for "Variable = Expression" syntax (top-level '=' only)
        target_variable = None
        assignment = self._split_top_level_assignment(content)
        if assignment:
            possible_var, rhs = assignment
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", possible_var):
                target_variable = possible_var
                content = rhs
        
        expr = self._normalize_expr(content)
        if "=" in expr:
             expr = self._to_equation(expr)

        if not expr:
            raise SyntaxError(f"Sub-problem expression required on line {number}.")
        return ast.SubProblemNode(expr=expr, raw_expr=raw_expr, target_variable=target_variable, line=number)

    def _parse_step_legacy(self, content: str, step_id: str | None, number: int) -> ast.StepNode:
        # Handle potential double prefixing
        if content.strip().lower().startswith("step:"):
            content = content.strip()[5:].strip()

        raw = content.strip()
        expr = self._normalize_expr(content)
        if "=" in expr:
             expr = self._to_equation(expr)

        if not expr:
            raise SyntaxError(f"Step expression required on line {number}.")
        return ast.StepNode(step_id=step_id, expr=expr, raw_expr=raw, line=number)

    def _parse_step_block(self, block_lines: List[str], number: int) -> ast.StepNode:
        data = self._parse_mapping(block_lines)
        after = data.get("after")
        before = data.get("before")
        note = data.get("note")
        if not isinstance(after, str) or not after.strip():
            raise SyntaxError(f"Step block missing 'after' expression near line {number}.")
        raw_after = after.strip()
        node = ast.StepNode(expr=self._normalize_expr(raw_after), raw_expr=raw_after, line=number)
        if isinstance(before, str):
            normalized_before = self._normalize_expr(before.strip())
            node.before_expr = normalized_before
        if isinstance(note, str):
            node.note = note.strip()
        return node

    def _parse_end(self, content: str, number: int) -> ast.EndNode:
        expr_text = content.strip()
        if not expr_text or expr_text.lower() == "done":
            return ast.EndNode(expr=None, is_done=True, line=number)
        return ast.EndNode(expr=self._normalize_expr(expr_text), is_done=False, line=number)

    def _parse_explain(self, content: str, number: int) -> ast.ExplainNode:
        raw_text = content.strip()
        if not raw_text:
            raise SyntaxError(f"Explain statement requires a string literal on line {number}.")
        text = self._strip_string_literal(raw_text, number)
        return ast.ExplainNode(text=text, line=number)

    def _strip_string_literal(self, literal: str, number: int) -> str:
        if len(literal) < 2 or literal[0] not in {'"', "'"} or literal[-1] != literal[0]:
            raise SyntaxError(f"Explain statement requires a quoted string on line {number}.")
        return literal[1:-1]

    def _extract_keyword_parts(self, stripped: str) -> tuple[str, str, str, bool]:
        leading, sep, rest = stripped.partition(":")
        if not sep:
            return leading.lower(), "", "", False
        keyword_part = leading.strip()
        if not keyword_part:
            return "", "", rest, True
        lower_part = keyword_part.lower()
        if lower_part.startswith("step"):
            base = "step"
            tail = keyword_part[len("step") :].strip()
        else:
            base_segment = keyword_part.split()[0]
            base = base_segment.lower()
            tail = keyword_part[len(base_segment) :].strip()
        return base, tail, rest, True

    def _collect_block(self, start_index: int) -> tuple[List[str], int]:
        block: List[str] = []
        index = start_index
        while index < len(self._lines):
            raw = self._lines[index].content
            stripped = raw.strip()
            if not raw.startswith(" ") and stripped and not stripped.startswith("#"):
                break
            if not raw.strip() and not raw.startswith(" "):
                break
            block.append(raw)
            index += 1
        return block, index

    def _parse_mapping(self, block_lines: List[str]) -> dict:
        result: dict = {}
        index = 0
        while index < len(block_lines):
            raw = block_lines[index]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if ":" not in stripped:
                index += 1
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                nested_lines, new_index = self._collect_nested_lines(block_lines, index + 1, indent)
                result[key] = self._parse_mapping(nested_lines) if nested_lines else {}
                index = new_index
            else:
                result[key] = value
                index += 1
        return result

    def _collect_nested_lines(self, block_lines: List[str], start: int, base_indent: int) -> tuple[List[str], int]:
        nested: List[str] = []
        index = start
        while index < len(block_lines):
            raw = block_lines[index]
            stripped = raw.strip()
            if not stripped:
                nested.append(raw)
                index += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent <= base_indent:
                break
            nested.append(raw)
            index += 1
        return nested, index

    def _parse_prepare_inline(self, content: str, number: int) -> ast.PrepareNode:
        lower = content.lower()
        if lower == "auto":
            return ast.PrepareNode(kind="auto", line=number)
        if not content:
            return ast.PrepareNode(kind="empty", line=number)
        if self._looks_like_directive(content):
            return ast.PrepareNode(kind="directive", directive=content, line=number)
        return ast.PrepareNode(kind="expr", expr=self._normalize_statement(content), line=number)

    def _parse_prepare_block(self, block_lines: List[str], number: int) -> ast.PrepareNode:
        statements: List[str] = []
        for raw in block_lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("-"):
                normalized = self._normalize_statement(stripped[1:].strip())
                if normalized:
                    statements.append(normalized)
        if statements:
            return ast.PrepareNode(kind="list", statements=statements, line=number)
        return ast.PrepareNode(kind="empty", line=number)

    def _parse_config_options(self, data: dict) -> dict:
        options: dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                options[key] = value
            elif isinstance(value, str):
                options[key] = self._parse_scalar(value)
            else:
                options[key] = value
        return options

    def _parse_scalar(self, value: str) -> object:
        lower = value.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _normalize_expr(self, value: str) -> str:
        return CausalScriptInputParser.normalize(value)

    def _split_top_level_assignment(self, text: str) -> tuple[str, str] | None:
        """
        Split on the first '=' that is not nested in parentheses/brackets/braces.
        Returns (lhs, rhs) or None if no such '=' exists.
        """
        depth = 0
        for idx, ch in enumerate(text):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(depth - 1, 0)
            elif ch == "=" and depth == 0:
                return text[:idx].strip(), text[idx + 1 :].strip()
        return None

    def _normalize_statement(self, statement: str) -> str:
        stripped = statement.strip()
        if not stripped:
            return ""
        if "=" in stripped:
            name, expr = stripped.split("=", 1)
            normalized = CausalScriptInputParser.normalize(expr.strip())
            return f"{name.strip()} = {normalized}"
        return CausalScriptInputParser.normalize(stripped)

    def _looks_like_directive(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\([^()]*\)", value))

    def _is_mapping_block(self, block_lines: List[str]) -> bool:
        """Check if a block looks like a key-value mapping (e.g. before: ..., after: ...)."""
        for line in block_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                key, _ = stripped.split(":", 1)
                if key.strip() in {"before", "after", "note", "rule", "id"}:
                    return True
        return False

    def _to_equation(self, expr: str) -> str:
        assignment = self._split_top_level_assignment(expr)
        if assignment:
            lhs, rhs = assignment
            return f"Eq({lhs.strip()}, {rhs.strip()})"
        return expr

    def _parse_problem_block(self, block_lines: List[str], number: int) -> ast.ProblemNode:
        # Join lines into a System(...) expression
        exprs = []
        for line in block_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                norm = self._normalize_expr(stripped)
                exprs.append(self._to_equation(norm))
        
        if not exprs:
            raise SyntaxError(f"Problem block is empty on line {number}.")
            
        if len(exprs) == 1:
            return ast.ProblemNode(expr=exprs[0], line=number)
            
        # Create a System expression
        system_expr = f"System({', '.join(exprs)})"
        return ast.ProblemNode(expr=system_expr, line=number)

    def _parse_step_multiline(self, block_lines: List[str], number: int) -> ast.StepNode:
        exprs = []
        raw_lines = []
        for line in block_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                norm = self._normalize_expr(stripped)
                exprs.append(self._to_equation(norm))
                raw_lines.append(stripped)
        
        if not exprs:
            raise SyntaxError(f"Step block is empty on line {number}.")
            
        if len(exprs) == 1:
            return ast.StepNode(expr=exprs[0], raw_expr=raw_lines[0], line=number)
            
        system_expr = f"System({', '.join(exprs)})"
        raw_expr = "\n".join(raw_lines)
        return ast.StepNode(expr=system_expr, raw_expr=raw_expr, line=number)

    def _parse_end_block(self, block_lines: List[str], number: int) -> ast.EndNode:
        exprs = []
        for line in block_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                norm = self._normalize_expr(stripped)
                exprs.append(self._to_equation(norm))
        
        if not exprs:
            # Empty block -> done?
            return ast.EndNode(expr=None, is_done=True, line=number)
            
        if len(exprs) == 1:
            return ast.EndNode(expr=exprs[0], is_done=False, line=number)
            
        system_expr = f"System({', '.join(exprs)})"
        return ast.EndNode(expr=system_expr, is_done=False, line=number)
