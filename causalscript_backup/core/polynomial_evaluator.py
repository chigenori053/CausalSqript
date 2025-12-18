"""Polynomial evaluator implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import ast_nodes as ast
from .errors import MissingProblemError
from .learning_logger import LearningLogger
from .fuzzy.judge import FuzzyJudge
from .fuzzy.types import NormalizedExpr


class PolynomialEvaluator:
    """Evaluate MathLang programs by comparing normalized polynomial strings."""

    def __init__(
        self,
        program: ast.ProgramNode,
        normalizer: Callable[[str], str],
        learning_logger: Optional[LearningLogger] = None,
        fuzzy_judge: FuzzyJudge | None = None,
    ) -> None:
        self.program = program
        self.normalizer = normalizer
        self.learning_logger = learning_logger or LearningLogger()
        self._state = "INIT"
        self._fatal_error = False
        self._mode = "strict"
        self._config: Dict[str, Any] = {}
        self._current_problem_expr: str | None = None
        self._last_expr_raw: str | None = None
        self._current_normalized: str | None = None
        self._fuzzy_judge = fuzzy_judge

    def run(self) -> bool:
        for node in self.program.body:
            if isinstance(node, ast.MetaNode):
                self._handle_meta(node)
            elif isinstance(node, ast.ConfigNode):
                self._handle_config(node)
            elif isinstance(node, ast.ModeNode):
                self._handle_mode(node)
            elif isinstance(node, ast.PrepareNode):
                self._handle_prepare(node)
            elif isinstance(node, ast.CounterfactualNode):
                self._handle_counterfactual(node)
            elif isinstance(node, ast.ProblemNode):
                self._handle_problem(node)
            elif isinstance(node, ast.StepNode):
                self._handle_step(node)
            elif isinstance(node, ast.EndNode):
                self._handle_end(node)
            elif isinstance(node, ast.ExplainNode):
                self._handle_explain(node)
        if self._state != "END":
            exc = MissingProblemError("Program did not reach an end statement.")
            self._fatal(
                phase="end",
                expression=None,
                rendered="Fatal: missing end statement.",
                exc=exc,
            )
        return not self._fatal_error and self._state == "END"

    def _handle_problem(self, node: ast.ProblemNode) -> None:
        if self._state != "INIT":
            exc = MissingProblemError("Problem already defined.")
            self._fatal(
                phase="problem",
                expression=node.expr,
                rendered=f"Duplicate problem: {node.expr}",
                exc=exc,
            )
        normalized = self._normalize_expr(node.expr, phase="problem")
        self._current_normalized = normalized
        self._current_problem_expr = node.expr
        self._last_expr_raw = node.expr
        self.learning_logger.record(
            phase="problem",
            expression=node.expr,
            rendered=f"Problem: {node.expr}",
            status="ok",
        )
        self._state = "PROBLEM_SET"

    def _handle_step(self, node: ast.StepNode) -> None:
        if self._state not in {"PROBLEM_SET", "STEP_RUN"}:
            exc = MissingProblemError("step declared before problem.")
            self._fatal(
                phase="step",
                expression=node.expr,
                rendered=f"Step ({node.step_id or 'unnamed'}): {node.expr}",
                exc=exc,
            )
        normalized = self._normalize_expr(node.expr, phase="step")
        is_valid = normalized == self._current_normalized
        status = "ok" if is_valid else "mistake"
        meta = {}
        if not is_valid:
            meta = {
                "reason": "invalid_step",
                "expected": self._current_normalized,
                "actual": normalized,
            }
        self.learning_logger.record(
            phase="step",
            expression=node.expr,
            rendered=f"Step ({node.step_id or 'unnamed'}): {node.expr}",
            status=status,
            meta=meta,
        )
        if is_valid:
            self._current_normalized = normalized
            self._state = "STEP_RUN"
            self._last_expr_raw = node.expr
            return

        self._run_fuzzy_judge(
            previous_expr=self._last_expr_raw or (self._current_problem_expr or ""),
            candidate_expr=node.expr,
            applied_rule_id=None,
        )

    def _handle_end(self, node: ast.EndNode) -> None:
        if self._state not in {"PROBLEM_SET", "STEP_RUN"}:
            exc = MissingProblemError("end declared before problem.")
            self._fatal(
                phase="end",
                expression=node.expr,
                rendered=f"End: {node.expr}",
                exc=exc,
            )
        if self._current_normalized is None:
            exc = MissingProblemError("Problem must be declared before end.")
            self._fatal(
                phase="end",
                expression=node.expr,
                rendered=f"End: {node.expr}",
                exc=exc,
            )
        if node.is_done:
            normalized_target = self._current_normalized
        else:
            normalized_target = self._normalize_expr(node.expr or "", phase="end")
        is_valid = normalized_target == self._current_normalized
        status = "ok" if is_valid else "mistake"
        meta = {}
        if not is_valid:
            meta = {
                "reason": "final_result_mismatch",
                "expected": self._current_normalized,
                "actual": normalized_target,
            }
        rendered = "End: done" if node.is_done else f"End: {node.expr}"
        self.learning_logger.record(
            phase="end",
            expression=node.expr if not node.is_done else None,
            rendered=rendered,
            status=status,
            meta=meta,
        )
        self._state = "END"
        if not node.is_done:
            self._last_expr_raw = node.expr

    def _handle_explain(self, node: ast.ExplainNode) -> None:
        if self._state == "INIT":
            exc = MissingProblemError("Explain cannot appear before problem.")
            self._fatal(
                phase="explain",
                expression=None,
                rendered="Explain before problem.",
                exc=exc,
            )
        self.learning_logger.record(
            phase="explain",
            expression=None,
            rendered=node.text,
            status="ok",
        )

    def _handle_meta(self, node: ast.MetaNode) -> None:
        self.learning_logger.record(
            phase="meta",
            expression=None,
            rendered=f"Meta: {node.data}",
            status="ok",
            meta={"data": dict(node.data)},
        )

    def _handle_config(self, node: ast.ConfigNode) -> None:
        self._config.update(node.options)
        self.learning_logger.record(
            phase="config",
            expression=None,
            rendered=f"Config: {node.options}",
            status="ok",
            meta={"options": dict(node.options)},
        )

    def _handle_mode(self, node: ast.ModeNode) -> None:
        self._mode = node.mode or "strict"
        self.learning_logger.record(
            phase="mode",
            expression=None,
            rendered=f"Mode: {self._mode}",
            status="ok",
            meta={"mode": self._mode},
        )

    def _handle_prepare(self, node: ast.PrepareNode) -> None:
        statements = list(node.statements)
        if node.kind == "expr" and node.expr:
            statements.append(node.expr)
        if statements:
            for stmt in statements:
                self.learning_logger.record(
                    phase="prepare",
                    expression=stmt,
                    rendered=f"Prepare: {stmt}",
                    status="ok",
                )
            return
        if node.kind == "directive" and node.directive:
            self.learning_logger.record(
                phase="prepare",
                expression=node.directive,
                rendered=f"Prepare directive: {node.directive}",
                status="ok",
            )
            return
        if node.kind == "auto":
            self.learning_logger.record(
                phase="prepare",
                expression=None,
                rendered="Prepare: auto",
                status="ok",
            )
            return
        self.learning_logger.record(
            phase="prepare",
            expression=None,
            rendered="Prepare: (empty)",
            status="ok",
        )

    def _handle_counterfactual(self, node: ast.CounterfactualNode) -> None:
        block = {"assume": dict(node.assume), "expect": node.expect}
        assume_repr = block["assume"] if block["assume"] else {}
        rendered_expect = block["expect"] or ""
        self.learning_logger.record(
            phase="counterfactual",
            expression=node.expect,
            rendered=f"Counterfactual assume={assume_repr} expect={rendered_expect}",
            status="ok",
            meta=block,
        )

    def _run_fuzzy_judge(
        self,
        *,
        previous_expr: str,
        candidate_expr: str,
        applied_rule_id: str | None,
    ) -> None:
        if self._fuzzy_judge is None or self._current_problem_expr is None:
            return
        if self._mode not in {"fuzzy", "causal", "cf"}:
            return
        normalized_problem = self._normalized_expr(self._current_problem_expr)
        normalized_prev = self._normalized_expr(previous_expr)
        normalized_candidate = self._normalized_expr(candidate_expr)
        fuzzy_result = self._fuzzy_judge.judge_step(
            problem_expr=normalized_problem,
            previous_expr=normalized_prev,
            candidate_expr=normalized_candidate,
            applied_rule_id=applied_rule_id,
            candidate_rule_id=None,
            explain_text=None,
        )
        self.learning_logger.record(
            phase="fuzzy",
            expression=candidate_expr,
            rendered=f"Fuzzy: {fuzzy_result['label'].value} ({fuzzy_result['score']['combined_score']:.2f})",
            status="ok",
            meta=fuzzy_result,
        )

    def _normalized_expr(self, expr: str) -> NormalizedExpr:
        tokens = expr.split()
        return {"raw": expr, "sympy": expr, "tokens": tokens}

    def _normalize_expr(self, expr: str, *, phase: str) -> str:
        try:
            return self.normalizer(expr)
        except Exception as exc:  # pragma: no cover - delegated to fatal log
            self._fatal(
                phase=phase,
                expression=expr,
                rendered=f"{phase.title()}: {expr}",
                exc=exc,
            )
        raise AssertionError("unreachable")  # pragma: no cover - satisfies type checkers

    def _fatal(
        self,
        *,
        phase: str,
        expression: str | None,
        rendered: str | None,
        exc: Exception,
    ) -> None:
        self._fatal_error = True
        self.learning_logger.record(
            phase=phase,
            expression=expression,
            rendered=rendered,
            status="fatal",
            meta={"exception": exc.__class__.__name__, "message": str(exc)},
        )
        raise exc
