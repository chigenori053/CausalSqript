"""Evaluator and engine implementations for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import ast_nodes as ast
from core.errors import CausalScriptError, EvaluationError, InconsistentEndError, InvalidStepError, InvalidExprError, MissingProblemError
from .learning_logger import LearningLogger
from .symbolic_engine import SymbolicEngine
from .knowledge_registry import KnowledgeRegistry
from .fuzzy.judge import FuzzyJudge
from .fuzzy.types import NormalizedExpr


class Engine:
    """Abstract engine interface."""

    def set(self, expr: str) -> None:  # pragma: no cover - documentation guard
        raise NotImplementedError

    def check_step(self, expr: str) -> dict:  # pragma: no cover - documentation guard
        raise NotImplementedError

    def finalize(self, expr: str | None) -> dict:  # pragma: no cover - documentation guard
        raise NotImplementedError

    def set_variable(self, name: str, value: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def add_scenario(self, name: str, context: Dict[str, Any]) -> None:  # pragma: no cover
        """Add a scenario context."""
        pass

    def evaluate(self, expr: str, context: Optional[Dict[str, Any]] = None) -> Any:  # pragma: no cover
        raise NotImplementedError


@dataclass
class SymbolicEvaluationEngine(Engine):
    """Concrete engine that relies on SymbolicEngine and KnowledgeRegistry."""

    symbolic_engine: SymbolicEngine
    knowledge_registry: KnowledgeRegistry | None = None
    _context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._current_expr: str | None = None

    def set(self, expr: str) -> None:
        self._current_expr = expr

    def set_variable(self, name: str, value: Any) -> None:
        self._context[name] = value

    def add_scenario(self, name: str, context: Dict[str, Any]) -> None:
        # SymbolicEvaluationEngine doesn't support scenarios natively yet, 
        # but we can ignore or log warning. For now, just pass.
        pass

    def evaluate(self, expr: str, context: Optional[Dict[str, Any]] = None) -> Any:
        eval_context = self._context.copy()
        if context:
            eval_context.update(context)
        return self.symbolic_engine.evaluate(expr, eval_context)

    def _apply_context(self, expr: str) -> str:
        if not self._context:
            return expr
        value = self.symbolic_engine.evaluate(expr, self._context.copy())
        if isinstance(value, dict) and value.get("not_evaluatable"):
            return expr
        return str(value)

    def check_step(self, expr: str) -> dict:
        if self._current_expr is None:
            raise MissingProblemError("Problem expression must be set before steps.")
        before = self._current_expr
        after = expr
        before_ctx = self._apply_context(before)
        after_ctx = self._apply_context(after)
        valid = self.symbolic_engine.is_equiv(before_ctx, after_ctx)

        rule_id: str | None = None
        rule_meta: dict[str, Any] | None = None
        if valid and self.knowledge_registry is not None:
            matched = self.knowledge_registry.match(before, after)
            if matched:
                rule_id = matched.id
                rule_meta = matched.to_metadata()

        details = {"explanation": self.symbolic_engine.explain(before, after)}
        result = {
            "before": before,
            "after": after,
            "valid": valid,
            "rule_id": rule_id,
            "rule_meta": rule_meta,
            "details": details,
        }
        if valid:
            self._current_expr = after
        return result

    def finalize(self, expr: str | None) -> dict:
        if self._current_expr is None:
            raise MissingProblemError("Cannot finalize before a problem is declared.")
        target = expr if expr is not None else self._current_expr
        
        before_ctx = self._apply_context(self._current_expr)
        target_ctx = self._apply_context(target)
        valid = self.symbolic_engine.is_equiv(before_ctx, target_ctx)
        details = {"explanation": self.symbolic_engine.explain(self._current_expr, target)}
        return {
            "before": self._current_expr,
            "after": target,
            "valid": valid,
            "rule_id": None,
            "details": details,
        }

    def _apply_context(self, expr: str) -> str:
        if not self._context:
            return expr
        try:
            value = self.symbolic_engine.evaluate(expr, self._context.copy())
        except EvaluationError:
            return expr
        if isinstance(value, dict) and value.get("not_evaluatable"):
            return expr
        return str(value)


class Evaluator:
    """Process a ProgramNode using the specified engine."""

    def __init__(
        self,
        program: ast.ProgramNode,
        engine: Engine,
        learning_logger: LearningLogger | None = None,
        fuzzy_judge: FuzzyJudge | None = None,
    ) -> None:
        self.program = program
        self.engine = engine
        self.learning_logger = learning_logger or LearningLogger()
        self._fuzzy_judge = fuzzy_judge
        self._state = "INIT"
        self._completed = False
        self._fatal_error = False
        self._has_mistake = False
        self._has_critical_mistake = False
        self._current_problem_expr: str | None = None
        self._last_expr_raw: str | None = None
        self._meta: Dict[str, str] = {}
        self._config: Dict[str, Any] = {}
        self._mode: str = "strict"
        self._context_stack: list[dict] = []
        self._scope_stack: list[dict] = []
        self._scope_counter: int = 0

    def _log(
        self,
        *,
        phase: str,
        expression: str | None,
        rendered: str | None,
        status: str,
        rule_id: str | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        scope_id = current_scope["id"] if current_scope else "main"
        parent_scope_id = current_scope["parent_id"] if current_scope else None
        depth = len(self._scope_stack)
        # Detect consecutive duplicates; if found, mark prior as redundant and skip logging.
        if self.learning_logger.records:
            last = self.learning_logger.records[-1]
            if (
                last.phase == phase
                and last.expression == expression
                and getattr(last, "rendered", rendered) == rendered
                and last.status == status
                and getattr(last, "scope_id", scope_id) == scope_id
                and getattr(last, "rule_id", rule_id) == rule_id
                and (getattr(last, "meta", {}) or {}) == (meta or {})
            ):
                try:
                    last.is_redundant = True
                except Exception:
                    pass
                return
        self.learning_logger.record(
            phase=phase,
            expression=expression,
            rendered=rendered,
            status=status,
            rule_id=rule_id,
            meta=meta,
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
            depth=depth,
            is_redundant=False,
        )

    @property
    def _symbolic_engine(self) -> SymbolicEngine:
        if hasattr(self.engine, "symbolic_engine"):
            return self.engine.symbolic_engine
        if hasattr(self.engine, "computation_engine") and hasattr(self.engine.computation_engine, "symbolic_engine"):
            return self.engine.computation_engine.symbolic_engine
        raise NotImplementedError("Engine does not support symbolic operations.")

    def run(self) -> bool:
        # Reset run-scoped flags
        self._has_mistake = False
        self._has_critical_mistake = False
        for node in self.program.body:
            if isinstance(node, ast.ProblemNode):
                self._handle_problem(node)
            elif isinstance(node, ast.MetaNode):
                self._handle_meta(node)
            elif isinstance(node, ast.ConfigNode):
                self._handle_config(node)
            elif isinstance(node, ast.ModeNode):
                self._handle_mode(node)
            elif isinstance(node, ast.PrepareNode):
                self._handle_prepare(node)
            elif isinstance(node, ast.StepNode):
                self._handle_step(node)
            elif isinstance(node, ast.EndNode):
                self._handle_end(node)
            elif isinstance(node, ast.ExplainNode):
                self._handle_explain(node)
            elif isinstance(node, ast.CounterfactualNode):
                self._handle_counterfactual(node)
            elif isinstance(node, ast.ScenarioNode):
                self._handle_scenario(node)
            elif isinstance(node, ast.SubProblemNode):
                self._handle_sub_problem(node)
            else:  # pragma: no cover - defensive block.
                raise SyntaxError(f"Unsupported node type: {type(node)}")
        if self._state != "END":
            exc = MissingProblemError("Program did not reach an end statement.")
            self._fatal(
                phase="end",
                expression=None,
                rendered="Fatal: missing end statement.",
                exc=exc,
            )
        self._completed = True
        return not self._fatal_error and not self._has_critical_mistake and self._state == "END"

    def _handle_problem(self, node: ast.ProblemNode) -> None:
        if self._state != "INIT":
            exc = MissingProblemError("Problem already defined.")
            self._fatal(
                phase="problem",
                expression=node.expr,
                rendered=f"Duplicate problem: {node.expr}",
                exc=exc,
            )
        try:
            self.engine.set(node.expr)
        except CausalScriptError as exc:
            self._fatal(
                phase="problem",
                expression=node.expr,
                rendered=f"Problem: {node.expr}",
                exc=exc,
            )
        self._current_problem_expr = node.expr
        self._last_expr_raw = node.expr
        self._log(
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
        try:
            result = self.engine.check_step(node.expr)
        except CausalScriptError as exc:
            self._fatal(
                phase="step",
                expression=node.expr,
                rendered=f"Step ({node.step_id or 'unnamed'}): {node.expr}",
                exc=exc,
            )
        is_valid = bool(result["valid"])
        status = "ok" if is_valid else "mistake"
        meta = dict(result.get("details", {}) or {})
        rule_meta = result.get("rule_meta")
        if rule_meta:
            meta.setdefault("rule", rule_meta)
        if not is_valid:
            meta.update(
                {
                    "reason": "invalid_step",
                    "expected": result.get("before"),
                    "before": result.get("before"),
                    "after": result.get("after"),
                }
            )
        self._log(
            phase="step",
            expression=node.expr,
            rendered=f"Step ({node.step_id or 'unnamed'}): {node.expr}",
            status=status,
            rule_id=result.get("rule_id"),
            meta=meta,
        )
        if status == "mistake":
            self._has_mistake = True
            if meta.get("critical"):
                self._has_critical_mistake = True
        if is_valid:
            self._last_expr_raw = node.expr
            self._state = "STEP_RUN"
            return

        self._run_fuzzy_judge(
            previous_expr=self._last_expr_raw or "",
            candidate_expr=node.expr,
            applied_rule_id=result.get("rule_id"),
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
        try:
            result = self.engine.finalize(node.expr)
        except CausalScriptError as exc:
            self._fatal(
                phase="end",
                expression=node.expr,
                rendered=f"End: {node.expr}",
                exc=exc,
            )
        is_valid = bool(result["valid"])
        status = "ok" if is_valid else "mistake"
        meta = dict(result.get("details", {}) or {})
        if not is_valid:
            meta.update(
                {
                    "reason": "final_result_mismatch",
                    "expected": result.get("before"),
                    "actual": result.get("after"),
                }
            )
        rendered = "End: done" if node.is_done else f"End: {node.expr}"
        self._log(
            phase="end",
            expression=node.expr if not node.is_done else None,
            rendered=rendered,
            status=status,
            meta=meta,
        )
        if status == "mistake":
            self._has_mistake = True
            if meta.get("critical"):
                self._has_critical_mistake = True
        
        if self._context_stack:
            ctx = self._context_stack.pop()
            parent_expr = ctx.get("parent_expr_for_replace") or ctx["parent_expr"]
            target_sub = ctx["target_sub_expr"]
            final_sub_result = result["after"]

            # Substitute back
            new_parent_expr = self._symbolic_engine.replace(parent_expr, target_sub, final_sub_result)
            
            self.engine.set(new_parent_expr)
            
            # Scope Management: Pop and Log
            if self._scope_stack:
                ended_scope = self._scope_stack.pop()
                self._log(
                    phase="scope_end",
                    expression=final_sub_result,
                    rendered=f"Ending scope: {ended_scope['id']}",
                    status="ok"
                )

            self._log(
                phase="sub_problem_end",
                expression=final_sub_result,
                rendered=f"Sub-problem done. Return to: {new_parent_expr}",
                status="ok"
            )
            
            self._state = "STEP_RUN"
            return

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
        self._log(
            phase="explain",
            expression=None,
            rendered=node.text,
            status="ok",
        )

    def _handle_meta(self, node: ast.MetaNode) -> None:
        self._meta.update(node.data)
        self._log(
            phase="meta",
            expression=None,
            rendered=f"Meta: {node.data}",
            status="ok",
            meta={"data": dict(node.data)},
        )

    def _handle_config(self, node: ast.ConfigNode) -> None:
        self._config.update(node.options)
        self._log(
            phase="config",
            expression=None,
            rendered=f"Config: {node.options}",
            status="ok",
            meta={"options": dict(node.options)},
        )

    def _handle_mode(self, node: ast.ModeNode) -> None:
        self._mode = node.mode
        self._log(
            phase="mode",
            expression=None,
            rendered=f"Mode: {node.mode}",
            status="ok",
            meta={"mode": node.mode},
        )

    def _handle_prepare(self, node: ast.PrepareNode) -> None:
        statements = list(node.statements)
        if node.kind == "expr" and node.expr:
            statements.append(node.expr)
        if statements:
            for stmt in statements:
                if "=" in stmt:
                    try:
                        name, expr = stmt.split("=", 1)
                        name = name.strip()
                        value = self.engine.evaluate(expr.strip())
                        if isinstance(value, dict) and value.get("not_evaluatable"):
                            self._log(
                                phase="prepare",
                                expression=stmt,
                                rendered=f"Prepare skipped (not evaluatable): {stmt}",
                                status="ok",
                            )
                        else:
                            self.engine.set_variable(name, value)
                            self._log(
                                phase="prepare",
                                expression=stmt,
                                rendered=f"Prepare: {stmt}",
                                status="ok",
                            )
                    except (ValueError, EvaluationError) as exc:
                        if isinstance(exc, EvaluationError) and str(exc) == "not_evaluatable":
                            self._log(
                                phase="prepare",
                                expression=stmt,
                                rendered=f"Prepare skipped (not evaluatable): {stmt}",
                                status="ok",
                            )
                            continue
                        self._fatal(
                            phase="prepare",
                            expression=stmt,
                            rendered=f"Prepare failed: {stmt}",
                            exc=exc,
                        )
                else:
                    self._log(
                        phase="prepare",
                        expression=stmt,
                        rendered=f"Prepare: {stmt}",
                        status="ok",
                    )
            return
        if node.kind == "directive" and node.directive:
            self._log(
                phase="prepare",
                expression=node.directive,
                rendered=f"Prepare directive: {node.directive}",
                status="ok",
            )
            return
        if node.kind == "auto":
            self._log(
                phase="prepare",
                expression=None,
                rendered="Prepare: auto",
                status="ok",
            )
            return
        self._log(
            phase="prepare",
            expression=None,
            rendered="Prepare: (empty)",
            status="ok",
        )

    def _handle_counterfactual(self, node: ast.CounterfactualNode) -> None:
        if not node.expect:
            return

        assume_context = {}
        for name, expr in node.assume.items():
            try:
                value = self.engine.evaluate(expr)
                if isinstance(value, dict) and value.get("not_evaluatable"):
                    raise EvaluationError("not_evaluatable")
                assume_context[name] = value
            except EvaluationError as exc:
                self._fatal(
                    phase="counterfactual",
                    expression=expr,
                    rendered=f"Counterfactual assumption failed: {name} = {expr}",
                    exc=exc,
                )
        
        try:
            result = self.engine.evaluate(node.expect, context=assume_context)
            if isinstance(result, dict) and result.get("not_evaluatable"):
                raise EvaluationError("not_evaluatable")
            self._log(
                phase="counterfactual",
                expression=node.expect,
                rendered=f"Counterfactual: expect {node.expect} -> {result}",
                status="ok",
                meta={"assume": node.assume, "result": result},
            )
        except EvaluationError as exc:
            self._fatal(
                phase="counterfactual",
                expression=node.expect,
                rendered=f"Counterfactual evaluation failed: {node.expect}",
                exc=exc,
            )

    def _handle_scenario(self, node: ast.ScenarioNode) -> None:
        context = {}
        for name, expr in node.assignments.items():
            try:
                value = self.engine.evaluate(expr)
                if isinstance(value, dict) and value.get("not_evaluatable"):
                    # Warn but continue? Or fail?
                    # For scenarios, we probably want concrete values.
                    self._log(
                        phase="scenario",
                        expression=expr,
                        rendered=f"Scenario assignment skipped (not evaluatable): {name} = {expr}",
                        status="warning",
                    )
                    continue
                context[name] = value
            except EvaluationError as exc:
                self._fatal(
                    phase="scenario",
                    expression=expr,
                    rendered=f"Scenario assignment failed: {name} = {expr}",
                    exc=exc,
                )
        
        self.engine.add_scenario(node.name, context)
        self._log(
            phase="scenario",
            expression=None,
            rendered=f"Scenario added: {node.name}",
            status="ok",
            meta={"context": context},
        )

    def _handle_sub_problem(self, node: ast.SubProblemNode) -> None:
        if self._state not in {"PROBLEM_SET", "STEP_RUN"}:
             exc = MissingProblemError("sub_problem declared before problem.")
             self._fatal(
                 phase="sub_problem",
                 expression=node.expr,
                 rendered=f"Sub-problem: {node.expr}",
                 exc=exc,
             )

        current_expr = getattr(self.engine, "_current_expr", None)
        if current_expr is None:
             exc = MissingProblemError("No active problem for sub-problem.")
             self._fatal(
                 phase="sub_problem",
                 expression=node.expr,
                 rendered=f"Sub-problem: {node.expr}",
                 exc=exc,
             )

        # Apply context to current_expr if available to handle variables defined in prepare
        current_expr_for_check = current_expr
        context = {}
        if hasattr(self.engine, "_context"):
            context = self.engine._context
        elif hasattr(self.engine, "computation_engine") and hasattr(self.engine.computation_engine, "variables"):
            context = self.engine.computation_engine.variables
            
        if context:
            # Avoid triggering full evaluation (e.g., trigonometric identities) while
            # checking sub-problem structure. Use lightweight textual substitution.
            current_expr_for_check = current_expr
            try:
                for k, v in context.items():
                    current_expr_for_check = current_expr_for_check.replace(str(k), str(v))
            except Exception:
                current_expr_for_check = current_expr

        if not self._symbolic_engine.is_subexpression(node.expr, current_expr_for_check):
             exc = InvalidStepError(f"'{node.expr}' is not a sub-expression of '{current_expr_for_check}'")
             self._fatal(
                 phase="sub_problem",
                 expression=node.expr,
                 rendered=f"Invalid sub-problem: {node.expr}",
                 exc=exc
             )

        self._context_stack.append({
            "parent_expr": current_expr,
            "parent_expr_for_replace": current_expr_for_check,
            "target_sub_expr": node.expr
        })

        self.engine.set(node.expr)
        
        # Scope Management
        self._scope_counter += 1
        new_scope_id = f"sub_{self._scope_counter}"
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        parent_id = current_scope["id"] if current_scope else "main"
        
        # Log scope start
        self._log(
            phase="scope_start",
            expression=node.expr,
            rendered=f"Starting sub-problem scope: {new_scope_id}",
            status="ok",
            meta={"scope_id": new_scope_id, "parent_id": parent_id}
        )
        
        self._scope_stack.append({"id": new_scope_id, "parent_id": parent_id})

        self._log(
            phase="sub_problem",
            expression=getattr(node, "raw_expr", node.expr),
            rendered=f"Sub-problem: {getattr(node, 'raw_expr', node.expr)}",
            status="ok"
        )
        self._state = "PROBLEM_SET"

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
        self._log(
            phase="fuzzy",
            expression=candidate_expr,
            rendered=f"Fuzzy: {fuzzy_result['label'].value} ({fuzzy_result['score']['combined_score']:.2f})",
            status="ok",
            meta=fuzzy_result,
        )

    def _normalized_expr(self, expr: str) -> NormalizedExpr:
        tokens = expr.split()
        return {"raw": expr, "sympy": expr, "tokens": tokens}

    def _fatal(
        self,
        *,
        phase: str,
        expression: str | None,
        rendered: str | None,
        exc: Exception,
    ) -> None:
        if isinstance(exc, EvaluationError) and str(exc) == "not_evaluatable":
            self._log(
                phase=phase,
                expression=expression,
                rendered=rendered,
                status="info",
                meta={"reason": "not_evaluatable"},
            )
            return
        self._fatal_error = True
        self._log(
            phase=phase,
            expression=expression,
            rendered=rendered,
            status="fatal",
            meta={"exception": exc.__class__.__name__, "message": str(exc)},
        )
        raise exc
