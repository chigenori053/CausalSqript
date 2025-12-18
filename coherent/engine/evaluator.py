"""Evaluator and engine implementations for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import ast_nodes as ast
from coherent.engine.errors import CoherentError, EvaluationError, InconsistentEndError, InvalidStepError, InvalidExprError, MissingProblemError
from .learning_logger import LearningLogger
from .symbolic_engine import SymbolicEngine
from .knowledge_registry import KnowledgeRegistry
from .fuzzy.judge import FuzzyJudge
from .fuzzy.judge import FuzzyJudge
from .fuzzy.types import NormalizedExpr
from .classifier import ExpressionClassifier
from .category_analyzer import CategoryAnalyzer


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
        self._scenarios: Dict[str, Dict[str, Any]] = {}
        # Initialize classifier
        # We need to import ExpressionClassifier inside the method or assume it's available globally
        # It is imported at module level in previous step (Step 843).
        self.classifier = ExpressionClassifier(self.symbolic_engine)

    def set(self, expr: str) -> None:
        self._current_expr = expr

    def set_variable(self, name: str, value: Any) -> None:
        self._context[name] = value

    def add_scenario(self, name: str, context: Dict[str, Any]) -> None:
        """Add a scenario context."""
        self._scenarios[name] = context

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
            # Detect category for context
            category = CategoryAnalyzer.detect(before)
            matched = self.knowledge_registry.match(before, after, category=category.value)
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
        else:
            # Check for System Implication (Subset of solutions)
            if self.symbolic_engine.is_system(before):
                if self.symbolic_engine.check_implication(before, after):
                    valid = True
                    result["valid"] = True
                    result["status"] = "ok" # or "implication"
                    details["explanation"] = "Step is implied by the system."
                    details["reason"] = "implication"
                    # Do NOT update self._current_expr, so subsequent steps are checked against the System
                    return result

            # Check for "Pending Bounds" (Intermediate Indefinite Step)
            # If problem is Definite Integral and step is Indefinite Integral (contains variables)
            # and diff(step) == integrand, then it's a valid intermediate step.
            try:
                # 1. Is the problem a Definite Integral?
                # We need to check the original problem expression, not just the current state.
                # But self._current_expr tracks the *current* state.
                # If we are in the middle of a calculation, _current_expr might already be partially evaluated.
                # However, the user requirement implies checking against the *problem* context.
                # For now, let's check if the *current* expression is a definite integral.
                # Or, more robustly: check if 'after' is the antiderivative of 'before's integrand.
                
                # Heuristic:
                # before: Integral(f, (x, a, b))  (Definite)
                # after: F(x)                     (Indefinite)
                # Check: diff(after, x) == f
                
                # We need access to symbolic engine's advanced methods.
                # Let's assume symbolic_engine has a method or we use raw sympy if available.
                # Since we are in SymbolicEvaluationEngine, we can use self.symbolic_engine.
                
                # This requires 'before' to be a definite integral.
                # We can try to parse 'before' to see if it's a definite integral.
                # But we don't have easy access to AST here.
                # We can rely on string matching or symbolic engine helper.
                
                if "Integral" in before and ", (" in before: # Rough check for definite integral
                     # Try to verify if 'after' is the antiderivative
                     # We need to know the variable of integration.
                     # Let's ask symbolic engine to check "is_antiderivative(after, before)"
                     if hasattr(self.symbolic_engine, "is_antiderivative"):
                         is_antideriv = self.symbolic_engine.is_antiderivative(after, before)
                         if is_antideriv:
                             valid = True
                             # We don't update _current_expr because the value hasn't effectively changed 
                             # towards the *numerical* goal, but it's a valid intermediate representation.
                             # Wait, if we don't update, the next step will be compared against 'before' again.
                             # But the user wrote 'after'. The next step will likely be [F(x)]_a^b or F(b)-F(a).
                             # If we update _current_expr to 'after' (the function), 
                             # then the next step (number) will be compared to 'after' (function) -> mismatch again.
                             # This is the core issue: type mismatch (Function vs Number).
                             
                             # Solution: Mark as "partial" or "ok" but with a warning/hint.
                             # And DO NOT update _current_expr?
                             # If we don't update, the user must repeat the integral in the next step? No.
                             # If the user writes:
                             # 1. Int(x^2, 0, 1)
                             # 2. x^3/3          <-- We accept this as "Pending Bounds"
                             # 3. 1/3 - 0        <-- This should be compared against... what?
                             # If we keep state as Int(x^2, 0, 1) (value 1/3), 
                             # then 1/3 - 0 (value 1/3) is valid against Int(x^2, 0, 1).
                             # So NOT updating _current_expr seems correct for the *value* chain.
                             # But we need to tell the UI/User that step 2 was accepted.
                             
                             result["valid"] = True
                             result["status"] = "partial" # Custom status if supported, else "ok"
                             details["hint"] = "Indefinite integral is correct. Don't forget to apply the bounds!"
                             details["reason"] = "pending_bounds"
                             # We do NOT update self._current_expr
            except Exception:
                pass

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
        self.classifier = ExpressionClassifier(self.engine)
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
        force_redundant: bool = False,
    ) -> None:
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        scope_id = current_scope["id"] if current_scope else "main"
        parent_scope_id = current_scope["parent_id"] if current_scope else None
        depth = len(self._scope_stack)
        context_type = "sub_problem" if depth > 0 else "main"
        
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
                # Relax meta check: ignore small diffs or just check keys?
                # For now, strict equality is fine, but maybe 'meta' is the culprit for "Fail".
                # Let's assume strict equality is desired.
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
            context_type=context_type,
            is_redundant=force_redundant,
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
        self._completed = False
        self._fatal_error = False
        self._state = "INIT"

        for node in self.program.body:
            if isinstance(node, ast.ProblemNode):
                # Hierarchical execution
                if not self._run_problem_node(node):
                     # If sub-problem failed fatally?
                     if self._fatal_error:
                         return False
            elif isinstance(node, ast.MetaNode):
                self._handle_meta(node)
            elif isinstance(node, ast.ConfigNode):
                self._handle_config(node)
            elif isinstance(node, ast.ModeNode):
                self._handle_mode(node)
            elif isinstance(node, ast.PrepareNode):
                self._handle_prepare(node) # Global prepare (legacy/compat)
            elif isinstance(node, ast.ExplainNode):
                self._handle_explain(node)
            elif isinstance(node, ast.CounterfactualNode):
                self._handle_counterfactual(node)
            elif isinstance(node, ast.ScenarioNode):
                self._handle_scenario(node)
            elif isinstance(node, ast.SubProblemNode):
                self._handle_sub_problem(node)
            # Legacy fallback: if step/end appear globally (parser shouldn't allow this usually for new format)
            elif isinstance(node, ast.StepNode):
                 self._handle_step(node)
            elif isinstance(node, ast.EndNode):
                 self._handle_end(node)
            else:
                pass

        if self._state != "END" and not self._fatal_error:
            # If we just ran a problem node which has its own end, state might be "END" or "PROBLEM_INIT" depending on implementation.
            # But "run()" expects global end state? 
            # If multiple problems, state tracks the *last* one.
            # If processed problems, we consider completed?
            pass
        
        self._completed = True
        return not self._fatal_error and not self._has_critical_mistake

    def _run_problem_node(self, node: ast.ProblemNode) -> bool:
        """Execute a ProblemNode and its children."""
        self._handle_problem(node)
        
        # Override mode if specified
        if node.mode:
             self._mode = node.mode
        
        # Handle Prepare
        if node.prepare:
            self._handle_prepare(node.prepare)
            
        # Handle Steps
        for step in node.steps:
            if self._fatal_error: break
            if isinstance(step, ast.SubProblemNode):
                self._handle_sub_problem(step)
            elif isinstance(step, ast.ExplainNode):
                self._handle_explain(step)
            elif isinstance(step, ast.EndNode):
                self._handle_end(step)
            else:
                self._handle_step(step)
            
        return not self._fatal_error

    def _handle_problem(self, node: ast.ProblemNode) -> None:
        if self._state not in ("INIT", "END"):
            exc = MissingProblemError("Problem already defined (previous problem not finished?).")
            self._fatal(
                phase="problem",
                expression=node.expr,
                rendered=f"Duplicate problem: {node.expr}",
                exc=exc,
            )
        # Reset run-scoped state for new problem
        self._state = "INIT" 
        
        try:
            self.engine.set(node.expr)
        except CoherentError as exc:
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
            meta={"scope": node.name} if node.name else None,
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
        except CoherentError as exc:
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
        # Check for no-op (redundant step)
        is_noop = False
        if is_valid and result.get("before") == result.get("after"):
            is_noop = True

        self._log(
            phase="step",
            expression=node.raw_expr or node.expr,
            rendered=f"Step ({node.step_id or 'unnamed'}): {node.raw_expr or node.expr}",
            status=status,
            rule_id=result.get("rule_id"),
            meta=meta,
            force_redundant=is_noop,
        )

        # Log fuzzy details if available (restoring behavior for CLI/Tests)
        details = result.get("details", {})
        if "fuzzy_score" in details:
             label = details.get("fuzzy_label", "unknown")
             score = details.get("fuzzy_score", 0.0)
             # Reconstruct a metadata object similar to what tests expect
             fuzzy_meta = {
                 "label": label,
                 "score": {"combined_score": score},
             }
             if "fuzzy_debug" in details:
                 fuzzy_meta["debug"] = details["fuzzy_debug"]
             
             self._log(
                phase="fuzzy",
                expression=node.raw_expr or node.expr,
                rendered=f"Fuzzy: {label} ({score:.2f})",
                status="ok",
                meta=fuzzy_meta,
            )

        if status == "mistake":
            self._has_mistake = True
            if meta.get("critical"):
                self._has_critical_mistake = True
        if is_valid:
            self._last_expr_raw = node.raw_expr or node.expr
            self._state = "STEP_RUN"
            return
        
        # Fuzzy logic is now integrated into engine.check_step
        # If we reached here, the step is invalid despite all checks.
        pass

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
        except CoherentError as exc:
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
            target_variable = ctx.get("target_variable")
            final_sub_result = result["after"]

            if target_variable:
                # Variable Binding Mode: store result and restore parent untouched.
                try:
                    self.engine.set_variable(target_variable, final_sub_result)
                except Exception:
                    pass
                new_parent_expr = parent_expr
            else:
                # Legacy Mode (Inline Replacement)
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

            if target_variable:
                self._log(
                    phase="sub_problem_end",
                    expression=final_sub_result,
                    rendered=f"Sub-problem stored: {target_variable} = {final_sub_result}",
                    status="ok",
                    meta={"target_variable": target_variable},
                )
            else:
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
            result_str = self._symbolic_engine.to_string(result)
            self._log(
                phase="counterfactual",
                expression=node.expect,
                rendered=f"Counterfactual: expect {node.expect} -> {result_str}",
                status="ok",
                meta={"assume": node.assume, "result": result_str},
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
        
        # Stringify context values for logging to avoid JSON serialization issues
        log_context = {k: self._symbolic_engine.to_string(v) for k, v in context.items()}
        
        self._log(
            phase="scenario",
            expression=None,
            rendered=f"Scenario added: {node.name}",
            status="ok",
            meta={"context": log_context},
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

        # If target_variable is set, this is a "Variable Binding" sub-problem (independent calculation).
        # We skip the is_subexpression check.
        if not node.target_variable:
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
            "target_sub_expr": node.expr,
            "target_variable": node.target_variable,  # Store target variable
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
            expression=f"{node.target_variable} = {node.expr}" if node.target_variable else getattr(node, "raw_expr", node.expr),
            rendered=f"Sub-problem: {node.target_variable} = {node.expr}" if node.target_variable else f"Sub-problem: {getattr(node, 'raw_expr', node.expr)}",
            status="ok"
        )
        self._state = "PROBLEM_SET"


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
