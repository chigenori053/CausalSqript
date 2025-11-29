"""Core Runtime for MathLang."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, Optional

from .evaluator import Engine
from .computation_engine import ComputationEngine
from .validation_engine import ValidationEngine
from .hint_engine import HintEngine
from .exercise_spec import ExerciseSpec
from .learning_logger import LearningLogger
from .knowledge_registry import KnowledgeRegistry
from .function_analysis import FunctionAnalyzer
from .stats_engine import StatsEngine
from .trig_engine import TrigHelper
from .calculus_engine import CalculusEngine
from .linear_algebra_engine import LinearAlgebraEngine
from core.errors import CausalScriptError, InvalidStepError, MissingProblemError

_EQUATION_SAMPLE_ASSIGNMENTS = [
    {"x": -2, "y": 1, "z": 3, "a": 1},
    {"x": 0, "y": 0, "z": 0, "a": 2, "b": -1},
    {"x": 1, "y": 2, "z": -1, "b": 3, "c": 4},
    {"a": 2, "b": 5, "c": -3},
    {},
]


class CoreRuntime(Engine):
    """
    Orchestrates the execution of MathLang programs using specialized engines.
    
    Integrates:
    - ComputationEngine: For symbolic/numeric evaluation
    - ValidationEngine: For answer checking
    - HintEngine: For generating feedback
    """


    def __init__(
        self,
        computation_engine: ComputationEngine,
        validation_engine: ValidationEngine,
        hint_engine: HintEngine,
        exercise_spec: Optional[ExerciseSpec] = None,
        learning_logger: Optional[LearningLogger] = None,
        knowledge_registry: Optional[KnowledgeRegistry] = None,
        decision_config: Optional[Any] = None, # DecisionConfig
        hint_persona: str = "balanced",
    ):
        """
        Initialize the CoreRuntime.
        
        Args:
            computation_engine: Engine for computation
            validation_engine: Engine for validation
            hint_engine: Engine for hints
            exercise_spec: Optional specification for the current exercise
            learning_logger: Optional logger for learning analytics
            decision_config: Optional DecisionConfig for fuzzy judge
            hint_persona: Persona for hint generation ("balanced", "sparta", "support")
        """
        self.computation_engine = computation_engine
        self.validation_engine = validation_engine
        self.hint_engine = hint_engine
        self.exercise_spec = exercise_spec
        self.learning_logger = learning_logger or LearningLogger()
        self.knowledge_registry = knowledge_registry
        self.hint_persona = hint_persona
        
        # Apply decision config if provided and fuzzy judge exists
        if decision_config and hasattr(self.validation_engine, 'fuzzy_judge') and self.validation_engine.fuzzy_judge:
             self.validation_engine.fuzzy_judge.decision_config = decision_config
             # Re-init decision engine
             from core.decision_theory import DecisionEngine
             self.validation_engine.fuzzy_judge.decision_engine = DecisionEngine(decision_config)

        self.function_analyzer = FunctionAnalyzer(computation_engine)
        self.stats_engine = StatsEngine()
        self.trig_helper = TrigHelper()
        self.calculus_engine = CalculusEngine(computation_engine)
        self.linear_algebra = LinearAlgebraEngine()
        
        self._current_expr: str | None = None
        self._context: Dict[str, Any] = {}
        self._scenarios: Dict[str, Dict[str, Any]] = {}
        self._equation_mode: bool = False

    def _normalize_expression(self, expr: str) -> str:
        """
        Convert optional equation syntax (left = right) into an expression form.
        """
        expr = expr.strip()
        if "=" not in expr:
            return expr
        left, _, right = expr.partition("=")
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError("Invalid equation format. Expected 'left = right'.")
        return f"({left}) - ({right})"

    def _expressions_equivalent_up_to_scalar(self, expr1: str, expr2: str) -> bool:
        sym = self.computation_engine.symbolic_engine
        
        # Check if both are constants first
        # If they are constants, they must be equal (ratio ~ 1)
        # We don't want "1.0 is equivalent to 0.25"
        try:
            c1 = self._try_constant_value(sym.simplify(expr1))
            c2 = self._try_constant_value(sym.simplify(expr2))
            if c1 is not None and c2 is not None:
                if abs(c2) < 1e-12:
                    return abs(c1) < 1e-12
                ratio = c1 / c2
                return abs(ratio - 1.0) < 1e-9
        except Exception:
            pass

        ratio_expr = f"(({expr1})) / (({expr2}))"
        if sym.has_sympy():
            try:
                ratio_internal = sym.to_internal(ratio_expr)
                free_symbols = getattr(ratio_internal, "free_symbols", None)
                if free_symbols is not None and len(free_symbols) == 0:
                    # If it evaluates to a constant, check if it's not zero/infinite
                    # But be careful with float precision
                    pass
            except Exception:
                pass
        
        # We used to simplify here, but it caused issues with near-zero denominators (e.g. 1 / 1e-16)
        # resulting in large constants that looked like valid scalars.
        # We will rely on numeric sampling which handles zero-checks better.
        
        return self._numeric_scalar_equiv(expr1, expr2)

    def _try_constant_value(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            lowered = cleaned.lower()
            if lowered in {"nan", "zoo", "oo"}:
                return None
            if any(ch.isalpha() for ch in cleaned):
                return None
            try:
                return float(Fraction(cleaned))
            except (ValueError, ZeroDivisionError):
                try:
                    return float(cleaned)
                except ValueError:
                    return None
        return None

    def _numeric_scalar_equiv(self, expr1: str, expr2: str) -> bool:
        ratios: list[float] = []
        for assignment in _EQUATION_SAMPLE_ASSIGNMENTS:
            try:
                val1 = self.computation_engine.numeric_eval(expr1, assignment)
                val2 = self.computation_engine.numeric_eval(expr2, assignment)
            except Exception:
                continue
            if isinstance(val1, dict) or isinstance(val2, dict):
                continue
            try:
                val1_float = float(val1)
                val2_float = float(val2)
            except Exception:
                continue
            if abs(val2_float) < 1e-12:
                continue
            ratio = val1_float / val2_float
            if abs(ratio) < 1e-12:
                continue
            ratios.append(ratio)
            if len(ratios) > 1 and abs(ratio - ratios[0]) > 1e-6:
                return False
        return bool(ratios)

    def set(self, expr: str) -> None:
        """
        Set the initial problem expression.
        
        Supports both expressions and equations:
        - Expression: "x**2 + 2*x + 1"
        - Equation: "2*x + 3 = 7" (converted internally to "2*x + 3 - (7)")
        
        Args:
            expr: The problem expression or equation string
        """
        self._equation_mode = "=" in expr
        self._current_expr = self._normalize_expression(expr)
        
    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the execution context.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self._context[name] = value
        self.computation_engine.bind(name, value)

    def add_scenario(self, name: str, context: Dict[str, Any]) -> None:
        """
        Add a scenario with a specific context.
        
        Args:
            name: Scenario name
            context: Variable assignments for this scenario
        """
        self._scenarios[name] = context

    def evaluate(self, expr: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate an expression.
        
        Args:
            expr: Expression to evaluate
            context: Optional temporary context
            
        Returns:
            Evaluation result
        """
        return self.computation_engine.numeric_eval(expr, context)

    def check_step(self, expr: str) -> dict:
        """
        Check if a step is valid (equivalent to the previous expression).
        
        Args:
            expr: The new expression for this step
            
        Returns:
            Dictionary containing validation results and metadata
        """
        if self._current_expr is None:
            raise MissingProblemError("Problem expression must be set before steps.")
            
        before = self._current_expr
        after = self._normalize_expression(expr)
        # Default validation (symbolic)
        # Apply context if variables are bound
        if self._context:
            try:
                before_eval = self.computation_engine.substitute(before, self._context)
                after_eval = self.computation_engine.substitute(after, self._context)
            except CausalScriptError as e:
                before_eval = before
                after_eval = after
        else:
            before_eval = before
            after_eval = after
            
        is_valid_symbolic = False

        if "=" in expr:
            self._equation_mode = True
            # Check for "LHS = RHS" where LHS is equivalent to Before
            lhs, _, rhs = expr.partition("=")
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Check if LHS == Before
            # We need to apply context if needed
            if self._context:
                try:
                    lhs_eval = self.computation_engine.substitute(lhs, self._context)
                    rhs_eval = self.computation_engine.substitute(rhs, self._context)
                except CausalScriptError:
                    lhs_eval = lhs
                    rhs_eval = rhs
            else:
                lhs_eval = lhs
                rhs_eval = rhs

            is_lhs_equiv = self.computation_engine.symbolic_engine.is_equiv(before_eval, lhs_eval)
            # We do NOT use scalar equivalence here. LHS must be strictly equivalent to Before
            # to justify replacing Before with RHS as the new state.
            
            if is_lhs_equiv:
                # Check if LHS == RHS
                is_eqn_valid = self.computation_engine.symbolic_engine.is_equiv(lhs_eval, rhs_eval)
                if not is_eqn_valid and self._equation_mode:
                    is_eqn_valid = self._expressions_equivalent_up_to_scalar(lhs_eval, rhs_eval)
                
                if is_eqn_valid:
                    # This is a valid transition: Before -> LHS -> RHS
                    # We treat RHS as the new state
                    # We override 'after' to be RHS for the result
                    after = rhs
                    is_valid_symbolic = True
                    # Skip the standard check since we already verified it
        
        if not is_valid_symbolic:
             is_valid_symbolic = self.computation_engine.symbolic_engine.is_equiv(before_eval, after_eval)
             if not is_valid_symbolic and self._equation_mode:
                 is_valid_symbolic = self._expressions_equivalent_up_to_scalar(before_eval, after_eval)
        
        # Scenario validation
        scenario_results = {}
        is_valid_scenarios = True
        if self._scenarios:
            scenario_results = self.computation_engine.check_equivalence_in_scenarios(
                before, after, self._scenarios
            )
            is_valid_scenarios = all(scenario_results.values())
        
        is_valid = is_valid_symbolic
        if not is_valid and self._scenarios:
            is_valid = is_valid_scenarios

        # Partial Calculation Logic
        is_partial = False
        is_partial_attempt = False
        if not is_valid and "=" in expr:
            # Check if this is a valid partial calculation
            # 1. It must be a valid equation itself (LHS == RHS)
            lhs, _, rhs = expr.partition("=")
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Evaluate LHS and RHS to check equality
            try:
                lhs_in_before = self.computation_engine.symbolic_engine.is_subexpression(lhs, before)
            except Exception:
                lhs_in_before = False

            try:
                lhs_rhs_equiv = self.computation_engine.symbolic_engine.is_equiv(lhs, rhs)
            except Exception:
                lhs_rhs_equiv = False

            if lhs_in_before:
                is_partial_attempt = True

            try:
                if lhs_rhs_equiv and lhs_in_before:
                    is_valid = True
                    is_partial = True
            except Exception:
                is_partial = False

        # Fuzzy Validation Fallback
        fuzzy_result = None
        if not is_valid and hasattr(self.validation_engine, 'fuzzy_judge') and self.validation_engine.fuzzy_judge:
            try:
                encoder = self.validation_engine.fuzzy_judge.encoder
                # Use 'before' as a proxy for problem/previous since we are checking step validity
                norm_before = encoder.normalize(before)
                norm_after = encoder.normalize(after)
                
                fuzzy_result = self.validation_engine.fuzzy_judge.judge_step(
                    problem_expr=norm_before,
                    previous_expr=norm_before,
                    candidate_expr=norm_after
                )
                
                from core.fuzzy.types import FuzzyLabel
                if fuzzy_result.label in [FuzzyLabel.EXACT, FuzzyLabel.EQUIVALENT, FuzzyLabel.APPROX_EQ, FuzzyLabel.ANALOGOUS]:
                    is_valid = True
            except Exception:
                pass
        
        result = {
            "before": before,
            "after": after,
            "valid": is_valid,
            "rule_id": None,
            "details": {},
        }

        # Calculate evaluated form for UI display
        try:
            evaluated_form = self.computation_engine.simplify(after)
            result["details"]["evaluated"] = evaluated_form
        except Exception:
            result["details"]["evaluated"] = after

        if is_valid and self.knowledge_registry:
            rule_node = self.knowledge_registry.match(before, after)
            if rule_node:
                result["rule_id"] = rule_node.id
                result["details"]["rule"] = rule_node.to_metadata()
        
        if self._scenarios:
            result["details"]["scenarios"] = scenario_results
            
        if fuzzy_result:
            result["details"]["fuzzy_label"] = fuzzy_result.label.value
            result["details"]["fuzzy_score"] = fuzzy_result.score.combined_score

            if "decision_action" in fuzzy_result.debug:
                result["details"]["decision_action"] = fuzzy_result.debug["decision_action"]
            if "decision_utility" in fuzzy_result.debug:
                result["details"]["decision_utility"] = fuzzy_result.debug["decision_utility"]
            if "decision_utils" in fuzzy_result.debug:
                result["details"]["decision_utils"] = fuzzy_result.debug["decision_utils"]

            if is_valid and not is_valid_symbolic and not is_partial:
                # If valid via fuzzy but not symbolic, suggest the 'before' state as the corrected form
                # or simply note that it was fuzzy matched.
                result["details"]["corrected_form"] = before

        if is_valid:
            if is_partial:
                # Do NOT update current_expr for partial steps
                result["details"]["partial"] = True
            else:
                self._current_expr = after
        else:
            if is_partial_attempt:
                # Treat incorrect partial calculations as critical mistakes
                result["details"]["critical"] = True
            # Generate hint if invalid
            # Use the previous expression as the target for the hint
            hint = self.hint_engine.generate_hint(after, before, persona=self.hint_persona)
            result["details"]["hint"] = {
                "message": hint.message,
                "type": hint.hint_type,
                "details": hint.details
            }
            
        print(f"DEBUG: Step '{expr}' -> Valid: {is_valid}, Partial: {is_partial}, Before: '{before}', After: '{after}'")
        return result

    def analyze_function(self, expr: str, variable: str = "x") -> Dict[str, Any]:
        """
        Analyze a function for domain, range, intercepts, and behavior.

        Args:
            expr: Function expression to analyze.
            variable: Variable name to treat as the independent variable.

        Returns:
            Dictionary representation of FunctionAnalysisResult.
        """
        result = self.function_analyzer.analyze(expr, variable)
        return {
            "expression": result.expression,
            "variable": result.variable,
            "domain": result.domain,
            "range": result.range,
            "intercepts": result.intercepts,
            "behavior": result.behavior,
            "samples": result.samples,
        }

    def plot_function(
        self,
        expr: str,
        variable: str = "x",
        start: float | None = None,
        end: float | None = None,
        num_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate plot data (and a matplotlib figure when available) for a function.
        """
        return self.function_analyzer.plot(expr, variable, start, end, num_points)

    def describe_dataset(self, data: list[float]) -> Dict[str, Any]:
        """Return descriptive statistics for the provided sequence."""
        return self.stats_engine.describe(data)

    def compute_probability(
        self,
        distribution: str,
        value: float,
        *,
        kind: str = "continuous",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate pdf/cdf for the specified distribution."""
        result = self.stats_engine.distribution_info(
            distribution, value, kind=kind, params=params
        )
        return {
            "distribution": result.distribution,
            "value": result.value,
            "params": result.params,
            "pdf": result.pdf,
            "cdf": result.cdf,
        }

    def visualize_dataset(self, data: list[float], bins: int = 10) -> Dict[str, Any]:
        """Build histogram data and optional matplotlib figure."""
        return self.stats_engine.visualize(data, bins=bins)

    # ------------------------------------------------------------------
    # Trigonometry helpers
    # ------------------------------------------------------------------

    def trig_unit_circle(self, angle: float, unit: str = "degrees") -> Dict[str, float]:
        return self.trig_helper.unit_circle_point(angle, unit)

    def trig_standard_angle(self, angle: int) -> Dict[str, float] | None:
        return self.trig_helper.standard_angle(angle)

    def trig_evaluate(self, func: str, angle: float, unit: str = "radians") -> float:
        return self.trig_helper.evaluate(func, angle, unit)

    def trig_apply_identity(
        self,
        identity: str,
        angle: float,
        unit: str = "radians",
        other_angle: float | None = None,
        other_unit: str | None = None,
    ) -> float:
        return self.trig_helper.apply_identity(
            identity,
            angle,
            unit,
            other_angle=other_angle,
            other_unit=other_unit,
        )

    # ------------------------------------------------------------------
    # Calculus helpers
    # ------------------------------------------------------------------

    def calc_derivative(self, expr: str, variable: str = "x", at: float | None = None) -> Dict[str, Any]:
        return self.calculus_engine.derivative(expr, variable, at=at)

    def calc_integral(
        self,
        expr: str,
        variable: str = "x",
        lower: float | None = None,
        upper: float | None = None,
    ) -> Dict[str, Any]:
        return self.calculus_engine.integral(expr, variable, lower=lower, upper=upper)

    def calc_slope_of_tangent(self, expr: str, variable: str, at: float) -> float | None:
        return self.calculus_engine.slope_of_tangent(expr, variable, at)

    def calc_area_under_curve(self, expr: str, variable: str, lower: float, upper: float) -> float | None:
        return self.calculus_engine.area_under_curve(expr, variable, lower, upper)

    # ------------------------------------------------------------------
    # Linear algebra helpers
    # ------------------------------------------------------------------

    def vector_add(self, v1: list[float], v2: list[float]) -> list[float]:
        return self.linear_algebra.vector_add(v1, v2)

    def vector_dot(self, v1: list[float], v2: list[float]) -> float:
        return self.linear_algebra.dot(v1, v2)

    def vector_cross(self, v1: list[float], v2: list[float]) -> list[float]:
        return self.linear_algebra.cross(v1, v2)

    def matrix_add(self, m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]:
        return self.linear_algebra.matrix_add(m1, m2)

    def matrix_multiply(self, m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]:
        return self.linear_algebra.matrix_multiply(m1, m2)

    def matrix_transpose(self, matrix: list[list[float]]) -> list[list[float]]:
        return self.linear_algebra.matrix_transpose(matrix)

    def matrix_determinant(self, matrix: list[list[float]]) -> float:
        return self.linear_algebra.determinant(matrix)

    def solve_linear_system(self, coefficients: list[list[float]], constants: list[float]) -> list[float]:
        return self.linear_algebra.solve_linear_system(coefficients, constants)

    def matrix_eigenvalues(self, matrix: list[list[float]]) -> list[float]:
        return self.linear_algebra.eigenvalues(matrix)

    def matrix_eigenvectors(self, matrix: list[list[float]]) -> list[list[float]]:
        return self.linear_algebra.eigenvectors(matrix)

    def finalize(self, expr: str | None) -> dict:
        """
        Finalize the problem and validate the answer.
        
        Args:
            expr: The final answer expression (or None to use current)
            
        Returns:
            Dictionary containing validation results
        """
        if self._current_expr is None:
            raise MissingProblemError("Cannot finalize before a problem is declared.")
            
        final_expr = expr if expr is not None else self._current_expr
        if expr is not None:
            final_expr = self._normalize_expression(expr)
        else:
            final_expr = self._current_expr
        
        # If we have an exercise spec, use ValidationEngine
        if self.exercise_spec:
            validation_result = self.validation_engine.check_answer(
                final_expr, self.exercise_spec
            )
            
            result = {
                "before": self._current_expr,
                "after": final_expr,
                "valid": validation_result.is_correct,
                "rule_id": None,
                "details": {
                    "message": validation_result.message,
                    "validation_details": validation_result.details
                }
            }
            
            # If incorrect, generate hint
            if not validation_result.is_correct:
                hint = self.hint_engine.generate_hint_for_spec(final_expr, self.exercise_spec, persona=self.hint_persona)
                result["details"]["hint"] = {
                    "message": hint.message,
                    "type": hint.hint_type
                }
                
            return result
            
        else:
            # Fallback to simple equivalence check if no spec
            # This behaves like the old SymbolicEvaluationEngine
            # But wait, what is the target? 
            # In finalize(expr), 'expr' is the user's final answer.
            # But usually finalize checks against a target.
            # In the old engine, finalize(expr) checked if current_expr == expr (if expr provided)
            # OR it just returned the current state.
            
            # Let's look at Evaluator logic.
            # Evaluator calls finalize(node.expr).
            # If node.expr is provided (e.g. "End: x = 5"), it checks if current state matches it.
            # If node.expr is NOT provided (e.g. "End: done"), it assumes current state is final.
            
            # Without ExerciseSpec, we can only check if the provided expr is equivalent to current state
            if expr is not None:
                normalized_expr = self._normalize_expression(expr)
                is_valid = self.computation_engine.symbolic_engine.is_equiv(self._current_expr, normalized_expr)
                return {
                    "before": self._current_expr,
                    "after": normalized_expr,
                    "valid": is_valid,
                    "rule_id": None,
                    "details": {}
                }
            else:
                return {
                    "before": self._current_expr,
                    "after": self._current_expr,
                    "valid": True,
                    "rule_id": None,
                    "details": {}
                }
