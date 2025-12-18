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
# Extensions are imported lazily
# from .stats_engine import StatsEngine
# from .trig_engine import TrigHelper
# from .calculus_engine import CalculusEngine
# from .linear_algebra_engine import LinearAlgebraEngine
from .classifier import ExpressionClassifier
from .category_identifier import CategoryIdentifier
from .math_category import MathCategory
from .renderers import RenderingEngine
from causalscript.core.errors import CausalScriptError, InvalidExprError, MissingProblemError

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
        if hasattr(self.validation_engine, 'fuzzy_judge') and self.validation_engine.fuzzy_judge:
            self.validation_engine.fuzzy_judge.symbolic_engine = computation_engine.symbolic_engine
        self.hint_engine = hint_engine
        self.exercise_spec = exercise_spec
        self.learning_logger = learning_logger or LearningLogger()
        self.knowledge_registry = knowledge_registry
        self.hint_persona = hint_persona
        
        # Apply decision config if provided and fuzzy judge exists
        if decision_config and hasattr(self.validation_engine, 'fuzzy_judge') and self.validation_engine.fuzzy_judge:
             self.validation_engine.fuzzy_judge.decision_config = decision_config
             # Re-init decision engine
             from causalscript.core.decision_theory import DecisionEngine
             self.validation_engine.fuzzy_judge.decision_engine = DecisionEngine(decision_config)

        self.function_analyzer = FunctionAnalyzer(computation_engine)
        self.function_analyzer = FunctionAnalyzer(computation_engine)
        
        # Extensions (Lazy Loaded)
        self._extensions: Dict[str, Any] = {}
        
        self.classifier = ExpressionClassifier(computation_engine.symbolic_engine)
        self.category_identifier = CategoryIdentifier(computation_engine.symbolic_engine)
        
        self._current_expr: str | None = None
        self._current_domains: list[str] = []
        self._current_category: MathCategory = MathCategory.ALGEBRA
        self._context: Dict[str, Any] = {}
        self._scenarios: Dict[str, Dict[str, Any]] = {}
        self._equation_mode: bool = False
        
        # Initialize Rendering Engine
        self.rendering_engine = RenderingEngine(computation_engine.symbolic_engine)

    def get_extension(self, name: str) -> Any:
        """
        Get a computation extension by name, loading it if necessary.
        
        Args:
            name: The name of the extension (e.g., 'calculus', 'linear_algebra')
            
        Returns:
            The requested extension instance.
        """
        if name not in self._extensions:
            self._extensions[name] = self._load_extension(name)
        return self._extensions[name]

    def _load_extension(self, name: str) -> Any:
        """Factory for loading extensions."""
        if name == "calculus":
            from .calculus_engine import CalculusEngine
            return CalculusEngine(self.computation_engine)
        elif name == "linear_algebra":
            from .linear_algebra_engine import LinearAlgebraEngine
            return LinearAlgebraEngine()
        elif name == "stats":
            from .stats_engine import StatsEngine
            return StatsEngine()
        elif name == "trig":
            from .trig_engine import TrigHelper
            return TrigHelper()
        else:
            raise ValueError(f"Unknown extension: {name}")

    @property
    def calculus_engine(self) -> Any:
        return self.get_extension("calculus")

    @property
    def linear_algebra(self) -> Any:
        return self.get_extension("linear_algebra")

    @property
    def stats_engine(self) -> Any:
        return self.get_extension("stats")
        
    @property
    def trig_helper(self) -> Any:
        return self.get_extension("trig")

    def _normalize_expression(self, expr: str) -> str:
        """
        Convert optional equation syntax (left = right) into an expression form.
        """
        expr = expr.strip()
        equation = self._extract_equation_sides(expr)
        if not equation:
            return expr
        left, right = equation
        return f"({left}) - ({right})"

    def _extract_equation_sides(self, expr: str) -> tuple[str, str] | None:
        """
        Return (lhs, rhs) if the expression represents an equation, either via
        explicit '=' or SymPy's Eq(...).
        """
        stripped = expr.strip()
        if "=" in stripped:
            left, _, right = stripped.partition("=")
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
            return None

        if stripped.startswith("Eq(") and stripped.endswith(")"):
            inner = stripped[3:-1]
            parts: list[str] = []
            buffer: list[str] = []
            depth = 0
            for ch in inner:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(depth - 1, 0)
                if ch == "," and depth == 0:
                    parts.append("".join(buffer).strip())
                    buffer = []
                    continue
                buffer.append(ch)
            if buffer:
                parts.append("".join(buffer).strip())
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]

        try:
            from sympy import Eq  # type: ignore

            internal = self.computation_engine.symbolic_engine.to_internal(stripped)
            if isinstance(internal, Eq):
                return str(internal.lhs), str(internal.rhs)
        except Exception:
            return None

        return None

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
        
        # Auto-detect and store category
        self._current_category = self.computation_engine.detect_category(self._current_expr)
        self._current_domains = [self._current_category.value]
        
        # Propagate context to SymbolicEngine
        self.computation_engine.symbolic_engine.set_context([self._current_category])
        
        print(f"DEBUG: Detected Category: {self._current_category.value}")
        
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
        equation = self._extract_equation_sides(expr)
        lhs = rhs = None
        if equation:
            lhs, rhs = equation
        self._equation_mode = bool(equation)
        after = self._normalize_expression(expr)
        # Default validation (Integrated Pipeline)
        # Apply context if variables are bound
        # We pass raw strings effectively, normalization handled by validate_step if necessary (or passed normalized)
        # CoreRuntime normalizes before calling check_step usually? No, check_step receives raw 'expr'.
        # We compute 'before' and 'after' here.
        
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

        if equation:
            self._equation_mode = True
            # Check for "LHS = RHS" where LHS is equivalent to Before
            # Apply context if needed
            if self._context:
                try:
                    lhs_eval = self.computation_engine.substitute(lhs, self._context) if lhs is not None else None
                    rhs_eval = self.computation_engine.substitute(rhs, self._context) if rhs is not None else None
                except CausalScriptError:
                    lhs_eval = lhs
                    rhs_eval = rhs
            else:
                lhs_eval = lhs
                rhs_eval = rhs

            is_lhs_equiv = False
            if lhs is not None:
                is_lhs_equiv = self.computation_engine.symbolic_engine.is_equiv(before_eval, lhs_eval, context=self._context)
            
            if is_lhs_equiv and rhs is not None:
                # Check if LHS == RHS
                is_eqn_valid = self.computation_engine.symbolic_engine.is_equiv(lhs_eval, rhs_eval, context=self._context)
                if not is_eqn_valid and self._equation_mode:
                    is_eqn_valid = self._expressions_equivalent_up_to_scalar(lhs_eval, rhs_eval)
                
                if is_eqn_valid:
                    # This is a valid transition: Before -> LHS -> RHS
                    # We treat RHS as the new state
                    # We override 'after' to be RHS for the result
                    after = rhs

        # Prepare context for validation
        validation_context = self._context.copy() if self._context else None
        
        # Delegate to ValidationEngine
        val_result = self.validation_engine.validate_step(before, after, context=validation_context)

        
        is_valid = val_result["valid"]
        status = val_result["status"]
        
        # Fallback: Equation Scalar Equivalence (e.g., dividing both sides by constant)
        if not is_valid and self._equation_mode:
             if self._expressions_equivalent_up_to_scalar(before_eval, after_eval):
                 is_valid = True
                 status = "correct"
        
        
        # Scenario validation (Keep in CoreRuntime for now)
        scenario_results = {}
        is_valid_scenarios = True
        
        # Initialize partial flags early
        is_partial = False
        is_partial_attempt = False

        if self._scenarios:
            scenario_results = self.computation_engine.check_equivalence_in_scenarios(
                before, after, self._scenarios
            )
            is_valid_scenarios = all(scenario_results.values())
            
            if not is_valid and self._scenarios:
                 if is_valid_scenarios:
                     is_valid = True
                     status = "correct" # Override status

        # Implication Logic (System -> Solution)
        # If still invalid, check if 'after' is implied by 'before'
        if not is_valid:
            try:
                # Construct a target string that preserves Equation structure 
                # (instead of 'after' which might be normalized to implicit subtraction)
                implication_target = after
                if equation and lhs is not None and rhs is not None:
                     # Reconstruct Eq for Symbolic Engine
                     implication_target = f"Eq({lhs}, {rhs})"

                if self.computation_engine.symbolic_engine.is_implied_by_system(implication_target, before):
                    is_valid = True
                    status = "partial" # Or "implication"
                    # We accept it, but we should treat it as partial because information might be lost (e.g. system -> single var)
                    is_partial = True 
                    # If we treat it as partial, current_expr won't be updated, preserving the system context.
                    # This is likely what we want for "x=4" derived from a system.
                    result["details"]["note"] = "Step is implied by the previous system."
            except Exception:
                pass

        # Partial Calculation Logic (Override invalid)
        if not is_valid and equation:
            # Check if this is a valid partial calculation
            lhs, rhs = equation
            try:
                lhs_in_before = self.computation_engine.symbolic_engine.is_subexpression(lhs, before) if lhs is not None else False
            except Exception:
                lhs_in_before = False

            try:
                lhs_rhs_equiv = self.computation_engine.symbolic_engine.is_equiv(lhs, rhs, context=self._context) if lhs is not None and rhs is not None else False
            except Exception:
                lhs_rhs_equiv = False

            if lhs_in_before:
                is_partial_attempt = True

            try:
                if lhs_rhs_equiv and lhs_in_before:
                    is_valid = True
                    is_partial = True
                    status = "partial"
            except Exception:
                is_partial = False

        result = {
            "before": before,
            "after": after,
            "valid": is_valid,
            "rule_id": None,
            "details": val_result.get("details", {}).copy(),
        }
        
        # Merge status into details if useful
        result["details"]["status"] = status
        
        # Determine if symbolic validity holds (for precision logic)
        # If validate_step returns valid without fuzzy details, it considered it symbolically valid.
        is_valid_symbolic = is_valid and ("fuzzy_score" not in val_result.get("details", {}))

        # === Improved Precision Logic ===
        
        # 1. Calculate expected formula for logging
        expected_expr = None
        try:
            # Simplify 'before' to get the mathematically "correct" next state or value
            expected_expr = self.computation_engine.simplify(before)
            result["details"]["expected_formula"] = expected_expr
        except Exception:
            pass

        # 2. Check for mathematical errors regardless of final "valid" status (which might be fuzzy-ok)
        if not is_valid_symbolic and not is_partial:
            # If not symbolically valid (and not a valid partial step), mark as math error
            # This applies even if fuzzy logic saved it (is_valid=True)
            result["details"]["mathematical_error"] = True
            
            # Generate hint targeting the correction
            # target=before is used so the hint mechanism sees the transition from Before -> After
            # Generate hint targeting the correction
            # target=before is used so the hint mechanism sees the transition from Before -> After
            hint = self.hint_engine.generate_hint(
                after, 
                before, 
                persona=self.hint_persona,
                validation_details=result["details"]
            )
            
            feedback_info = {
                "message": hint.message,
                "type": hint.hint_type,
                "correction": f"Should be close to: {expected_expr}" if expected_expr else None
            }

            if is_valid:
                # Case A: Accepted by Fuzzy Judge (or Scenarios), but has symbolic error
                # We add a correction notice so the user/LLM knows it wasn't perfect
                result["details"]["correction_notice"] = feedback_info
                result["details"]["evaluation_note"] = "Accepted by fuzzy match, but contains symbolic error."
            else:
                # Case B: Not valid at all
                # Provide the feedback as the main hint
                result["details"]["hint"] = feedback_info

        # Calculate evaluated form for UI display
        try:
            evaluated_form = self.computation_engine.simplify(after)
            result["details"]["evaluated"] = evaluated_form
        except Exception:
            result["details"]["evaluated"] = after

        if is_valid and self.knowledge_registry:
            # Use detected category
            current_cat = self._current_category.value if self._current_category else None
            
            rule_node = self.knowledge_registry.match(
                before, 
                after, 
                category=current_cat
            )
            if rule_node:
                result["rule_id"] = rule_node.id
                result["details"]["rule"] = rule_node.to_metadata()
        
        # Include category in result metadata (for rendering)
        result["details"]["category"] = self._current_category.value
        
        if self._scenarios:
            result["details"]["scenarios"] = scenario_results
            
            # The decision details are already in result["details"] due to merging val_result["details"]
            
            # Additional processing for REVIEW status?
            # If status=="review", we might want to ensure it is treated as "valid but check hint"
            if status == "review":
                result["details"]["review_needed"] = True
                
            if is_valid and not is_partial and status not in ["correct", "partial"]:
                # If valid via fuzzy (e.g. status="correct" from fuzzy decision?), suggest corrected form
                # status might be "correct" even if fuzzy.
                # Check fuzzy_score existence
                if "fuzzy_score" in result["details"]:
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
            # This hint generation is now handled by the improved precision logic if it's a mathematical error
            if "hint" not in result["details"]:
                hint = self.hint_engine.generate_hint(
                    after, 
                    before, 
                    persona=self.hint_persona,
                    validation_details=result["details"]
                )
                result["details"]["hint"] = {
                    "message": hint.message,
                    "type": hint.hint_type,
                    "details": hint.details
                }
            
        # Render the result for display
        self.rendering_engine.render_result(result)
        
        print(f"DEBUG: Step '{expr}' -> Valid: {is_valid}, Partial: {is_partial}, Before: '{before}', After: '{after}'")
        return result

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generates a report on the optimization strategy and classification results.
        """
        # Re-identify to get the structured result if needed, or rely on stored state
        # For now, we use the stored domains but we could expose the full CategoryResult
        return {
            "classification": self._current_domains,
            "symbolic_engine_mode": "optimized" if "calculus" in self._current_domains else "standard",
            "rule_matching_scope": self._current_domains,
            "report_rendering_strategy": "latex_enhanced" if "calculus" in self._current_domains or "linear_algebra" in self._current_domains else "standard"
        }

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
                

            
            self.rendering_engine.render_result(result)
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
                result = {
                    "before": self._current_expr,
                    "after": normalized_expr,
                    "valid": is_valid,
                    "rule_id": None,
                    "details": {}
                }
                self.rendering_engine.render_result(result)
                return result
            else:
                result = {
                    "before": self._current_expr,
                    "after": self._current_expr,
                    "valid": True,
                    "rule_id": None,
                    "details": {}
                }
                self.rendering_engine.render_result(result)
                return result
