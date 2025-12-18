"""Fuzzy judge implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FuzzyThresholdConfig
from .encoder import ExpressionEncoder
from .metric import SimilarityMetric
from .types import FuzzyLabel, FuzzyResult, FuzzyScore, NormalizedExpr
from ..i18n import get_language_pack
from causalscript.core.decision_theory import DecisionConfig, DecisionEngine, DecisionAction



def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


from ..symbolic_engine import SymbolicEngine

@dataclass
class FuzzyJudge:
    encoder: ExpressionEncoder
    metric: SimilarityMetric
    thresholds: FuzzyThresholdConfig | None = None
    decision_config: DecisionConfig | None = None
    symbolic_engine: SymbolicEngine | None = None

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = FuzzyThresholdConfig()
        
        # Initialize Decision Engine
        if self.decision_config is None:
            self.decision_config = DecisionConfig()
        self.decision_engine = DecisionEngine(self.decision_config)

    def judge_step(
        self,
        *,
        problem_expr: NormalizedExpr,
        previous_expr: NormalizedExpr,
        candidate_expr: NormalizedExpr,
        applied_rule_id: str | None = None,
        candidate_rule_id: str | None = None,
        explain_text: str | None = None,
    ) -> FuzzyResult:
        v_prev = self.encoder.encode_expr(previous_expr)
        v_cand = self.encoder.encode_expr(candidate_expr)
        expr_sim = _clamp(self.metric.similarity(v_prev, v_cand))

        if applied_rule_id and candidate_rule_id:
            rule_sim = 1.0 if applied_rule_id == candidate_rule_id else 0.5
        else:
            rule_sim = 0.0

        if explain_text:
            v_text = self.encoder.encode_text(explain_text)
            text_sim = _clamp(self.metric.similarity(v_text, v_cand))
        else:
            text_sim = 0.0

        combined = (
            0.6 * expr_sim +
            0.2 * rule_sim +
            0.2 * text_sim
        )

        # --- Calculus Check (Antiderivative) ---
        calculus_bonus = 0.0
        calculus_reason = ""
        if combined < 0.8 and self.symbolic_engine:
            is_antiderivative = self._check_antiderivative(problem_expr["raw"], candidate_expr["raw"])
            if is_antiderivative:
                calculus_bonus = 0.3  # Boost score to likely acceptance
                combined = min(1.0, combined + calculus_bonus)
                calculus_reason = "Correct antiderivative (missing bounds?)"
            else:
                # Check for common mistakes
                power_rule_error = self._check_power_rule_error(problem_expr["raw"], candidate_expr["raw"])
                if power_rule_error:
                    # We don't boost score significantly for mistakes, but we provide a reason.
                    # Or maybe we boost slightly to "Analogous" (Review)?
                    # Let's keep score low but add specific reason.
                    calculus_reason = power_rule_error

        # --- Decision Theory Layer ---
        # We treat 'combined' as P(Match)
        action, utility, debug_utils = self.decision_engine.decide(combined)

        # Map Decision Action to FuzzyLabel
        # This mapping reconciles the decision theory output with the existing FuzzyLabel system.
        if action == DecisionAction.ACCEPT:
            # If accepted, we check how strong the match is for granular labeling
            if combined >= self.thresholds.exact:
                label = FuzzyLabel.EXACT
                label_key = "fuzzy.label.exact"
            elif combined >= self.thresholds.equivalent:
                label = FuzzyLabel.EQUIVALENT
                label_key = "fuzzy.label.equivalent"
            else:
                # Even if score is lower, if Decision Engine says ACCEPT (e.g. encouraging mode),
                # we treat it as Approx Eq.
                label = FuzzyLabel.APPROX_EQ
                label_key = "fuzzy.label.approx_eq"
        
        elif action == DecisionAction.REVIEW:
            # The engine is unsure or suggests review
            label = FuzzyLabel.ANALOGOUS
            label_key = "fuzzy.label.analogous"
            
        else:  # REJECT
            label = FuzzyLabel.CONTRADICT
            label_key = "fuzzy.label.contradict"

        i18n = get_language_pack()
        
        # Main result summary
        label_text = i18n.text(label_key)
        main_msg = i18n.text("fuzzy.result", label=label_text, score=combined)
        
        # Specific reason if applicable
        reason_msg = ""
        if label == FuzzyLabel.APPROX_EQ:
            reason_msg = i18n.text("fuzzy.reason.approx_eq")
        elif label == FuzzyLabel.ANALOGOUS:
            reason_msg = i18n.text("fuzzy.reason.analogous")
            
        # Detail score breakdown
        detail_msg = i18n.text(
            "fuzzy.judge.detail",
            combined=combined,
            expr=expr_sim,
            rule=rule_sim,
            text=text_sim
        )

        full_reason = main_msg
        if reason_msg:
            full_reason += f" | {reason_msg}"
        if calculus_reason:
            full_reason += f" | {calculus_reason}"
        full_reason += f" | {detail_msg}"
        
        # Add decision debug info
        full_reason += f" | Strategy: {self.decision_config.strategy}, Action: {action.value}, Utility: {utility:.2f}"

        return FuzzyResult(
            label=label,
            score=FuzzyScore(
                expr_similarity=expr_sim,
                rule_similarity=rule_sim,
                text_similarity=text_sim,
                combined_score=combined,
            ),
            reason=full_reason,
            debug={
                "problem_raw": problem_expr["raw"],
                "previous_raw": previous_expr["raw"],
                "candidate_raw": candidate_expr["raw"],
                "decision_action": action.value,
                "decision_utility": utility,
                "decision_utils": debug_utils,
                "calculus_bonus": calculus_bonus
            },
        )

    def _check_power_rule_error(self, problem_raw: str, candidate_raw: str) -> str | None:
        """
        Check for common power rule mistakes.
        Returns a hint string if a mistake is detected, else None.
        """
        if not self.symbolic_engine:
            return None
            
        try:
            # Parse problem to find Integral
            problem_sym = self.symbolic_engine.to_internal(problem_raw)
            import sympy
            integrals = problem_sym.atoms(sympy.Integral)
            if not integrals:
                return None
            
            target_integral = list(integrals)[0]
            integrand = target_integral.function
            limits = target_integral.limits
            variable = limits[0][0] if limits else sympy.Symbol('x') # Default fallback
            
            # Check if integrand is polynomial-like (x^n)
            # We can try to match x**n
            # Or just check if candidate is x^(n+1) when it should be x^(n+1)/(n+1)
            
            # Calculate correct antiderivative
            correct_antideriv = sympy.integrate(integrand, variable)
            
            candidate_sym = self.symbolic_engine.to_internal(candidate_raw)
            
            # Case 1: Missing Division
            # candidate == correct * (n+1) ?
            # Or candidate == diff(correct) * x ? No.
            # If correct is x^3/3, candidate x^3 is correct * 3.
            # So check if candidate / correct is a constant (the exponent).
            
            ratio = sympy.simplify(candidate_sym / correct_antideriv)
            if ratio.is_constant() and ratio != 1:
                # It's a constant multiple off.
                # If ratio is > 1, likely missing division.
                return f"Did you forget to divide by the new exponent? (Ratio: {ratio})"
                
            # Case 2: Forgot to integrate (candidate == integrand)
            if sympy.simplify(candidate_sym - integrand) == 0:
                return "You wrote the integrand but didn't integrate it."
                
            return None
            
        except Exception:
            return None

    def _check_antiderivative(self, problem_raw: str, candidate_raw: str) -> bool:
        """
        Check if candidate is the antiderivative of the problem's integrand.
        """
        if not self.symbolic_engine:
            return False
        
        try:
            # 1. Parse problem to find Integral
            # We need to handle "integrate(...)" or "Integral(...)"
            # Use symbolic_engine to parse
            problem_sym = self.symbolic_engine.to_internal(problem_raw)
            
            # Check if it's an Integral (or Mul with Integral)
            # We might need to traverse the expression tree
            import sympy
            
            integrals = problem_sym.atoms(sympy.Integral)
            if not integrals:
                return False
            
            # Assume the main integral is the one we care about (simplification)
            # Or check if the whole expression is equivalent to an Integral
            # For "3 * integrate(...)", it's a Mul.
            
            # Let's try to differentiate the candidate and see if it matches the integrand * constant
            # But we need to know the variable of integration.
            
            target_integral = list(integrals)[0]
            integrand = target_integral.function
            limits = target_integral.limits # ((x, 0, 2),)
            variable = limits[0][0]
            
            # Differentiate candidate
            candidate_sym = self.symbolic_engine.to_internal(candidate_raw)
            derivative = sympy.diff(candidate_sym, variable)
            
            # Now we need to check if derivative is equivalent to the integrand * (problem / integral)
            # This is tricky for "3 * integrate(...)".
            # Easier: Differentiate candidate and see if it equals the derivative of the problem *with respect to the upper bound*?
            # No, that's for FTC.
            
            # If user wrote "x^3" for "integrate(3*x^2, ...)", then diff(x^3) = 3*x^2.
            # And "3 * integrate(x^2)" -> integrand is 3*x^2 (effectively).
            
            # Let's try: diff(candidate) == integrand of (problem expressed as single integral)?
            # Or: diff(candidate) == problem_without_integral_sign?
            
            # If problem is "3 * Integral(x^2)", effectively "Integral(3*x^2)".
            # If we differentiate candidate "x^3", we get "3*x^2".
            
            # So, we want to check if "Integral(diff(candidate))" is equivalent to "problem_sym" (ignoring bounds)?
            # Or "diff(candidate)" equivalent to "integrand of problem"?
            
            # Let's try to convert problem to indefinite integral and compare with candidate?
            # No, candidate is "x^3", indefinite integral of "3*x^2".
            
            # Approach:
            # 1. Differentiate candidate: D
            # 2. Integrate D (indefinite): I_D
            # 3. Compare I_D with problem's indefinite version?
            
            # Better:
            # 1. Differentiate candidate: D
            # 2. Compare D with the integrand of the problem.
            # How to get "integrand of the problem" if it's "3 * Integral(x^2)"?
            # We can try to put everything inside the integral?
            
            # Alternative:
            # Check if diff(candidate, variable) == problem_sym.replace(Integral, lambda f, *args: f) ?
            # No, "3 * Integral(x^2)" -> "3 * x^2".
            # diff("x^3") -> "3 * x^2". Match!
            
            # So:
            # 1. Identify integration variable from the Integral atom.
            # 2. Differentiate candidate w.r.t that variable.
            # 3. Strip the Integral wrapper from problem_sym (replace Integral(f, ...) with f).
            # 4. Compare derivative with stripped problem.
            
            # Handle multiple integrals? Just take the first one found.
            
            # Replace all Integrals in problem_sym with their integrands
            problem_stripped = problem_sym.replace(
                sympy.Integral,
                lambda f, *args: f
            )
            
            return self.symbolic_engine.is_equiv(derivative, problem_stripped)
            
        except Exception:
            return False
