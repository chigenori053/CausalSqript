"""Hint engine for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum

from .computation_engine import ComputationEngine
from .exercise_spec import ExerciseSpec
from .errors import InvalidExprError, EvaluationError



@dataclass
class HintCandidate:
    """
    A candidate hint that could be shown to the user.
    """
    content: str
    type: str
    probability: float  # Confidence that this hint addresses the actual error
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HintPersona(str, Enum):
    """Educational persona for hint selection."""
    SPARTA = "sparta"    # Minimal hints, encourages self-correction
    SUPPORT = "support"  # Maximum support, prevents frustration
    BALANCED = "balanced"


@dataclass
class HintResult:
    """
    Result of generating a hint for a user's answer.
    """
    message: str
    hint_type: str
    details: Dict[str, Any] = field(default_factory=dict)


class HintEngine:
    """
    Generates hints based on incorrect answer patterns or symbolic differences,
    using a utility-based selection mechanism.
    """
    
    def __init__(self, computation_engine: ComputationEngine):
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
    
    def generate_candidates(
        self, 
        user_expr: str, 
        target_expr: str, 
        hint_rules: Optional[Dict[str, str]] = None,
        validation_details: Optional[Dict[str, Any]] = None
    ) -> List[HintCandidate]:
        """
        Generate a list of possible hint candidates with confidence scores.
        """
        candidates = []
        
        # 0. Review Encouragement (Highest Priority)
        # If validation marked this as "Review", it means it's mathematically sound but imperfect.
        if validation_details and (validation_details.get("review_needed") or validation_details.get("status") == "review"):
             # Extract specific advice from decision engine debug if available
             advice = "You are extremely close! The logic seems correct, but the form might be slightly off."
             decision_data = validation_details.get("decision", {})
             if "debug" in decision_data and "advice" in decision_data["debug"]:
                 advice = decision_data["debug"]["advice"]
                 
             candidates.append(HintCandidate(
                 content=advice,
                 type="review_encouragement",
                 probability=0.99, # Extremely high confidence
                 source="decision_review"
             ))

        # 1. Validate expression
        try:
            self.symbolic_engine.to_internal(user_expr)
        except InvalidExprError:
            return [HintCandidate(
                content="Your expression contains syntax errors.",
                type="syntax_error",
                probability=1.0,
                source="parser"
            )]

        # 2. Pattern Matching (High confidence)
        if hint_rules:
            for pattern, hint_msg in hint_rules.items():
                try:
                    if self.symbolic_engine.is_equiv(user_expr, pattern):
                        candidates.append(HintCandidate(
                            content=hint_msg,
                            type="pattern_match",
                            probability=0.95,
                            source="rule",
                            metadata={"pattern": pattern}
                        ))
                except (InvalidExprError, EvaluationError):
                    continue

        # 3. Heuristics
        
        # 3a. Sign Error
        try:
            neg_target = f"-({target_expr})"
            if self.symbolic_engine.is_equiv(user_expr, neg_target):
                candidates.append(HintCandidate(
                    content="It looks like you might have a sign error. Check your positives and negatives.",
                    type="heuristic_sign_error",
                    probability=0.8,
                    source="heuristic"
                ))
        except (InvalidExprError, EvaluationError):
            pass
            
        # 3b. Constant Offset
        try:
            diff_expr = f"({user_expr}) - ({target_expr})"
            simplified_diff = self.symbolic_engine.simplify(diff_expr)
            
            try:
                val = float(simplified_diff)
                if val != 0:
                    candidates.append(HintCandidate(
                        content="You're close, but your answer differs by a constant amount.",
                        type="heuristic_constant_offset",
                        probability=0.7,
                        source="heuristic",
                        metadata={"offset": simplified_diff}
                    ))
            except ValueError:
                diff_text = str(simplified_diff)
                has_add_or_sub = ("+" in diff_text) or ("-" in diff_text[1:])
                if not has_add_or_sub:
                     candidates.append(HintCandidate(
                        content=f"You might be missing or have an extra term: {simplified_diff}",
                        type="heuristic_term_difference",
                        probability=0.6,
                        source="heuristic",
                        metadata={"diff": diff_text}
                    ))
        except (InvalidExprError, EvaluationError):
            pass

        # 3c. Missing Bounds (Definite Integral)
        try:
            if self.symbolic_engine.is_numeric(target_expr) and not self.symbolic_engine.is_numeric(user_expr):
                candidates.append(HintCandidate(
                    content="It looks like you have a variable in your answer, but the result should be a number. Did you forget to apply the bounds?",
                    type="heuristic_missing_bounds",
                    probability=0.85,
                    source="heuristic"
                ))
        except (InvalidExprError, EvaluationError):
            pass

        # 4. Fallback / Generic Hints
        candidates.append(HintCandidate(
            content="Try checking your steps carefully.",
            type="none",
            probability=0.1,  # Low probability of being "the right hint" but always applicable
            source="fallback"
        ))

        return candidates

    def select_best_hint(self, candidates: List[HintCandidate], persona: str = "balanced") -> HintResult:
        """
        Select the best hint based on the persona's utility function.
        """
        if not candidates:
            return HintResult(message="No hint available.", hint_type="none")

        # Utility Parameters based on Persona
        # U_solve: Utility of solving (fixed)
        # U_self: Utility of solving by oneself (higher for abstract hints)
        # C_confusion: Cost of confusing hint
        # C_giveup: Cost of user giving up (higher for Support persona)

        if persona == HintPersona.SPARTA:
            u_self_bonus = 50.0
            c_giveup = -20.0
        elif persona == HintPersona.SUPPORT:
            u_self_bonus = 10.0
            c_giveup = -100.0
        else: # Balanced
            u_self_bonus = 30.0
            c_giveup = -50.0

        best_candidate = None
        max_utility = -float('inf')

        for cand in candidates:
            # Determine hint abstractness (simplified)
            is_specific = cand.type in ["pattern_match", "heuristic_constant_offset", "syntax_error"]
            
            # U_self depends on abstractness
            u_self = 0.0 if is_specific else u_self_bonus
            
            # Probability that this hint leads to solution
            p_solve = cand.probability * (0.9 if is_specific else 0.5)
            
            # Expected Utility
            # EU = P(Solve) * (U_solve + U_self) + (1 - P(Solve)) * C_giveup
            # We assume U_solve = 100
            u_solve = 100.0
            
            eu = p_solve * (u_solve + u_self) + (1 - p_solve) * c_giveup
            
            if eu > max_utility:
                max_utility = eu
                best_candidate = cand

        if best_candidate:
            return HintResult(
                message=best_candidate.content,
                hint_type=best_candidate.type,
                details={
                    **best_candidate.metadata,
                    "selection_utility": max_utility,
                    "persona": persona
                }
            )
        
        # Should not happen if fallback is present
        return HintResult(message="Try again.", hint_type="none")

    def generate_hint(
        self, 
        user_expr: str, 
        target_expr: str, 
        hint_rules: Optional[Dict[str, str]] = None,
        persona: str = "balanced",
        validation_details: Optional[Dict[str, Any]] = None
    ) -> HintResult:
        """
        Generate and select a hint.
        """
        candidates = self.generate_candidates(user_expr, target_expr, hint_rules, validation_details)
        return self.select_best_hint(candidates, persona)

    def generate_hint_for_spec(self, user_expr: str, spec: ExerciseSpec, persona: str = "balanced") -> HintResult:
        """
        Generate a hint based on an ExerciseSpec.
        """
        # Check if spec has a preferred persona, otherwise use default
        # (Assuming ExerciseSpec might be extended later, for now use argument)
        return self.generate_hint(
            user_expr, 
            spec.target_expression, 
            spec.hint_rules,
            persona
        )
