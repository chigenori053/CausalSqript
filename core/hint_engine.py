"""Hint engine for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .computation_engine import ComputationEngine
from .exercise_spec import ExerciseSpec
from .errors import InvalidExprError, EvaluationError


@dataclass
class HintResult:
    """
    Result of generating a hint for a user's answer.
    
    Attributes:
        message: The hint message to display to the user
        hint_type: Type of hint generated (e.g., "pattern_match", "heuristic", "none")
        details: Additional details about the hint generation
    """
    
    message: str
    hint_type: str
    details: Dict[str, Any] = field(default_factory=dict)


class HintEngine:
    """
    Generates hints based on incorrect answer patterns or symbolic differences.
    
    The HintEngine analyzes user answers against the target expression and
    defined hint rules to provide helpful feedback.
    """
    
    def __init__(self, computation_engine: ComputationEngine):
        """
        Initialize the hint engine.
        
        Args:
            computation_engine: ComputationEngine instance for symbolic operations
        """
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
    
    def generate_hint(
        self, 
        user_expr: str, 
        target_expr: str, 
        hint_rules: Optional[Dict[str, str]] = None
    ) -> HintResult:
        """
        Generate a hint for a user's answer against a specific target.
        
        Args:
            user_expr: The user's answer expression
            target_expr: The target expression (correct answer or previous step)
            hint_rules: Optional dictionary mapping error patterns to hints
            
        Returns:
            HintResult containing the hint message and metadata
        """
        # 1. Validate expression first
        try:
            self.symbolic_engine.to_internal(user_expr)
        except InvalidExprError:
            return HintResult(
                message="Your expression contains syntax errors.",
                hint_type="syntax_error"
            )
            
        # 2. Check for Pattern Matching (hint_rules)
        if hint_rules:
            for pattern, hint_msg in hint_rules.items():
                try:
                    if self.symbolic_engine.is_equiv(user_expr, pattern):
                        return HintResult(
                            message=hint_msg,
                            hint_type="pattern_match",
                            details={"pattern": pattern}
                        )
                except (InvalidExprError, EvaluationError):
                    continue

        # 3. Heuristics
        
        # 3a. Sign Error (user == -target)
        try:
            neg_target = f"-({target_expr})"
            if self.symbolic_engine.is_equiv(user_expr, neg_target):
                return HintResult(
                    message="It looks like you might have a sign error. Check your positives and negatives.",
                    hint_type="heuristic_sign_error"
                )
        except (InvalidExprError, EvaluationError):
            pass
            
        # 3b. Constant Offset (user - target == constant)
        try:
            diff_expr = f"({user_expr}) - ({target_expr})"
            simplified_diff = self.symbolic_engine.simplify(diff_expr)
            
            try:
                # A robust way: try to convert simplified_diff to float
                val = float(simplified_diff)
                # If successful and not 0 (which would mean correct answer)
                if val != 0:
                     return HintResult(
                        message="You're close, but your answer differs by a constant amount.",
                        hint_type="heuristic_constant_offset",
                        details={"offset": simplified_diff}
                    )
            except ValueError:
                pass # Not a number
                
        except (InvalidExprError, EvaluationError):
            pass

        # 4. Fallback
        return HintResult(
            message="Try checking your steps carefully.",
            hint_type="none"
        )

    def generate_hint_for_spec(self, user_expr: str, spec: ExerciseSpec) -> HintResult:
        """
        Generate a hint based on an ExerciseSpec (backward compatibility).
        
        Args:
            user_expr: The user's answer expression
            spec: Exercise specification
            
        Returns:
            HintResult
        """
        return self.generate_hint(
            user_expr, 
            spec.target_expression, 
            spec.hint_rules
        )
