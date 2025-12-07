"""Validation engine for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .computation_engine import ComputationEngine
from .exercise_spec import ExerciseSpec
from .errors import InvalidExprError, EvaluationError
from .fuzzy.judge import FuzzyJudge


@dataclass
class ValidationResult:
    """
    Result of validating a user's answer against an exercise specification.
    
    Attributes:
        is_correct: Whether the answer is correct
        validation_mode: The validation mode used
        user_expr: The user's expression
        target_expr: The target expression
        message: Human-readable feedback message
        details: Additional validation details (e.g., symbolic difference, format issues)
    """
    
    is_correct: bool
    validation_mode: str
    user_expr: str
    target_expr: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class ValidationEngine:
    """
    Performs mathematical equivalence checking and format-based validation.
    
    The ValidationEngine validates user answers against exercise specifications,
    supporting multiple validation modes:
    - symbolic_equiv: Check if expressions are symbolically equivalent
    - exact_form: Check if expressions match exactly (after normalization)
    - canonical_form: Check if expression matches a specific canonical form
    """
    
    def __init__(self, computation_engine: ComputationEngine, fuzzy_judge: FuzzyJudge | None = None):
        """
        Initialize the validation engine.
        
        Args:
            computation_engine: ComputationEngine instance for symbolic operations
            fuzzy_judge: Optional FuzzyJudge instance for fuzzy validation
        """
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
        self.fuzzy_judge = fuzzy_judge
    
    def check_answer(self, user_expr: str, spec: ExerciseSpec) -> ValidationResult:
        """
        Check a user's answer against an exercise specification.
        
        Args:
            user_expr: The user's answer expression
            spec: Exercise specification defining the validation criteria
            
        Returns:
            ValidationResult with correctness status and feedback
            
        Raises:
            InvalidExprError: If the user expression is invalid
        """
        # Validate that the user expression can be parsed
        try:
            self.symbolic_engine.to_internal(user_expr)
        except InvalidExprError as exc:
            return ValidationResult(
                is_correct=False,
                validation_mode=spec.validation_mode,
                user_expr=user_expr,
                target_expr=spec.target_expression,
                message=f"Invalid expression: {exc}",
                details={"error": str(exc)},
            )
        
        # Delegate to appropriate validation method
        if spec.validation_mode == "symbolic_equiv":
            return self._check_symbolic_equiv(user_expr, spec.target_expression)
        elif spec.validation_mode == "exact_form":
            return self._check_exact_form(user_expr, spec.target_expression)
        elif spec.validation_mode == "canonical_form":
            return self._check_canonical_form(user_expr, spec)
        else:
            # This should not happen due to ExerciseSpec validation
            raise ValueError(f"Unknown validation mode: {spec.validation_mode}")
    
    def _check_symbolic_equiv(
        self, user_expr: str, target_expr: str
    ) -> ValidationResult:
        """
        Check if two expressions are symbolically equivalent.
        
        Args:
            user_expr: The user's expression
            target_expr: The target expression
            
        Returns:
            ValidationResult indicating equivalence
        """
        try:
            is_equiv = self.symbolic_engine.is_equiv(user_expr, target_expr)
        except (InvalidExprError, EvaluationError) as exc:
            return ValidationResult(
                is_correct=False,
                validation_mode="symbolic_equiv",
                user_expr=user_expr,
                target_expr=target_expr,
                message=f"Error checking equivalence: {exc}",
                details={"error": str(exc)},
            )
        
        if is_equiv:
            message = "Correct! Your answer is symbolically equivalent to the target."
        else:
            message = "Incorrect. Your answer is not equivalent to the target expression."
            # Provide additional details about the difference
            try:
                simplified_user = self.symbolic_engine.simplify(user_expr)
                simplified_target = self.symbolic_engine.simplify(target_expr)
                details = {
                    "simplified_user": simplified_user,
                    "simplified_target": simplified_target,
                }
            except Exception:
                details = {}
        
        return ValidationResult(
            is_correct=is_equiv,
            validation_mode="symbolic_equiv",
            user_expr=user_expr,
            target_expr=target_expr,
            message=message,
            details=details if not is_equiv else {},
        )
    
    def _check_exact_form(
        self, user_expr: str, target_expr: str
    ) -> ValidationResult:
        """
        Check if two expressions match exactly after normalization.
        
        Args:
            user_expr: The user's expression
            target_expr: The target expression
            
        Returns:
            ValidationResult indicating exact match
        """
        # Normalize both expressions by simplifying
        try:
            normalized_user = self.symbolic_engine.simplify(user_expr)
            normalized_target = self.symbolic_engine.simplify(target_expr)
        except (InvalidExprError, EvaluationError) as exc:
            return ValidationResult(
                is_correct=False,
                validation_mode="exact_form",
                user_expr=user_expr,
                target_expr=target_expr,
                message=f"Error normalizing expressions: {exc}",
                details={"error": str(exc)},
            )
        
        is_exact_match = normalized_user == normalized_target
        
        if is_exact_match:
            message = "Correct! Your answer matches the expected form exactly."
        else:
            message = (
                "Incorrect. Your answer must match the exact form. "
                f"Expected: {normalized_target}, Got: {normalized_user}"
            )
        
        return ValidationResult(
            is_correct=is_exact_match,
            validation_mode="exact_form",
            user_expr=user_expr,
            target_expr=target_expr,
            message=message,
            details={
                "normalized_user": normalized_user,
                "normalized_target": normalized_target,
            },
        )
    
    def _check_canonical_form(
        self, user_expr: str, spec: ExerciseSpec
    ) -> ValidationResult:
        """
        Check if expression matches the canonical form specified in the spec.
        
        Args:
            user_expr: The user's expression
            spec: Exercise specification with canonical_form
            
        Returns:
            ValidationResult indicating canonical form match
        """
        if spec.canonical_form is None:
            raise ValueError("canonical_form must be provided in ExerciseSpec")
        
        # First check symbolic equivalence with target
        try:
            is_equiv = self.symbolic_engine.is_equiv(user_expr, spec.target_expression)
        except (InvalidExprError, EvaluationError) as exc:
            return ValidationResult(
                is_correct=False,
                validation_mode="canonical_form",
                user_expr=user_expr,
                target_expr=spec.target_expression,
                message=f"Error checking equivalence: {exc}",
                details={"error": str(exc)},
            )
        
        if not is_equiv:
            return ValidationResult(
                is_correct=False,
                validation_mode="canonical_form",
                user_expr=user_expr,
                target_expr=spec.target_expression,
                message="Incorrect. Your answer is not equivalent to the target expression.",
                details={},
            )
        
        # Check if it matches the canonical form
        try:
            normalized_user = self.symbolic_engine.simplify(user_expr)
            normalized_canonical = self.symbolic_engine.simplify(spec.canonical_form)
        except (InvalidExprError, EvaluationError) as exc:
            return ValidationResult(
                is_correct=False,
                validation_mode="canonical_form",
                user_expr=user_expr,
                target_expr=spec.target_expression,
                message=f"Error normalizing expressions: {exc}",
                details={"error": str(exc)},
            )
        
        is_canonical_match = normalized_user == normalized_canonical
        
        if is_canonical_match:
            message = "Correct! Your answer is equivalent and in the correct canonical form."
        else:
            message = (
                "Your answer is mathematically correct, but not in the required form. "
                f"Please express it as: {spec.canonical_form}"
            )
        
        return ValidationResult(
            is_correct=is_canonical_match,
            validation_mode="canonical_form",
            user_expr=user_expr,
            target_expr=spec.target_expression,
            message=message,
            details={
                "normalized_user": normalized_user,
                "canonical_form": normalized_canonical,
                "is_mathematically_correct": True,
            },
        )
