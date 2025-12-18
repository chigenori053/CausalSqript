"""Validation engine for MathLang Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .computation_engine import ComputationEngine
from .exercise_spec import ExerciseSpec
from .errors import InvalidExprError, EvaluationError
from .fuzzy.judge import FuzzyJudge
from coherent.logic.causal_engine import CausalEngine
from .decision_theory import DecisionEngine, DecisionAction
from .fuzzy.judge import FuzzyJudge
from coherent.logic.causal_engine import CausalEngine
from .decision_theory import DecisionEngine, DecisionAction
from .fuzzy.types import FuzzyLabel
from .knowledge_registry import KnowledgeRegistry
from coherent.optical.vectorizer import FeatureExtractor
from coherent.optical.layer import OpticalInterferenceEngine
import numpy as np


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
    
    
    def __init__(self, computation_engine: ComputationEngine, fuzzy_judge: FuzzyJudge | None = None, causal_engine: CausalEngine | None = None, decision_engine: DecisionEngine | None = None, knowledge_registry: KnowledgeRegistry | None = None):
        """
        Initialize the validation engine.
        
        Args:
            computation_engine: ComputationEngine instance for symbolic operations
            fuzzy_judge: Optional FuzzyJudge instance for fuzzy validation
            causal_engine: Optional CausalEngine for context awareness
            decision_engine: Optional DecisionEngine for strategic judgment
            knowledge_registry: Optional KnowledgeRegistry for rule-based prediction
        """
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
        self.fuzzy_judge = fuzzy_judge
        self.causal_engine = causal_engine
        self.decision_engine = decision_engine
        self.knowledge_registry = knowledge_registry
        
        # Initialize Optical Components (Lazy or direct)
        self.vectorizer = FeatureExtractor()
        # Mock weights path for now or let it be random/default
        self.optical_layer = OpticalInterferenceEngine(input_dim=64, memory_capacity=100)

    def validate_step(self, before: str, after: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Validate a transition step using the Integrated Evaluation Pipeline.
        
        Pipeline:
        1. Context & Optimization (Causal) [Check if we can skip symbolic]
        2. Symbolic Check (Strict correctness)
        3. Fuzzy Perception (Approximation/Typo check if symbolic fails)
        4. Strategic Decision (Final verdict based on utility)
        
        Args:
            before: The previous expression (or problem specification)
            after: The user's new expression
            context: Variable context if any
            
        Returns:
            Dictionary containing validation results and metadata.
        """
        # --- 1. Preparation ---
        # Normalize expressions (handle equations)
        # Note: Normalization logic currently resides in CoreRuntime. 
        # Ideally, we passed normalized strings here, but let's assume 'before' and 'after' are passed as raw strings
        # and we delegate normalization to ComputationEngine or handle it if passed already normalized.
        # Implementation assumes 'before' and 'after' are the expressions to compare directly.
        
        # Apply context if exists
        before_eval = before
        after_eval = after
        if context:
            try:
                before_eval = self.computation_engine.substitute(before, context)
                after_eval = self.computation_engine.substitute(after, context)
            except Exception:
                pass # Fallback to raw
        
            except Exception:
                pass # Fallback to raw

        # --- 1. Predictive Skip (Optimization) ---
        # If the user's step matches a known rule application outcome (predicted by Causal/Knowledge),
        # we can skip the expensive symbolic check.
        if self.knowledge_registry:
            try:
                # Use raw 'before' as knowledge rules are often defined on raw patterns (with wildcards).
                # But 'before_eval' might be better if variables are involved?
                # KnowledgeRegistry uses 'match_structure' which handles structure.
                # Let's use 'before' (user input form) to match rules written for user input form.
                candidates = self.knowledge_registry.suggest_outcomes(before)
                norm_after = "".join(after.split())
                
                for cand in candidates:
                    if "".join(cand.split()) == norm_after:
                         return {
                            "valid": True,
                            "status": "correct",
                            "details": {"method": "predictive_skip", "candidate": cand}
                         }
            except Exception:
                pass
        
        # --- 2. Symbolic Check ---
        is_valid = False
        is_partial = False
        # We can reuse the extensive logic from CoreRuntime here or simplify.
        # For now, let's use the symbolic engine's is_equiv.
        try:
             is_valid = self.symbolic_engine.is_equiv(before_eval, after_eval, context=context)
        except Exception:
             is_valid = False
             
        # TODO: Add partial calculation logic if needed (moved from CoreRuntime)
        
        # If valid, we are potentially done, unless we want to "Review" valid steps (e.g. redundancy)
        if is_valid:
            return {
                "valid": True,
                "status": "correct",
                "details": {}
            }

        # --- 3. Fuzzy Perception ---
        fuzzy_score = 0.0
        fuzzy_label = None
        fuzzy_debug = {}
        
        if self.fuzzy_judge:
            try:
                # We use 'before' as both problem and previous for single-step check
                # In a full flow, CausalEngine would provide the true 'problem_expr'.
                # Here we treat the previous step as the reference.
                encoder = self.fuzzy_judge.encoder
                norm_before = encoder.normalize(before)
                norm_after = encoder.normalize(after)
                
                fuzzy_result = self.fuzzy_judge.judge_step(
                     problem_expr=norm_before,
                     previous_expr=norm_before,
                     candidate_expr=norm_after
                )
                fuzzy_score = fuzzy_result['score']['combined_score']
                fuzzy_label = fuzzy_result['label']
                fuzzy_debug = fuzzy_result.get('debug', {})
            except Exception:
                pass
        
        # --- 4. Strategic Decision ---
        action = DecisionAction.REJECT
        decision_meta = {}
        
        if self.decision_engine and self.fuzzy_judge:
             # Calculate Ambiguity using Optical Layer
             try:
                 # Vectorize 'before' (Mock/Simplify as parser access is limited here)
                 # Ideally: vector = self.vectorizer.vectorize(parsed_ast)
                 # For MVP: Zero vector or simplistic feature from string string?
                 # self.vectorizer expects AST.
                 # We will pass a zero vector just to exercise the pipeline (Phase 1)
                 # In Phase 2/3, we connect real AST.
                 # For MVP: Zero vector or simplistic feature from string string?
                 # self.vectorizer expects AST.
                 # We will pass a zero vector just to exercise the pipeline (Phase 1)
                 # In Phase 2/3, we connect real AST.
                 import torch
                 vec = torch.zeros(64) 
                 # Use PyTorch forward pass (no predict method)
                 # Returns (intensity, ambiguity)
                 with torch.no_grad():
                     ambiguity = self.optical_layer.get_ambiguity(res)
             except Exception:
                 ambiguity = 0.0

             # Use the decision engine to decide based on fuzzy score
             # We assume DecisionEngine uses 'probability of match' which maps to fuzzy_score
             action, utility, debug = self.decision_engine.decide(fuzzy_score, ambiguity=ambiguity)
             decision_meta = {
                 "utility": utility,
                 "debug": debug,
                 "action": action.value
             }
        else:
             # Fallback if no decision engine: Use simple fuzzy threshold or strict
             # If fuzzy_label is "high" enough, ACCEPT
             if fuzzy_label in [FuzzyLabel.EXACT, FuzzyLabel.EQUIVALENT, FuzzyLabel.APPROX_EQ, FuzzyLabel.ANALOGOUS]:
                 action = DecisionAction.ACCEPT
             else:
                 action = DecisionAction.REJECT

        # --- 5. Result Construction ---
        result_valid = (action == DecisionAction.ACCEPT)
        
        # Handling "Review" state
        # For now, Review -> Valid but with "review" status, or Invalid?
        # Usually Review means "Correct but needs improvement" or "Incorrect but close".
        # The prompt says: REVIEW: "惜しい判定。ヒントを出す。" -> Implies might be treated as invalid for progression?
        # Or Valid but generates a hint.
        # Let's say: ACCEPT = Valid, REJECT = Invalid.
        # REVIEW = Invalid (so user stays on step) but with encouraging hint?
        # OR REVIEW = Valid (proceed) but with warning?
        # "Mistake" usually blocks progress.
        # "Review" sounds like "Partial".
        # Let's map REVIEW to False (Invalid) for now to force user correction, OR True with warning.
        # "Review" in DecisionTheory often implies "Gather more info" or "Ask user". 
        # In this context (Education), "Review" likely means "Close, try again (don't fail hard)".
        # So Valid=False, but Status="review".
        
        if action == DecisionAction.ACCEPT:
            result_valid = True
            status = "correct"
        elif action == DecisionAction.REVIEW:
            result_valid = False # User must fix it? Or is it a 'pass with warning'?
            # If we mark it valid=False, the system won't advance.
            # If we mark it valid=True, the system advances.
            # "惜しい" means "Almost". Usually you have to fix "Almost".
            result_valid = False
            status = "review"
        else:
            result_valid = False
            status = "mistake"
            
        return {
            "valid": result_valid,
            "status": status,
            "details": {
                "fuzzy_score": fuzzy_score,
                "fuzzy_label": fuzzy_label.value if fuzzy_label else None,
                "decision": decision_meta
            }
        }
    
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
