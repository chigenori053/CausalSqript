"""Hint engine for MathLang Core (Agentic Edu Update)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum
import datetime
import uuid

from .computation_engine import ComputationEngine
from .exercise_spec import ExerciseSpec
from .errors import InvalidExprError, EvaluationError


class DriftType(str, Enum):
    """Types of structural drift between user input and target."""
    NO_DRIFT = "NO_DRIFT" # Exact match structure-wise (or close enough)
    REPRESENTATION_EQUIV = "REPRESENTATION_EQUIV" # Mathematically equal, form differs
    LIKE_TERMS_UNMERGED = "LIKE_TERMS_UNMERGED" # e.g. x + x instead of 2x
    SIGN_ERROR = "SIGN_ERROR" # e.g. -(a+b) vs -a+b errors
    CONSTANT_OFFSET = "CONSTANT_OFFSET" # Differs by a constant
    TERM_MISSING_OR_EXTRA = "TERM_MISSING_OR_EXTRA" # Structural mismatch
    UNRELATED = "UNRELATED_TRANSFORM" # Completely off


@dataclass
class ThoughtEvent:
    """Telemetry for agent decision process."""
    event_id: str
    timestamp: str
    user_expr: str
    target_expr: str
    detected_drift: DriftType
    resonance_score: float
    action_selected: str
    rationale: str


@dataclass
class HintResult:
    """Result of validation/hint generation."""
    message: str
    hint_type: str
    details: Dict[str, Any] = field(default_factory=dict)


class GapAnalyzer:
    """Analyzes the structural gap between User and Target Expressions."""
    
    def __init__(self, sym_engine):
        self.sym_engine = sym_engine

    def analyze(self, user_expr: str, target_expr: str) -> DriftType:
        """Categorize the drift."""
        try:
            # 1. Check Equivalence first
            is_equiv = self.sym_engine.is_equiv(user_expr, target_expr)
            
            if is_equiv:
                # If equivalent, check for unmerged terms or just representation diff
                # Simple heuristic: String length or complexity.
                # If user expr is significantly longer than target, assume unsimplification
                # stored as LIKE_TERMS_UNMERGED or REPRESENTATION_EQUIV
                # For now, let's use a naive length heuristic or assume if text differs it's Rep Equiv
                if user_expr.strip() == target_expr.strip():
                    return DriftType.NO_DRIFT
                
                # Check for "x+x" vs "2x" type things?
                # Without AST depth inspection, we guess REPRESENTATION_EQUIV
                return DriftType.REPRESENTATION_EQUIV

            # 2. Check Sign Error
            try:
                if self.sym_engine.is_equiv(user_expr, f"-({target_expr})"):
                    return DriftType.SIGN_ERROR
            except: pass

            # 3. Check Constant Offset
            try:
                diff = f"({user_expr}) - ({target_expr})"
                # Simplify diff
                val = self.sym_engine.simplify(diff)
                # Check if result is a number != 0
                # Assuming simplify returns a string of a number if it reduces
                try:
                    float(val)
                    return DriftType.CONSTANT_OFFSET
                except ValueError:
                    pass
            except: pass
            
            # 4. Fallback: Term missing/extra or Unrelated
            # This is hard to distinguish without tree edit distance.
            # Default to Unrelated for now, or Term Missing if we had finer grain.
            return DriftType.UNRELATED

        except (InvalidExprError, EvaluationError):
            return DriftType.UNRELATED


class HintEngine:
    """
    Agentic Hint Engine.
    Uses Gap Analysis and Decision Theory to select actions.
    """
    
    def __init__(self, computation_engine: ComputationEngine):
        self.computation_engine = computation_engine
        self.symbolic_engine = computation_engine.symbolic_engine
        self.gap_analyzer = GapAnalyzer(self.symbolic_engine)
        self.history: List[ThoughtEvent] = []
    
    def generate_hint(
        self, 
        user_expr: str, 
        target_expr: str, 
        hint_rules: Optional[Dict[str, str]] = None,
        persona: str = "balanced", # Kept for API compatibility, influences utility
        validation_details: Optional[Dict[str, Any]] = None
    ) -> HintResult:
        """
        Generate hint based on drift analysis.
        """
        # 1. Analyze Gap
        drift = self.gap_analyzer.analyze(user_expr, target_expr)
        
        # 2. Observe Context (Resonance)
        # In a real agent, we'd get this from the Memory System / State
        # Here we simulate or extract from validation_details if available
        resonance = 0.5 # Default middle ground
        if validation_details and "resonance" in validation_details:
             resonance = validation_details["resonance"]
             
        # 3. Decision Logic (Policy)
        action, message = self._decide_action(drift, resonance, persona, hint_rules)
        
        # 4. Log Thought
        event = ThoughtEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            user_expr=user_expr,
            target_expr=target_expr,
            detected_drift=drift,
            resonance_score=resonance,
            action_selected=action,
            rationale=f"Drift: {drift.value}, Resonance: {resonance}"
        )
        self.history.append(event)
        
        return HintResult(
            message=message,
            hint_type=action, # Map action to hint_type
            details={
                "drift": drift.value,
                "event_id": event.event_id
            }
        )

    def _decide_action(self, drift: DriftType, resonance: float, persona: str, hint_rules: Optional[Dict[str, str]]) -> tuple[str, str]:
        """
        Returns (action_type, content).
        Action Types: 'suppress', 'hint_generic', 'hint_specific', 'answer'
        """
        # High resonance (Understanding) -> Don't give answers, give nudges
        # Low resonance (Confusion) -> Give more specific help
        
        if drift == DriftType.NO_DRIFT:
            return "success", "Correct!"
            
        if drift == DriftType.REPRESENTATION_EQUIV:
            return "hint_structure", "Your answer is mathematically correct, but try to simplify it or match the requested format."
            
        if drift == DriftType.SIGN_ERROR:
             return "hint_specific", "Check your signs. It looks like a positive/negative inversion."
             
        if drift == DriftType.CONSTANT_OFFSET:
             return "hint_specific", "You are off by a constant value. Did you miss a term?"
        
        # Unrelated or Complex Error
        if resonance > 0.8:
            # User "should" know this (High Resonance), so maybe they made a typo or need a small nudge
            return "hint_generic", "Double check your steps. You're on the right track concept-wise."
        
        elif resonance < 0.3:
            # User is likely confused.
            # If we have specific hint rules (manual overrides), use them
            if hint_rules:
                # Naive check: return first rule? Or just generic fallback
                # Ideally GapAnalyzer matches rules.
                pass
                
            return "hint_scaffold", "Let's break this down. Review the previous step."
            
        else:
            # Middle ground
            return "hint_generic", "Something looks off. Try verifying your result."
    
    def generate_hint_for_spec(self, user_expr: str, spec: ExerciseSpec, persona: str = "balanced") -> HintResult:
        """Facade for ExerciseSpec."""
        return self.generate_hint(
            user_expr, 
            spec.target_expression, 
            spec.hint_rules,
            persona
        )
