"""Fuzzy judge implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FuzzyThresholdConfig
from .encoder import ExpressionEncoder
from .metric import SimilarityMetric
from .types import FuzzyLabel, FuzzyResult, FuzzyScore, NormalizedExpr
from ..i18n import get_language_pack
from core.decision_theory import DecisionConfig, DecisionEngine, DecisionAction



def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class FuzzyJudge:
    encoder: ExpressionEncoder
    metric: SimilarityMetric
    thresholds: FuzzyThresholdConfig | None = None
    decision_config: DecisionConfig | None = None

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
                "decision_utils": debug_utils
            },
        )
