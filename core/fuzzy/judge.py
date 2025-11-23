"""Fuzzy judge implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FuzzyThresholdConfig
from .encoder import ExpressionEncoder
from .metric import SimilarityMetric
from .types import FuzzyLabel, FuzzyResult, FuzzyScore, NormalizedExpr


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class FuzzyJudge:
    encoder: ExpressionEncoder
    metric: SimilarityMetric
    thresholds: FuzzyThresholdConfig | None = None

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = FuzzyThresholdConfig()

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

        t = self.thresholds
        if combined >= t.exact:
            label = FuzzyLabel.EXACT
        elif combined >= t.equivalent:
            label = FuzzyLabel.EQUIVALENT
        elif combined >= t.approx_eq:
            label = FuzzyLabel.APPROX_EQ
        elif combined >= t.analogous:
            label = FuzzyLabel.ANALOGOUS
        elif combined <= t.contradict:
            label = FuzzyLabel.CONTRADICT
        else:
            label = FuzzyLabel.UNKNOWN

        return FuzzyResult(
            label=label,
            score=FuzzyScore(
                expr_similarity=expr_sim,
                rule_similarity=rule_sim,
                text_similarity=text_sim,
                combined_score=combined,
            ),
            reason=f"Label={label.value}, score={combined:.3f}",
            debug={
                "problem_raw": problem_expr["raw"],
                "previous_raw": previous_expr["raw"],
                "candidate_raw": candidate_expr["raw"],
            },
        )
