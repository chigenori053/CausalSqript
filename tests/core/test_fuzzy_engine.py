import math
from typing import List

import pytest

from coherent.engine.fuzzy.config import FuzzyThresholdConfig
from coherent.engine.fuzzy.encoder import ExpressionEncoder, MLVector
from coherent.engine.fuzzy.judge import FuzzyJudge
from coherent.engine.fuzzy.metric import SimilarityMetric
from coherent.engine.fuzzy.types import FuzzyLabel, NormalizedExpr


def _expr(raw: str) -> NormalizedExpr:
    return {"raw": raw, "sympy": raw, "tokens": raw.split()}


def test_expression_encoder_is_deterministic():
    encoder = ExpressionEncoder(dimension=8)
    expr = _expr("a + b")
    vector1 = encoder.encode_expr(expr)
    vector2 = encoder.encode_expr(expr)
    assert vector1 == vector2


def test_encode_text_empty_returns_zero_vector():
    encoder = ExpressionEncoder(dimension=4)
    vector = encoder.encode_text("")
    assert all(value == 0.0 for value in vector.data)


def test_similarity_metric_identical_and_orthogonal():
    metric = SimilarityMetric()
    v1 = MLVector((1.0, 0.0))
    v2 = MLVector((1.0, 0.0))
    v3 = MLVector((0.0, 1.0))
    assert math.isclose(metric.similarity(v1, v2), 1.0, rel_tol=1e-9)
    assert math.isclose(metric.similarity(v1, v3), 0.0, rel_tol=1e-9)


class QueueMetric(SimilarityMetric):
    def __init__(self, values: List[float]) -> None:
        self._values = values

    def similarity(self, v1: MLVector, v2: MLVector) -> float:
        if not self._values:
            raise AssertionError("No similarity values left in queue.")
        return self._values.pop(0)


class DummyEncoder(ExpressionEncoder):
    def __init__(self) -> None:
        super().__init__(dimension=2)
        self._vectors = {
            "prev": MLVector((1.0, 0.0)),
            "cand": MLVector((0.0, 1.0)),
        }

    def encode_expr(self, expr: NormalizedExpr) -> MLVector:
        if expr["raw"] == "prev":
            return self._vectors["prev"]
        return self._vectors["cand"]

    def encode_text(self, text: str) -> MLVector:
        if not text:
            return MLVector.zeros(self.dimension)
        return MLVector((1.0, 1.0))


@pytest.mark.parametrize(
    ("expr_sim", "text_sim", "rule_ids", "expected_label"),
    [
        (1.0, 1.0, ("R1", "R1"), FuzzyLabel.EXACT),
        (0.9333333333, 1.0, ("R1", "R1"), FuzzyLabel.EQUIVALENT),
        (0.8, 0.8, ("R1", "R1"), FuzzyLabel.APPROX_EQ),
        (0.7, 0.7, ("R1", "R2"), FuzzyLabel.APPROX_EQ),  # Decision Engine accepts 0.66 as Approx Eq
        (0.0, 0.0, (None, None), FuzzyLabel.CONTRADICT),
    ],
)
def test_threshold_labels(expr_sim, text_sim, rule_ids, expected_label):
    metric = QueueMetric([expr_sim, text_sim])
    encoder = DummyEncoder()
    judge = FuzzyJudge(encoder=encoder, metric=metric)
    result = judge.judge_step(
        problem_expr=_expr("prev"),
        previous_expr=_expr("prev"),
        candidate_expr=_expr("cand"),
        applied_rule_id=rule_ids[0],
        candidate_rule_id=rule_ids[1],
        explain_text="details",
    )
    assert result["label"] == expected_label


def test_rule_similarity_cases():
    expr = _expr("prev")
    metric = QueueMetric([0.5, 0.0])
    judge = FuzzyJudge(encoder=DummyEncoder(), metric=metric)
    result = judge.judge_step(
        problem_expr=expr,
        previous_expr=expr,
        candidate_expr=_expr("cand"),
        applied_rule_id="R1",
        candidate_rule_id="R2",
    )
    assert result["score"]["rule_similarity"] == 0.5

    metric2 = QueueMetric([0.5, 0.0])
    judge2 = FuzzyJudge(encoder=DummyEncoder(), metric=metric2)
    result2 = judge2.judge_step(
        problem_expr=expr,
        previous_expr=expr,
        candidate_expr=_expr("cand"),
        applied_rule_id=None,
        candidate_rule_id=None,
    )
    assert result2["score"]["rule_similarity"] == 0.0

    metric3 = QueueMetric([0.5, 0.0])
    judge3 = FuzzyJudge(encoder=DummyEncoder(), metric=metric3)
    result3 = judge3.judge_step(
        problem_expr=expr,
        previous_expr=expr,
        candidate_expr=_expr("cand"),
        applied_rule_id="R1",
        candidate_rule_id="R1",
    )
    assert result3["score"]["rule_similarity"] == 1.0
