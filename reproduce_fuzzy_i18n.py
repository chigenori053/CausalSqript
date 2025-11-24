
import sys
import os
from unittest.mock import MagicMock

# Ensure we can import core
sys.path.append(os.getcwd())

from core.fuzzy.judge import FuzzyJudge, FuzzyThresholdConfig
from core.fuzzy.types import NormalizedExpr, FuzzyLabel
from core.fuzzy.encoder import ExpressionEncoder
from core.fuzzy.metric import SimilarityMetric
from core.causal.causal_analyzers import explain_error
from core.causal.causal_types import CausalNodeType

# Mock Encoder and Metric
class MockEncoder(ExpressionEncoder):
    def encode_expr(self, expr: NormalizedExpr) -> list[float]:
        return [0.0]
    def encode_text(self, text: str) -> list[float]:
        return [0.0]

class MockMetric(SimilarityMetric):
    def __init__(self, score: float):
        self.score = score
    def similarity(self, v1: list[float], v2: list[float]) -> float:
        return self.score

def test_judge(score: float, label_name: str):
    judge = FuzzyJudge(
        encoder=MockEncoder(),
        metric=MockMetric(score),
        thresholds=FuzzyThresholdConfig()
    )
    expr = NormalizedExpr(raw="x", sympy="x", tokens=["x"])
    
    result = judge.judge_step(
        problem_expr=expr,
        previous_expr=expr,
        candidate_expr=expr,
        applied_rule_id="rule1",
        candidate_rule_id="rule1", 
        explain_text="explanation" 
    )
    
    print(f"--- Testing for {label_name} (Input Score: {score}) ---")
    print(f"Result Reason: {result['reason']}")
    print(f"Result Label: {result['label']}")
    print("-" * 20)

# Test Exact: Score 1.0 -> 0.8*1.0 + 0.2 = 1.0 >= 0.95 -> EXACT
test_judge(1.0, "EXACT")

# Test Approx: We need combined ~ 0.75.
# 0.8*s + 0.2 = 0.75 -> 0.8*s = 0.55 -> s = 0.6875
test_judge(0.69, "APPROX_EQ")

# Test Analogous: We need combined ~ 0.55.
# 0.8*s + 0.2 = 0.55 -> 0.8*s = 0.35 -> s = 0.4375
test_judge(0.44, "ANALOGOUS")
