import json
from coherent.engine.fuzzy.config import FuzzyThresholdConfig
from coherent.engine.fuzzy.encoder import ExpressionEncoder
from coherent.engine.fuzzy.judge import FuzzyJudge
from coherent.engine.fuzzy.metric import SimilarityMetric
from coherent.engine.fuzzy.types import FuzzyLabel


def _load_cases() -> list[dict]:
    # Inline test data to avoid external file dependency
    return [
        {
            "problem": "Calculate 1+1",
            "previous": "1 + 1",
            "candidate": "2"
        },
        {
            "problem": "Simplify 2x + 3x",
            "previous": "2x + 3x",
            "candidate": "5x"
        }
    ]


def test_fuzzy_judge_realistic_samples():
    judge = FuzzyJudge(
        encoder=ExpressionEncoder(),
        metric=SimilarityMetric(),
        thresholds=FuzzyThresholdConfig(exact=0.99, equivalent=0.95, approx_eq=0.7, analogous=0.4, contradict=0.2),
    )
    cases = _load_cases()
    assert cases, "expected fuzzy sample data"
    for case in cases:
        result = judge.judge_step(
            problem_expr={"raw": case["problem"], "sympy": case["problem"], "tokens": case["problem"].split()},
            previous_expr={"raw": case["previous"], "sympy": case["previous"], "tokens": case["previous"].split()},
            candidate_expr={"raw": case["candidate"], "sympy": case["candidate"], "tokens": case["candidate"].split()},
        )
        assert isinstance(result["label"], FuzzyLabel)
