import json
from pathlib import Path

from core.fuzzy.config import FuzzyThresholdConfig
from core.fuzzy.encoder import ExpressionEncoder
from core.fuzzy.judge import FuzzyJudge
from core.fuzzy.metric import SimilarityMetric
from core.fuzzy.types import FuzzyLabel


def _load_cases() -> list[dict]:
    data_path = Path("docs/data/fuzzy_samples.json")
    return json.loads(data_path.read_text(encoding="utf-8"))


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
