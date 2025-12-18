from coherent.engine.parser import Parser
from coherent.engine.polynomial_evaluator import PolynomialEvaluator
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.fuzzy.types import FuzzyLabel, FuzzyResult, FuzzyScore
from coherent.engine.fuzzy.types import FuzzyLabel, FuzzyResult, FuzzyScore


def _normalizer():
    engine = SymbolicEngine()

    if engine.has_sympy():

        def normalize(expr: str) -> str:
            internal = engine.to_internal(expr)
            return str(internal.expand())  # type: ignore[attr-defined]

        return normalize

    assignments = [
        {"x": 1, "y": 2, "z": 3},
        {"x": 2, "y": 1, "z": 0},
    ]

    def normalize(expr: str) -> str:
        values = [str(engine.evaluate_numeric(expr, assignment)) for assignment in assignments]
        return "|".join(values)

    return normalize


class RecordingPolynomialFuzzyJudge:
    def __init__(self, label: FuzzyLabel) -> None:
        self.label = label
        self.calls = 0

    def judge_step(self, **kwargs) -> FuzzyResult:
        self.calls += 1
        return {
            "label": self.label,
            "score": FuzzyScore(
                expr_similarity=0.6,
                rule_similarity=0.0,
                text_similarity=0.0,
                combined_score=0.6,
            ),
            "reason": "poly",
            "debug": {"candidate_raw": kwargs["candidate_expr"]["raw"]},
        }


def test_polynomial_evaluator_accepts_equivalent_steps():
    source = """
problem: (x + 1) * (x + 2)
step: x^2 + 3*x + 2
end: x^2 + 3*x + 2
"""
    program = Parser(source).parse()
    logger = LearningLogger()
    evaluator = PolynomialEvaluator(program, normalizer=_normalizer(), learning_logger=logger)
    assert evaluator.run() is True
    records = logger.to_list()
    assert [r["phase"] for r in records] == ["problem", "step", "end"]
    assert all(record["status"] == "ok" for record in records)


def test_polynomial_evaluator_detects_invalid_step():
    source = """
problem: (x + 1) * (x + 2)
step: x^2 + 2*x + 1
end: done
"""
    program = Parser(source).parse()
    logger = LearningLogger()
    evaluator = PolynomialEvaluator(program, normalizer=_normalizer(), learning_logger=logger)
    assert evaluator.run() is True
    step_record = next(record for record in logger.to_list() if record["phase"] == "step")
    assert step_record["status"] == "mistake"
    assert step_record["meta"]["reason"] == "invalid_step"


def test_polynomial_evaluator_triggers_fuzzy_when_invalid_step():
    source = """
mode: fuzzy
problem: (x + 1) * (x + 2)
step: x^2 + 2*x + 1
end: done
"""
    program = Parser(source).parse()
    logger = LearningLogger()
    fuzzy = RecordingPolynomialFuzzyJudge(FuzzyLabel.APPROX_EQ)
    evaluator = PolynomialEvaluator(
        program,
        normalizer=_normalizer(),
        learning_logger=logger,
        fuzzy_judge=fuzzy,
    )
    assert evaluator.run() is True
    assert fuzzy.calls == 1
    assert any(record["phase"] == "fuzzy" for record in logger.to_list())
