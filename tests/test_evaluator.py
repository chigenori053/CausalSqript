import pytest

from pathlib import Path

from core.evaluator import Evaluator, SymbolicEvaluationEngine, Engine
from core.errors import MissingProblemError, SyntaxError as DslSyntaxError
from core.fuzzy.types import FuzzyLabel, FuzzyResult, FuzzyScore
from core.knowledge_registry import KnowledgeRegistry
from core.learning_logger import LearningLogger
from core.parser import Parser
from core.symbolic_engine import SymbolicEngine


def _program_from_source(source: str):
    return Parser(source).parse()


def _engine():
    sym = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), sym)
    return SymbolicEvaluationEngine(sym, registry)


def test_evaluator_records_problem_step_end():
    program = _program_from_source(
        """
        prepare:
            - x = 1
        problem: (x + 1) * (x + 2)
        step: x^2 + 3*x + 2
        end: x^2 + 3*x + 2
        """
    )
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    assert evaluator.run() is True
    records = logger.to_list()
    assert [record["phase"] for record in records] == ["prepare", "problem", "step", "end"]
    assert all(record["status"] == "ok" for record in records)


def test_evaluator_records_mistake_for_invalid_step():
    program = _program_from_source(
        """
problem: 1 + 1
step: 3
end: done
"""
    )
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    assert evaluator.run() is True
    records = logger.to_list()
    step_record = next(record for record in records if record["phase"] == "step")
    assert step_record["status"] == "mistake"
    assert step_record["meta"]["reason"] == "invalid_step"


def test_evaluator_records_mistake_for_end_mismatch():
    program = _program_from_source(
        """
problem: 1 + 1
step: 2
end: 3
"""
    )
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    assert evaluator.run() is True
    records = logger.to_list()
    end_record = records[-1]
    assert end_record["phase"] == "end"
    assert end_record["status"] == "mistake"
    assert end_record["meta"]["reason"] == "final_result_mismatch"


def test_evaluator_requires_problem():
    with pytest.raises(DslSyntaxError):
        _program_from_source(
            """
step: 1 + 1
end: done
"""
        )


def test_evaluator_fatal_when_step_precedes_problem():
    program = _program_from_source(
        """
step: 3
problem: 1 + 1
end: done
"""
    )
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    with pytest.raises(MissingProblemError):
        evaluator.run()
    fatal = logger.to_list()[-1]
    assert fatal["status"] == "fatal"
    assert fatal["phase"] == "step"


class AlwaysInvalidEngine(Engine):
    def __init__(self) -> None:
        self._current = None

    def set(self, expr: str) -> None:
        self._current = expr

    def check_step(self, expr: str) -> dict:
        return {
            "before": self._current or "",
            "after": expr,
            "valid": False,
            "rule_id": "RULE-1",
            "details": {},
        }

    def finalize(self, expr: str | None) -> dict:
        return {"before": self._current, "after": expr, "valid": False, "rule_id": None, "details": {}}


class StubFuzzyJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge_step(self, **kwargs) -> FuzzyResult:
        self.calls += 1
        return {
            "label": FuzzyLabel.UNKNOWN,
            "score": FuzzyScore(
                expr_similarity=0.5,
                rule_similarity=0.0,
                text_similarity=0.0,
                combined_score=0.5,
            ),
            "reason": "stub",
            "debug": {},
        }


def test_evaluator_logs_fuzzy_when_invalid_step():
    source = """
        mode: fuzzy
        problem: 1 + 1
        step: 3
        end: done
        """
    program = Parser(source).parse()
    fuzzy = StubFuzzyJudge()
    logger = LearningLogger()
    evaluator = Evaluator(program, AlwaysInvalidEngine(), learning_logger=logger, fuzzy_judge=fuzzy)
    assert evaluator.run() is True
    assert fuzzy.calls == 1
    assert any(record["phase"] == "fuzzy" for record in logger.to_list())


class RecordingFuzzyJudge:
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
            "reason": "recorded",
            "debug": {"candidate_raw": kwargs["candidate_expr"]["raw"]},
        }


def _assert_fuzzy_label(logger: LearningLogger, expected: FuzzyLabel) -> None:
    entries = [r for r in logger.to_list() if r["phase"] == "fuzzy"]
    assert entries, "expected fuzzy log entry"
    assert entries[-1]["meta"]["label"] == expected


def test_arithmetic_equivalent_step():
    source = """
problem: (3 + 5) * 2
step: 16
end: 16
"""
    program = Parser(source).parse()
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    assert evaluator.run() is True
    assert logger.to_list()[-1]["phase"] == "end"


def test_arithmetic_non_equivalent_triggers_fuzzy():
    source = """
        mode: fuzzy
        problem: (3 + 5) * 2
        step: 15
        end: done
        """
    program = Parser(source).parse()
    fuzzy = RecordingFuzzyJudge(FuzzyLabel.ANALOGOUS)
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger, fuzzy_judge=fuzzy)
    assert evaluator.run() is True
    assert fuzzy.calls == 1
    _assert_fuzzy_label(logger, FuzzyLabel.ANALOGOUS)


def test_polynomial_non_equivalent_triggers_fuzzy():
    source = """
        mode: fuzzy
        problem: (x + 1) * (x + 2)
        prepare:
            - x = 1
        step: x^2 + 2*x + 1
        end: done
        """
    program = Parser(source).parse()
    fuzzy = RecordingFuzzyJudge(FuzzyLabel.APPROX_EQ)
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger, fuzzy_judge=fuzzy)
    assert evaluator.run() is True
    _assert_fuzzy_label(logger, FuzzyLabel.APPROX_EQ)


def test_fraction_non_equivalent_triggers_fuzzy():
    source = """
        mode: fuzzy
        problem: 1/2 + 1/3
        step: 2/3
        end: done
        """
    program = Parser(source).parse()
    fuzzy = RecordingFuzzyJudge(FuzzyLabel.CONTRADICT)
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger, fuzzy_judge=fuzzy)
    assert evaluator.run() is True
    _assert_fuzzy_label(logger, FuzzyLabel.CONTRADICT)
