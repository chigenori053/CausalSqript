import pytest
from typing import Any
from pathlib import Path

from causalscript.core.evaluator import Evaluator, SymbolicEvaluationEngine, Engine
from causalscript.core.errors import MissingProblemError, SyntaxError as DslSyntaxError
from causalscript.core.fuzzy.types import FuzzyLabel, FuzzyResult
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.learning_logger import LearningLogger
from causalscript.core.parser import Parser
from causalscript.core.symbolic_engine import SymbolicEngine

def _program_from_source(source: str):
    return Parser(source).parse()

def _engine(fuzzy_judge=None):
    from causalscript.core.core_runtime import CoreRuntime
    from causalscript.core.computation_engine import ComputationEngine
    from causalscript.core.validation_engine import ValidationEngine
    from causalscript.core.hint_engine import HintEngine
    
    sym = SymbolicEngine()
    comp = ComputationEngine(sym)
    registry = KnowledgeRegistry(Path("causalscript/core/knowledge"), sym)
    
    val = ValidationEngine(comp, fuzzy_judge=fuzzy_judge, knowledge_registry=registry)
    hint = HintEngine(comp)
    
    return CoreRuntime(comp, val, hint, knowledge_registry=registry)

def create_evaluator_v2(source: str) -> Evaluator:
    # Helper for v2 tests
    parser = Parser(source)
    program = parser.parse()
    # Simple SymbolicEvaluationEngine for v2 tests that don't need full CoreRuntime stack
    symbolic_engine = SymbolicEngine()
    engine = SymbolicEvaluationEngine(symbolic_engine)
    logger = LearningLogger()
    return Evaluator(program, engine, logger)

# --- Original Tests ---

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


class DummyEncoder:
    def normalize(self, text: str) -> Any:
        return {"raw": text, "sympy": text, "tokens": []}

class StubFuzzyJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.encoder = DummyEncoder()

    def judge_step(self, **kwargs) -> FuzzyResult:
        self.calls += 1
        return {
            "label": FuzzyLabel.UNKNOWN,
            "score": {"combined_score": 0.5},
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
    engine = _engine(fuzzy_judge=fuzzy)
    
    evaluator = Evaluator(program, engine, learning_logger=logger)
    assert evaluator.run() is True
    assert fuzzy.calls == 1
    assert any(record["phase"] == "fuzzy" for record in logger.to_list())


# --- V2.5 Tests ---

def test_evaluator_prepare_block():
    source = """
problem: x + y
prepare:
    - x = 10
    - y = 20
step:
    after: 30
end: 30
"""
    evaluator = create_evaluator_v2(source)
    assert evaluator.run()
    assert evaluator.engine._context == {"x": 10, "y": 20}


def test_evaluator_counterfactual_block():
    source = """
problem: 3 * y
prepare:
    - y = 2
step: 3 * y
end: 6
counterfactual:
    assume:
        y: 5
    expect: 3 * y
"""
    evaluator = create_evaluator_v2(source)
    assert evaluator.run()
    
    log = evaluator.learning_logger.to_list()
    cf_log = next(item for item in log if item["phase"] == "counterfactual")
    assert cf_log["status"] == "ok"
    assert float(cf_log["meta"]["result"]) == 15.0

def test_evaluator_mode_block():
    source = """
mode: fuzzy
problem: 1+1
step: 2
end: 2
"""
    evaluator = create_evaluator_v2(source)
    assert evaluator.run()
    assert evaluator._mode == "fuzzy"
