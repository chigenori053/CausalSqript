
import pytest
from typing import Optional
from core.core_runtime import CoreRuntime
from core.computation_engine import ComputationEngine
from core.validation_engine import ValidationEngine
from core.hint_engine import HintEngine
from core.symbolic_engine import SymbolicEngine
from core.knowledge_registry import KnowledgeRegistry, KnowledgeNode
from core.evaluator import Evaluator
from core.parser import Parser
from core.learning_logger import LearningLogger
from core.fuzzy.judge import FuzzyJudge
from core.fuzzy.types import FuzzyResult, FuzzyLabel, FuzzyScore

class StubKnowledgeRegistry(KnowledgeRegistry):
    def __init__(self):
        self.node = KnowledgeNode(
            id="TEST-RULE",
            domain="arithmetic",
            category="equivalence",
            pattern_before="x + x",
            pattern_after="2*x",
            description="Combine like terms"
        )

    def match(self, before: str, after: str) -> Optional[KnowledgeNode]:
        # Simple stub matching
        if "x + x" in before and "2*x" in after:
            return self.node
        return None

class StubFuzzyJudge(FuzzyJudge):
    def __init__(self):
        pass

    def judge_step(self, **kwargs) -> FuzzyResult:
        return {
            "label": FuzzyLabel.APPROX_EQ,
            "score": {
                "expr_similarity": 0.8,
                "rule_similarity": 0.0,
                "text_similarity": 0.0,
                "combined_score": 0.8,
            },
            "reason": "Stub fuzzy match",
            "debug": {}
        }

def test_core_runtime_identifies_rule():
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    validation_engine = ValidationEngine(computation_engine)
    hint_engine = HintEngine(computation_engine)
    knowledge_registry = StubKnowledgeRegistry()

    runtime = CoreRuntime(
        computation_engine=computation_engine,
        validation_engine=validation_engine,
        hint_engine=hint_engine,
        knowledge_registry=knowledge_registry
    )

    runtime.set("x + x")
    result = runtime.check_step("2*x")

    assert result["valid"] is True
    assert result["rule_id"] == "TEST-RULE"
    assert result["details"]["rule"]["description"] == "Combine like terms"

def test_evaluator_logs_rule_id():
    source = """
    problem: x + x
    step: 2*x
    end: done
    """
    program = Parser(source).parse()
    
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    validation_engine = ValidationEngine(computation_engine)
    hint_engine = HintEngine(computation_engine)
    knowledge_registry = StubKnowledgeRegistry()

    runtime = CoreRuntime(
        computation_engine=computation_engine,
        validation_engine=validation_engine,
        hint_engine=hint_engine,
        knowledge_registry=knowledge_registry
    )

    logger = LearningLogger()
    evaluator = Evaluator(program, runtime, learning_logger=logger)
    evaluator.run()

    records = logger.to_list()
    step_record = next(r for r in records if r["phase"] == "step")
    
    assert step_record["status"] == "ok"
    assert step_record["rule_id"] == "TEST-RULE"

def test_evaluator_invokes_fuzzy_judge_on_mistake():
    source = """
    mode: fuzzy
    problem: x + x
    step: x * x
    end: done
    """
    program = Parser(source).parse()
    
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    validation_engine = ValidationEngine(computation_engine)
    hint_engine = HintEngine(computation_engine)
    
    runtime = CoreRuntime(
        computation_engine=computation_engine,
        validation_engine=validation_engine,
        hint_engine=hint_engine
        # No knowledge registry needed for this test
    )

    logger = LearningLogger()
    fuzzy_judge = StubFuzzyJudge()
    evaluator = Evaluator(program, runtime, learning_logger=logger, fuzzy_judge=fuzzy_judge)
    evaluator.run()

    records = logger.to_list()
    fuzzy_record = next((r for r in records if r["phase"] == "fuzzy"), None)
    
    assert fuzzy_record is not None
    assert fuzzy_record["status"] == "ok"
    assert "approx_eq" in fuzzy_record["rendered"]
