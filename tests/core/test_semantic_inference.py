
import pytest
from typing import Optional
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry, KnowledgeNode
from coherent.engine.evaluator import Evaluator
from coherent.engine.parser import Parser
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.fuzzy.judge import FuzzyJudge
from coherent.engine.fuzzy.types import FuzzyResult, FuzzyLabel, FuzzyScore

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

    def match(self, before: str, after: str, **kwargs) -> Optional[KnowledgeNode]:
        # Simple stub matching
        if "x + x" in before and "2*x" in after:
            return self.node
        return None

class DummyEncoder:
    def normalize(self, text: str):
        return {"raw": text, "sympy": text, "tokens": []}

class StubFuzzyJudge(FuzzyJudge):
    def __init__(self):
        self.encoder = DummyEncoder()

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
    fuzzy_judge = StubFuzzyJudge()
    # Inject fuzzy_judge into ValidationEngine via re-init or just create it correctly above?
    # Better to create correctly.
    validation_engine = ValidationEngine(computation_engine, fuzzy_judge=fuzzy_judge)
    hint_engine = HintEngine(computation_engine)
    
    runtime = CoreRuntime(
        computation_engine=computation_engine,
        validation_engine=validation_engine,
        hint_engine=hint_engine,
        # No knowledge registry needed for this test
    )

    logger = LearningLogger()
    evaluator = Evaluator(program, runtime, learning_logger=logger)
    evaluator.run()

    records = logger.to_list()
    fuzzy_record = next((r for r in records if r["phase"] == "fuzzy"), None)
    
    assert fuzzy_record is not None
    assert fuzzy_record["status"] == "ok"
    assert "approx_eq" in fuzzy_record["rendered"]
