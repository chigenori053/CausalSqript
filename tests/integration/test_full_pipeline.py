from __future__ import annotations
from typing import Any

from coherent.logic import CausalEngine
from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.fuzzy.types import FuzzyLabel, FuzzyResult, FuzzyScore
from coherent.engine.knowledge_registry import KnowledgeNode
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.parser import Parser
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.parser import Parser
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine


class StubKnowledgeRegistry:
    def __init__(self) -> None:
        self.node = KnowledgeNode(
            id="TEST-RULE",
            domain="arithmetic",
            category="equivalence",
            pattern_before="1 + 1",
            pattern_after="2",
            description="1+1=2",
        )

    def match(self, before: str, after: str, category: str | None = None) -> Any:
        if before.replace(" ", "") == "1+1" and after == "2":
            return self.node
        return None


class DummyEncoder:
    def normalize(self, text: str) -> Any:
        return {"raw": text, "sympy": text, "tokens": []}

class RecordingFuzzyJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.last_label: FuzzyLabel | None = None
        self.encoder = DummyEncoder()

    def judge_step(self, **kwargs) -> FuzzyResult:
        self.calls += 1
        self.last_label = FuzzyLabel.CONTRADICT
        return {
            "label": self.last_label,
            "score": {
                "expr_similarity": 0.6,
                "rule_similarity": 0.4,
                "text_similarity": 0.0,
                "combined_score": 0.5,
            },
            "reason": "recorded",
            "debug": {},
        }


def test_full_reasoning_pipeline():
    source = """
        mode: fuzzy
        problem: 1 + 1
        step: 2
        step: 4
        end: done
    """
    program = Parser(source).parse()
    program = Parser(source).parse()
    symbolic = SymbolicEngine()
    knowledge = StubKnowledgeRegistry()
    
    comp = ComputationEngine(symbolic)
    fuzzy = RecordingFuzzyJudge()
    
    val = ValidationEngine(comp, fuzzy_judge=fuzzy, knowledge_registry=knowledge)
    hint = HintEngine(comp)
    
    logger = LearningLogger()
    runtime = CoreRuntime(comp, val, hint, knowledge_registry=knowledge, learning_logger=logger)
    
    evaluator = Evaluator(program, runtime, learning_logger=logger)
    evaluator.run()
    records = logger.to_list()
    step_records = [record for record in records if record["phase"] == "step"]
    assert step_records[0]["status"] == "ok"
    assert step_records[0]["rule_id"] == "TEST-RULE"
    assert step_records[1]["status"] == "mistake"
    assert fuzzy.calls == 1
    assert any(record["phase"] == "fuzzy" for record in records)

    causal_engine = CausalEngine()
    causal_engine.ingest_log(records)
    error_id = causal_engine.to_dict()["errors"][-1]
    causes = causal_engine.why_error(error_id)
    assert any(node.node_id.startswith("step-2") for node in causes)
    fix_candidates = causal_engine.suggest_fix_candidates(error_id)
    assert fix_candidates and fix_candidates[0].node_id.startswith("step-2")

    intervention = {"phase": "step", "index": 2, "expression": "2"}
    cf_result = causal_engine.counterfactual_result(intervention, records)
    assert cf_result["changed"] is True
    assert cf_result["rerun_success"] is True
    assert cf_result["rerun_error"] is None
    assert cf_result["diff_steps"]
    assert cf_result["rerun_records"]
