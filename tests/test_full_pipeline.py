from __future__ import annotations

from core.causal import CausalEngine
from core.evaluator import Evaluator, SymbolicEvaluationEngine
from core.fuzzy.types import FuzzyLabel, FuzzyResult, FuzzyScore
from core.knowledge_registry import KnowledgeNode
from core.learning_logger import LearningLogger
from core.parser import Parser
from core.symbolic_engine import SymbolicEngine


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

    def match(self, before: str, after: str) -> KnowledgeNode | None:
        if before.replace(" ", "") == "1+1" and after == "2":
            return self.node
        return None


class RecordingFuzzyJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.last_label: FuzzyLabel | None = None

    def judge_step(self, **kwargs) -> FuzzyResult:
        self.calls += 1
        self.last_label = FuzzyLabel.APPROX_EQ
        return {
            "label": self.last_label,
            "score": FuzzyScore(
                expr_similarity=0.6,
                rule_similarity=0.4,
                text_similarity=0.0,
                combined_score=0.5,
            ),
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
    symbolic = SymbolicEngine()
    knowledge = StubKnowledgeRegistry()
    eval_engine = SymbolicEvaluationEngine(symbolic, knowledge)
    fuzzy = RecordingFuzzyJudge()
    logger = LearningLogger()
    evaluator = Evaluator(program, eval_engine, learning_logger=logger, fuzzy_judge=fuzzy)
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
