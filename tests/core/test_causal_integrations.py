from coherent.logic import CausalEngine
from coherent.logic.causal_types import CausalNodeType
from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.parser import Parser
from coherent.engine.symbolic_engine import SymbolicEngine
from pathlib import Path


def _engine() -> SymbolicEvaluationEngine:
    sym = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), sym)
    return SymbolicEvaluationEngine(sym, registry)


def test_causal_engine_builds_graph_from_learning_logger():
    source = """
problem: (3 + 5) * 4
step: 8 * 4
step: 32 * 4
end: 32
"""
    program = Parser(source).parse()
    logger = LearningLogger()
    evaluator = Evaluator(program, _engine(), learning_logger=logger)
    evaluator.run()
    records = logger.to_list()
    step_records = [rec for rec in records if rec["phase"] == "step"]
    assert step_records[-1]["status"] == "mistake"
    records[1]["rule_id"] = "CUSTOM-RULE-1"

    engine = CausalEngine()
    engine.ingest_log(records)
    error_id = engine.to_dict()["errors"][-1]
    causes = engine.why_error(error_id)
    assert any(node.node_type == CausalNodeType.STEP for node in causes)
    rule_node = engine.graph.nodes.get("rule-CUSTOM-RULE-1")
    assert rule_node is not None
    parents = {parent.node_id for parent in engine.graph.get_parents("step-1")}
    assert "rule-CUSTOM-RULE-1" in parents
