from copy import deepcopy

from coherent.logic import CausalEngine, CausalGraph, run_causal_analysis
from coherent.logic.causal_types import CausalEdge, CausalEdgeType, CausalNode, CausalNodeType


def _base_records():
    return [
        {"phase": "problem", "expression": "(3 + 5) * 4", "rendered": "problem", "status": "ok"},
        {
            "phase": "step",
            "expression": "8 * 4",
            "rendered": "step 1",
            "status": "ok",
            "rule_id": "ARITH-ADD-001",
            "meta": {
                "rule": {
                    "id": "ARITH-ADD-001",
                    "description": "Addition commutativity",
                    "domain": "arithmetic",
                    "category": "addition",
                    "pattern_before": "a + b",
                    "pattern_after": "b + a",
                }
            },
        },
        {
            "phase": "step",
            "expression": "32 * 4",
            "rendered": "step 2",
            "status": "mistake",
            "meta": {"reason": "invalid_step"},
        },
        {"phase": "end", "expression": "32", "rendered": "end", "status": "ok"},
    ]


def test_causal_graph_traversal_helpers():
    graph = CausalGraph()
    problem = CausalNode("problem-1", CausalNodeType.PROBLEM, {})
    step1 = CausalNode("step-1", CausalNodeType.STEP, {})
    step2 = CausalNode("step-2", CausalNodeType.STEP, {})
    graph.add_node(problem)
    graph.add_node(step1)
    graph.add_node(step2)
    graph.add_edge(CausalEdge("problem-1", "step-1", CausalEdgeType.STEP_TRANSITION))
    graph.add_edge(CausalEdge("step-1", "step-2", CausalEdgeType.STEP_TRANSITION))
    ancestors = [node.node_id for node in graph.ancestors("step-2")]
    assert "step-1" in ancestors
    assert "problem-1" in ancestors
    descendants = [node.node_id for node in graph.descendants("problem-1")]
    assert "step-2" in descendants


def test_engine_ingest_creates_nodes_and_edges():
    engine = CausalEngine()
    records = _base_records()
    engine.ingest_log(records)
    graph_dict = engine.to_dict()["graph"]
    node_types = {node["id"]: node["type"] for node in graph_dict["nodes"]}
    assert "problem-1" in node_types
    assert node_types["step-1"] == CausalNodeType.STEP.value
    assert any(edge["type"] == CausalEdgeType.STEP_TRANSITION.value for edge in graph_dict["edges"])
    parents_step2 = {parent.node_id for parent in engine.graph.get_parents("step-2")}
    assert "step-1" in parents_step2
    parents_step1 = {parent.node_id for parent in engine.graph.get_parents("step-1")}
    assert "rule-ARITH-ADD-001" in parents_step1


def test_why_error_returns_most_recent_invalid_step_first():
    engine = CausalEngine()
    engine.ingest_log(_base_records())
    error_id = engine.to_dict()["errors"][-1]
    causes = engine.why_error(error_id)
    assert causes, "expected at least one cause"
    assert causes[0].node_id.startswith("step-2")


def test_suggest_fix_candidates_prioritizes_invalid_steps():
    engine = CausalEngine()
    engine.ingest_log(_base_records())
    error_id = engine.to_dict()["errors"][-1]
    candidates = engine.suggest_fix_candidates(error_id)
    assert candidates
    assert candidates[0].node_id.startswith("step-2")


def test_counterfactual_result_reports_changes():
    engine = CausalEngine()
    records = _base_records()
    engine.ingest_log(records)
    intervention = {"phase": "step", "index": 2, "expression": "8 * 4"}
    result = engine.counterfactual_result(intervention, deepcopy(records))
    assert result["changed"] is True
    assert result["diff_steps"][0]["old_expression"] == "32 * 4"
    assert result["diff_steps"][0]["action"] == "replace"
    assert result["new_end_expr"] == "32"
    assert result["rerun_success"] is True
    assert result["rerun_error"] is None
    assert result["rerun_records"]
    assert result["rerun_first_error"] is None
    assert result["rerun_step_outcomes"]


def test_counterfactual_supports_multiple_interventions():
    engine = CausalEngine()
    records = _base_records()
    engine.ingest_log(records)
    interventions = [
        {"phase": "step", "index": 2, "expression": "8 * 4"},
        {"phase": "end", "index": 1, "expression": "40"},
    ]
    result = engine.counterfactual_result(interventions, deepcopy(records))
    assert result["changed"] is True
    assert result["diff_end"][0]["action"] == "replace"
    assert result["interventions"][0]["phase"] == "step"
    assert result["interventions"][1]["phase"] == "end"
    assert result["rerun_last_phase"] in {"end", "error"}


def test_counterfactual_insert_and_delete():
    engine = CausalEngine()
    records = _base_records()
    engine.ingest_log(records)
    interventions = [
        {"phase": "step", "index": 2, "action": "delete"},
        {"phase": "step", "index": 1, "action": "insert_after", "expression": "9 * 4", "rendered": "manual"},
    ]
    result = engine.counterfactual_result(interventions, deepcopy(records))
    assert result["changed"] is True
    assert any(step["action"] == "delete" for step in result["diff_steps"])
    assert any(step["action"] == "insert_after" for step in result["diff_steps"])
    assert result["rerun_step_outcomes"]


def test_run_causal_analysis_reports_errors_with_rule_details():
    engine, report = run_causal_analysis(_base_records())
    assert isinstance(engine, CausalEngine)
    assert report["errors"]
    assert report["explanations"]
    assert report["explanations"][0]["cause_steps"]
    assert report["rule_details"]["ARITH-ADD-001"]["description"] == "Addition commutativity"


def test_run_causal_analysis_rule_info_fallback():
    records = _base_records()
    records[1].pop("meta", None)
    rule_info = {
        "ARITH-ADD-001": {"description": "Addition commutativity"}
    }
    _, report = run_causal_analysis(records, rule_info=rule_info)
    assert report["rule_details"]["ARITH-ADD-001"]["description"] == "Addition commutativity"
