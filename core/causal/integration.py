"""Integration helpers that connect LearningLogger output with the causal engine."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .causal_analyzers import explain_error
from .causal_engine import CausalEngine
from .causal_types import CausalNodeType


def run_causal_analysis(
    records: List[Dict[str, Any]],
    *,
    include_graph: bool = False,
    rule_info: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[CausalEngine, Dict[str, Any]]:
    """
    Build a causal graph from LearningLogger records and return diagnostic data.

    Parameters
    ----------
    records:
        List of LearningLogger-style dictionaries produced by Evaluator/PolynomialEvaluator.
    include_graph:
        When True, attaches the serialized graph (nodes + edges) to the returned report.

    Returns
    -------
    (engine, report):
        - engine: CausalEngine instance populated with the provided records.
        - report: Dict containing `errors`, `explanations`, `records`, and optionally `graph`.
    """
    engine = CausalEngine()
    engine.ingest_log(records)
    export = engine.to_dict()
    error_ids: List[str] = list(export.get("errors", []))
    explanations = [explain_error(engine, error_id) for error_id in error_ids]
    report: Dict[str, Any] = {
        "errors": error_ids,
        "explanations": explanations,
        "records": export.get("records", []),
    }
    if include_graph:
        report["graph"] = export.get("graph")
    used_rules: Dict[str, Dict[str, Any]] = {}
    for node in engine.graph.nodes.values():
        if node.node_type != CausalNodeType.RULE_APPLICATION:
            continue
        rule_id = node.payload.get("rule_id")
        if not rule_id or rule_id in used_rules:
            continue
        rule_meta = node.payload.get("rule_meta")
        if isinstance(rule_meta, dict) and rule_meta:
            used_rules[rule_id] = rule_meta
        elif rule_info and rule_id in rule_info:
            used_rules[rule_id] = rule_info[rule_id]
    if used_rules:
        report["rule_details"] = used_rules
    return engine, report


__all__ = ["run_causal_analysis"]
