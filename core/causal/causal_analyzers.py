"""Convenience helpers for common causal queries."""

from __future__ import annotations

from typing import Any, Dict, List

from .causal_engine import CausalEngine
from .causal_types import CausalNodeType


def explain_error(engine: CausalEngine, error_node_id: str) -> Dict[str, Any]:
    """Produce a human-readable explanation for an error node."""
    error_node = engine.graph.nodes.get(error_node_id)
    if error_node is None:
        return {
            "error_id": error_node_id,
            "cause_steps": [],
            "cause_rules": [],
            "message": "Error node not found.",
        }
    causes = engine.why_error(error_node_id)
    step_nodes = [node for node in causes if node.node_type == CausalNodeType.STEP]
    rule_nodes = [node for node in causes if node.node_type == CausalNodeType.RULE_APPLICATION]
    step_ids = [node.node_id for node in step_nodes]
    rule_ids = [node.payload.get("rule_id") for node in rule_nodes if node.payload.get("rule_id")]

    parts: List[str] = []
    if step_ids:
        parts.append(f"Check steps: {', '.join(step_ids)}")
    if rule_ids:
        parts.append(f"Review rules: {', '.join(rule_ids)}")
    message = "; ".join(parts) if parts else "No upstream cause identified."

    return {
        "error_id": error_node_id,
        "cause_steps": step_ids,
        "cause_rules": rule_ids,
        "message": message,
    }


__all__ = ["explain_error"]
