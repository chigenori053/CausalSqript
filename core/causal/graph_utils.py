"""Helper functions to visualize causal graphs in notebooks or CLIs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def graph_to_text(graph: Dict[str, Any]) -> str:
    """Return a readable text representation of graph nodes/edges."""
    nodes = sorted(graph.get("nodes", []), key=lambda node: node.get("id", ""))
    edges = graph.get("edges", [])
    lines: List[str] = ["Nodes:"]
    for node in nodes:
        payload = node.get("payload") or {}
        summary = _node_summary(payload)
        lines.append(f"- {node.get('id')} [{node.get('type')}]{summary}")
    lines.append("Edges:")
    for edge in edges:
        lines.append(
            f"- {edge.get('source')} -> {edge.get('target')} [{edge.get('type')}]"
        )
    return "\n".join(lines)


def graph_to_dot(graph: Dict[str, Any]) -> str:
    """Return a Graphviz DOT string for visualization tools."""
    lines = ["digraph CausalGraph {"]
    for node in graph.get("nodes", []):
        label = node.get("id")
        node_type = node.get("type")
        lines.append(f'  "{label}" [label="{label}\\n({node_type})"];')
    for edge in graph.get("edges", []):
        lines.append(f'  "{edge.get("source")}" -> "{edge.get("target")}" [label="{edge.get("type")}"];')
    lines.append("}")
    return "\n".join(lines)


def _node_summary(payload: Dict[str, Any]) -> str:
    record = payload.get("record") or {}
    rendered = record.get("rendered")
    if rendered:
        return f": {rendered}"
    rule_id = payload.get("rule_id")
    if rule_id:
        return f": rule {rule_id}"
    return ""


__all__ = ["graph_to_text", "graph_to_dot"]
