"""In-memory causal graph representation."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List

from .causal_types import CausalEdge, CausalNode


class CausalGraph:
    """Stores nodes and edges for MathLang causal reasoning."""

    def __init__(self) -> None:
        self.nodes: Dict[str, CausalNode] = {}
        self.out_edges: Dict[str, List[CausalEdge]] = {}
        self.in_edges: Dict[str, List[CausalEdge]] = {}

    def add_node(self, node: CausalNode) -> None:
        """Register a node if it does not already exist."""
        if node.node_id in self.nodes:
            return
        self.nodes[node.node_id] = node
        self.out_edges.setdefault(node.node_id, [])
        self.in_edges.setdefault(node.node_id, [])

    def add_edge(self, edge: CausalEdge) -> None:
        """Register a directed edge."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Edges can only be added between known nodes.")
        self.out_edges.setdefault(edge.source_id, []).append(edge)
        self.in_edges.setdefault(edge.target_id, []).append(edge)

    def get_parents(self, node_id: str) -> List[CausalNode]:
        """Return immediate predecessor nodes."""
        return [self.nodes[edge.source_id] for edge in self.in_edges.get(node_id, [])]

    def get_children(self, node_id: str) -> List[CausalNode]:
        """Return immediate successor nodes."""
        return [self.nodes[edge.target_id] for edge in self.out_edges.get(node_id, [])]

    def ancestors(self, node_id: str) -> List[CausalNode]:
        """Return all nodes reachable upstream."""
        return list(self._traverse(self.in_edges, node_id))

    def descendants(self, node_id: str) -> List[CausalNode]:
        """Return all nodes reachable downstream."""
        return list(self._traverse(self.out_edges, node_id, forward=True))

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        """Serialize the graph into simple dict format."""
        return {
            "nodes": [
                {"id": node_id, "type": node.node_type.value, "payload": node.payload}
                for node_id, node in self.nodes.items()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type.value,
                    "metadata": edge.metadata or {},
                }
                for edges in self.out_edges.values()
                for edge in edges
            ],
        }

    def _traverse(
        self,
        adjacency: Dict[str, List[CausalEdge]],
        start_node_id: str,
        *,
        forward: bool = False,
    ) -> Iterable[CausalNode]:
        """Generic traversal helper used by ancestors/descendants."""
        if start_node_id not in self.nodes:
            return []
        result: List[CausalNode] = []
        visited = {start_node_id}
        queue: deque[str] = deque([start_node_id])
        while queue:
            current = queue.popleft()
            for edge in adjacency.get(current, []):
                next_id = edge.target_id if forward else edge.source_id
                if next_id in visited:
                    continue
                visited.add(next_id)
                result.append(self.nodes[next_id])
                queue.append(next_id)
        return result


__all__ = ["CausalGraph"]
