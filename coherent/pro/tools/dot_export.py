"""DOT export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def export_to_dot(graph_dict: Dict[str, Any], output_path: Path) -> None:
    """Persist a very small DOT representation."""
    lines = ["digraph causal {"]
    for edge in graph_dict.get("edges", []):
        lines.append(f'    "{edge["source_id"]}" -> "{edge["target_id"]}";')
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
