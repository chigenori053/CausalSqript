"""Text-based causal graph preview for notebooks."""

from __future__ import annotations

from typing import Dict, Any, Iterable

from causalscript.core.causal.graph_utils import graph_to_text


def graph_to_text_report(graph_dict: Dict[str, Any]) -> str:
    """Wrapper kept for backwards compatibility."""
    return graph_to_text(graph_dict)
