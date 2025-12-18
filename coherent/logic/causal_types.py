"""Data types for the MathLang causal inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class CausalNodeType(str, Enum):
    """Supported node categories in the causal graph."""

    PROBLEM = "problem"
    STEP = "step"
    END = "end"
    EXPLAIN = "explain"
    ERROR = "error"
    RULE_APPLICATION = "rule_application"


class CausalEdgeType(str, Enum):
    """Logical relation types encoded in the graph."""

    STEP_TRANSITION = "step_transition"
    RULE_USAGE = "rule_usage"
    ERROR_CAUSE = "error_cause"
    EXPLAIN_LINK = "explain_link"


@dataclass(frozen=True)
class CausalNode:
    """Description of a single causal node."""

    node_id: str
    node_type: CausalNodeType
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CausalEdge:
    """Directed relation between two causal nodes."""

    source_id: str
    target_id: str
    edge_type: CausalEdgeType
    metadata: Optional[Dict[str, Any]] = None


__all__ = [
    "CausalNode",
    "CausalNodeType",
    "CausalEdge",
    "CausalEdgeType",
]
