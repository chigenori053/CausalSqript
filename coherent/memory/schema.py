"""
Data schemas for the Memory Module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExperienceEntry:
    """Represents a past calculation step (Experience)."""
    id: str
    original_expr: str
    next_expr: str
    rule_id: str
    result_label: str  # EXACT, ANALOGOUS, CONTRADICT
    category: str
    score: float
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for vector store."""
        meta = {
            "original_expr": self.original_expr,
            "next_expr": self.next_expr,
            "rule_id": self.rule_id,
            "result_label": self.result_label,
            "category": self.category,
            "score": self.score,
        }
        meta.update(self.metadata)
        return meta

@dataclass
class KnowledgeEntry:
    """Represents a rule (Knowledge)."""
    id: str
    description: str
    pattern_before: str
    pattern_after: str
    domain: str
    priority: int
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for vector store."""
        meta = {
            "description": self.description,
            "pattern_before": self.pattern_before,
            "pattern_after": self.pattern_after,
            "domain": self.domain,
            "priority": self.priority,
        }
        meta.update(self.metadata)
        return meta
