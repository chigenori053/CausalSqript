from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from coherent.core.action_types import ActionType

@dataclass
class Action:
    """
    Represents a discrete reasoning step predicted by the system.
    This is the standard unit of output for the Reasoning LM.
    """
    type: ActionType
    name: str  # Human-readable name (e.g., "distribute_property", "calculator")
    inputs: Dict[str, Any] = field(default_factory=dict) # Arguments for the action
    
    # Meta-information for learning & confidence calibration
    confidence: float = 1.0
    ambiguity: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict) # e.g., {"rule_id": "...", "memory_hit": True}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "inputs": self.inputs,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "evidence": self.evidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        return cls(
            type=ActionType(data["type"]),
            name=data["name"],
            inputs=data.get("inputs", {}),
            confidence=data.get("confidence", 1.0),
            ambiguity=data.get("ambiguity", 0.0),
            evidence=data.get("evidence", {})
        )
