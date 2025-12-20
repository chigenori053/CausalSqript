from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict

class TaskType(Enum):
    SOLVE = "solve"
    VERIFY = "verify"
    HINT = "hint"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"

class MathDomain(Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    GEOMETRY = "geometry"
    UNKNOWN = "unknown"

class GoalType(Enum):
    FINAL_VALUE = "final_value" # E.g., "Calculate 1+1" -> 2
    TRANSFORMATION = "transformation" # E.g., "Simplify", "Factorize"
    PROOF = "proof" # E.g., "Prove that..."
    GRAPH = "graph" # E.g., "Plot..."

@dataclass
class InputItem:
    type: str # "expression", "number", "text", "reference"
    value: Any
    
@dataclass
class Constraints:
    symbolic_only: bool = False
    numeric_precision: Optional[int] = None
    steps_required: bool = True
    
@dataclass
class LanguageMeta:
    original_language: str
    original_text: str
    ambiguity_score: float = 0.0
    detected_intent_confidence: float = 1.0

@dataclass
class SemanticIR:
    """
    Semantic Intermediate Representation (SIR).
    Represents the structured meaning of a natural language input.
    """
    task: TaskType
    math_domain: MathDomain = MathDomain.UNKNOWN
    goal: Optional[GoalType] = None
    inputs: List[InputItem] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    explanation_level: int = 1 # 0: None, 1: Standard, 2: Detailed
    language_meta: Optional[LanguageMeta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "math_domain": self.math_domain.value,
            "goal": self.goal.value if self.goal else None,
            "inputs": [vars(i) for i in self.inputs],
            "constraints": vars(self.constraints),
            "explanation_level": self.explanation_level,
            "language_meta": vars(self.language_meta) if self.language_meta else None
        }
