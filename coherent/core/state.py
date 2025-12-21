from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from coherent.engine.language.semantic_types import TaskType, InputItem

@dataclass
class State:
    """
    Represents the full context of a problem-solving session.
    Used as input for the 'Next Action Prediction'.
    """
    # Problem Context
    task_goal: TaskType
    initial_inputs: List[InputItem]
    
    # Current Status
    current_expression: str  # Or AST representation
    step_history: List[Dict[str, Any]] = field(default_factory=list) # Log of executed actions
    
    # Memory Context (Phase 2/3)
    active_embeddings: Optional[Any] = None # Holographic tensors
    memory_context: Dict[str, Any] = field(default_factory=dict) # Retrieved items
    
    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    status: str = "ACTIVE" # ACTIVE, SOLVED, FAILED, WAITING
    
    def update_expression(self, new_expr: str):
        """Updates the current mathematical state."""
        self.current_expression = new_expr
        
    def add_history(self, action_dict: Dict[str, Any], result: Any):
        """Logs an action and its result."""
        self.step_history.append({
            "action": action_dict,
            "result": result
        })
