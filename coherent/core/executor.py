from typing import Any, Dict
from coherent.core.action import Action
from coherent.core.action_types import ActionType
from coherent.core.state import State
from coherent.engine.core_runtime import CoreRuntime
# Future: Import specific handlers

class ActionExecutor:
    """
    Executes discrete Actions and updates the State.
    Acts as the bridge between the Abstract Action Logic and the Concrete Runtime.
    """
    def __init__(self, runtime: CoreRuntime):
        self.runtime = runtime

    def execute(self, action: Action, state: State) -> Any:
        """
        Applies a single action to the current state.
        Returns the result of the execution.
        """
        # 1. Dispatch based on Type
        if action.type == ActionType.APPLY_RULE:
            return self._handle_apply_rule(action, state)
        elif action.type == ActionType.FINAL:
            return self._handle_final(action, state)
        elif action.type == ActionType.RECALL:
            return self._handle_recall(action, state)
        elif action.type == ActionType.CALL_TOOL:
            return self._handle_tool(action, state)
        else:
            return {"status": "error", "message": f"Unsupported action type: {action.type}"}

    def _handle_apply_rule(self, action: Action, state: State) -> Dict[str, Any]:
        """
        Invokes the CoreRuntime to apply a mathematical rule.
        """
        target_expr = action.inputs.get("target") or state.current_expression
        rule_id = action.evidence.get("rule_id")
        
        # Note: Runtime currently does check_step (validation), not direct rule application by ID 
        # unless we use the SymbolicEngine directly. 
        # For now, we simulate this by assuming the 'inputs' contains the 'next_state' 
        # OR we try to apply the rule if the engine supports it.
        # Ideally, hypothesis generator ALREADY generated the next state.
        
        next_state = action.inputs.get("next_state")
        
        if next_state:
            # Validate the transition
            # Temporarily set current expr to target if different
            original = self.runtime._current_expr
            self.runtime.set(target_expr)
            
            result = self.runtime.check_step(next_state)
            
            # Restore if needed (or keep if we want side effects? Executor should ideally be stateless or careful)
            # But State object is what tracks output.
            if result["valid"]:
                state.update_expression(next_state)
                state.status = "ACTIVE"
            
            return result
        else:
            return {"status": "error", "message": "Applying rule requires 'next_state' in inputs for now."}

    def _handle_final(self, action: Action, state: State) -> Dict[str, Any]:
        """
        Marks the completion of the task.
        """
        answer = action.inputs.get("answer")
        state.update_expression(answer)
        state.status = "SOLVED"
        return {"status": "success", "answer": answer}
        
    def _handle_recall(self, action: Action, state: State) -> Dict[str, Any]:
        # Placeholder for optical recall
        query = action.inputs.get("query")
        return {"status": "mock", "message": f"Recalled {query}"}

    def _handle_tool(self, action: Action, state: State) -> Dict[str, Any]:
        # Placeholder for tool execution
        return {"status": "mock", "message": "Tool called"}
