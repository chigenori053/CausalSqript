import pytest
from unittest.mock import MagicMock
from coherent.core.action_types import ActionType
from coherent.core.action import Action
from coherent.core.state import State
from coherent.core.executor import ActionExecutor
from coherent.engine.language.semantic_types import TaskType

def test_action_creation_and_serialization():
    action = Action(
        type=ActionType.APPLY_RULE,
        name="distribute",
        inputs={"target": "x(y+z)", "next_state": "xy+xz"},
        confidence=0.9,
        evidence={"rule_id": "distribute_rule"}
    )
    
    data = action.to_dict()
    assert data["type"] == "APPLY_RULE"
    assert data["name"] == "distribute"
    assert data["confidence"] == 0.9
    
    reconstituted = Action.from_dict(data)
    assert reconstituted.type == ActionType.APPLY_RULE
    assert reconstituted.inputs["next_state"] == "xy+xz"

def test_state_management():
    state = State(
        task_goal=TaskType.SOLVE,
        initial_inputs=[],
        current_expression="x+1=2"
    )
    
    assert state.status == "ACTIVE"
    state.update_expression("x=1")
    assert state.current_expression == "x=1"
    
    state.add_history({"type": "APPLY_RULE"}, {"valid": True})
    assert len(state.step_history) == 1
    assert state.step_history[0]["result"]["valid"] is True

def test_executor_flow():
    # Mock runtime
    mock_runtime = MagicMock()
    # Mock validate result
    mock_runtime.check_step.return_value = {"valid": True, "status": "correct"}
    
    executor = ActionExecutor(mock_runtime)
    
    state = State(
        task_goal=TaskType.SOLVE,
        initial_inputs=[],
        current_expression="x+1"
    )
    
    action = Action(
        type=ActionType.APPLY_RULE,
        name="add_to_both_sides",
        inputs={"target": "x+1", "next_state": "x+2"} # Dummy transition
    )
    
    result = executor.execute(action, state)
    
    assert result["valid"] is True
    assert state.current_expression == "x+2"
    mock_runtime.check_step.assert_called_once()
    
def test_executor_final_action():
    mock_runtime = MagicMock()
    executor = ActionExecutor(mock_runtime)
    
    state = State(
        task_goal=TaskType.SOLVE,
        initial_inputs=[],
        current_expression="x=5"
    )
    
    action = Action(
        type=ActionType.FINAL,
        name="finish",
        inputs={"answer": "x=5"}
    )
    
    result = executor.execute(action, state)
    
    assert result["status"] == "success"
    assert state.status == "SOLVED"
