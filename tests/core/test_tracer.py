import pytest
import time
from coherent.core.tracer import Tracer
from coherent.core.state import State
from coherent.core.action import Action
from coherent.core.action_types import ActionType
from coherent.engine.language.semantic_types import TaskType

def test_tracer_lifecycle():
    tracer = Tracer()
    
    # 1. Start Episode
    ep_id = tracer.start_episode("Solve x+1=2")
    assert ep_id is not None
    assert tracer._current_episode.problem_text == "Solve x+1=2"
    
    # 2. Log Step
    state = State(
        task_goal=TaskType.SOLVE,
        initial_inputs=[],
        current_expression="x+1=2"
    )
    
    action = Action(
        type=ActionType.APPLY_RULE,
        name="subtract_one",
        inputs={"target": "x+1=2"}
    )
    
    result = {"valid": True, "status": "correct"}
    
    tracer.log_step(state, action, result)
    
    current_ep = tracer._current_episode
    assert len(current_ep.steps) == 1
    assert current_ep.steps[0].state_snapshot["expression"] == "x+1=2"
    assert current_ep.steps[0].action["name"] == "subtract_one"
    
    # 3. End Episode
    tracer.end_episode("SUCCESS")
    assert tracer._current_episode is None
    assert len(tracer._episodes) == 1
    assert tracer._episodes[0].final_outcome == "SUCCESS"

def test_export_history():
    tracer = Tracer()
    tracer.start_episode("Test")
    tracer.end_episode("FAILURE")
    
    history = tracer.export_history()
    assert len(history) == 1
    assert history[0]["problem_text"] == "Test"
    assert history[0]["final_outcome"] == "FAILURE"
    assert "steps" in history[0]
