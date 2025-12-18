
import pytest
pytest.importorskip("sympy")
from coherent.logic import CausalEngine
from coherent.logic.causal_types import CausalNodeType

def records_freshman_dream():
    # Scenario: User incorrectly expands (x+y)^2 -> x^2 + y^2
    # Then tries to do something else and gets an error.
    return [
        {"phase": "problem", "expression": "(x + y)^2", "rendered": "problem", "status": "ok"},
        {
            "phase": "step",
            "expression": "x^2 + y^2", # Freshman's Dream!
            "rendered": "step 1",
            "status": "ok", # System might have missed it or Fuzzy accepted it
            "step_id": "step-1"
        },
        {
            "phase": "step",
            "expression": "x^2",
            "rendered": "step 2",
            "status": "mistake",
            "step_id": "step-2",
            "meta": {"reason": "something_wrong"}
        },
        {"phase": "error", "status": "mistake", "expression": "x^2"} # Generated error node
    ]

def test_suggest_fix_prioritizes_misuse():
    engine = CausalEngine()
    records = records_freshman_dream()
    engine.ingest_log(records)
    
    # Identify the error node (last added error)
    error_node_id = engine.to_dict()["errors"][-1]
    
    # Get suggestions
    candidates = engine.suggest_fix_candidates(error_node_id)
    
    # We expect step-1 to be highly ranked because of Freshman's Dream detection,
    # even though step-2 is closer (depth 1) and 'mistake'.
    # Step-1 is depth 2.
    
    # Let's inspect the returned order
    ids = [n.node_id for n in candidates]
    
    # Check if step-1 is the first or close to top
    # Since step-2 is "mistake" and depth 1, it has high base prob (0.8).
    # Step-1 is depth 2 (0.4 base), but boosted to 0.95.
    # So 0.95 > 0.8. Step-1 should be FIRST.
    
    # Note: CausalEngine generates node IDs like 'step-1' based on counter.
    # Ingest log creates nodes.
    # We should verify node IDs.
    
    assert len(candidates) >= 2
    
    # The first candidate should correspond to the Freshman's Dream step
    first_candidate = candidates[0]
    
    # Verify it is the node with expression 'x^2 + y^2'
    expr = first_candidate.payload["record"]["expression"]
    assert expr == "x^2 + y^2"
    
    # Verify the second candidate is likely step-2
    second_candidate = candidates[1]
    assert second_candidate.payload["record"]["expression"] == "x^2"

