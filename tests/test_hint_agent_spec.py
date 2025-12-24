
import pytest
from coherent.core.hint_engine import HintEngine, DriftType
from coherent.core.computation_engine import ComputationEngine
from coherent.core.symbolic_engine import SymbolicEngine

class MockComputationEngine:
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()

def test_drift_analysis_equiv():
    comp_engine = MockComputationEngine()
    engine = HintEngine(comp_engine)
    
    # x + x vs 2x
    # Depending on SymbolicEngine simplifiction, might be Equiv or No Drift if it auto-simplifies
    res = engine.generate_hint("x + x", "2*x")
    # If sym engine auto-simplifies "x+x" to "2x", then it sees NO_DRIFT or EQUIV depending on string compare
    # Let's check drift type in details
    assert res.details["drift"] in [DriftType.REPRESENTATION_EQUIV.value, DriftType.NO_DRIFT.value]

def test_drift_analysis_sign_error():
    comp_engine = MockComputationEngine()
    engine = HintEngine(comp_engine)
    
    # x vs -x
    res = engine.generate_hint("x", "-x")
    assert res.details["drift"] == DriftType.SIGN_ERROR.value
    assert "sign" in res.message.lower()

def test_drift_analysis_constant():
    comp_engine = MockComputationEngine()
    engine = HintEngine(comp_engine)
    
    # x + 5 vs x + 3
    # Diff is 2 (constant)
    res = engine.generate_hint("x + 5", "x + 3")
    assert res.details["drift"] == DriftType.CONSTANT_OFFSET.value
    assert "constant" in res.message.lower()

def test_telemetry_logging():
    comp_engine = MockComputationEngine()
    engine = HintEngine(comp_engine)
    
    engine.generate_hint("x", "y")
    assert len(engine.history) == 1
    event = engine.history[0]
    assert event.user_expr == "x"
    assert event.target_expr == "y"
    assert event.action_selected is not None
