import pytest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def runtime():
    symbolic_engine = SymbolicEngine()
    computation_engine = ComputationEngine(symbolic_engine)
    validation_engine = ValidationEngine(computation_engine)
    hint_engine = HintEngine(computation_engine)
    
    return CoreRuntime(
        computation_engine=computation_engine,
        validation_engine=validation_engine,
        hint_engine=hint_engine
    )

def test_step_hint_sign_error(runtime):
    runtime.set("x + 1")
    
    # User enters -(x + 1) which is -x - 1
    # This is a sign error relative to the previous step "x + 1" IF the user intended to keep it equivalent
    # Wait, check_step checks if current step is EQUIVALENT to previous step.
    # If I enter "-x - 1", it is NOT equivalent to "x + 1".
    # The hint engine will compare "-x - 1" (user) vs "x + 1" (target/previous).
    
    result = runtime.check_step("-x - 1")
    
    assert result["valid"] is False
    assert "hint" in result["details"]
    assert result["details"]["hint"]["type"] == "heuristic_sign_error"

def test_step_hint_constant_offset(runtime):
    runtime.set("x + 5")
    
    # User enters "x + 6"
    # Difference is 1
    
    result = runtime.check_step("x + 6")
    
    assert result["valid"] is False
    assert "hint" in result["details"]
    assert result["details"]["hint"]["type"] == "heuristic_constant_offset"
