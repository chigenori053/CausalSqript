
import pytest
from coherent.core.core_runtime import CoreRuntime
from coherent.core.computation_engine import ComputationEngine
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine

class MockValidationEngine(ValidationEngine):
    def __init__(self):
        pass
class MockHintEngine(HintEngine):
    def __init__(self):
        pass

def test_core_loads_complex_domain():
    # Setup minimal core runtime
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = MockValidationEngine()
    hint_engine = MockHintEngine()
    
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)
    
    # Check if complex domain is loaded lazily
    assert "complex" not in runtime._extensions
    
    cd = runtime.complex_domain
    assert cd is not None
    assert "complex" in runtime._extensions
    
    # Verify functionality via runtime
    assert cd.contains("1 + i")
    assert cd.canonicalize("(1+i)^2") == "2*i"

def test_core_complex_distance():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    val_engine = MockValidationEngine()
    hint_engine = MockHintEngine()
    runtime = CoreRuntime(comp_engine, val_engine, hint_engine)

    # Use complex domain logic
    d = runtime.complex_domain.distance("0", "3+4i")
    assert abs(d - 5.0) < 1e-9
