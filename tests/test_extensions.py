import pytest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def runtime():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    # Mocking validation/hint engines as they are not needed for extension tests
    return CoreRuntime(computation, None, None)

def test_extensions_start_empty(runtime):
    """Verify that extensions are not loaded initially."""
    assert len(runtime._extensions) == 0

def test_lazy_loading_calculus(runtime):
    """Verify calculus engine is loaded on demand."""
    calc = runtime.get_extension("calculus")
    assert calc is not None
    assert "calculus" in runtime._extensions
    assert runtime.calculus_engine is calc

def test_extension_caching(runtime):
    """Verify that extensions are cached (singleton behavior per runtime)."""
    calc1 = runtime.get_extension("calculus")
    calc2 = runtime.get_extension("calculus")
    assert calc1 is calc2
    
def test_unknown_extension_error(runtime):
    """Verify error for unknown extensions."""
    with pytest.raises(ValueError):
        runtime.get_extension("non_existent_engine")

def test_property_access_triggers_load(runtime):
    """Verify that accessing properties triggers loading."""
    assert "stats" not in runtime._extensions
    stats = runtime.stats_engine
    assert "stats" in runtime._extensions
    assert stats is not None
