import pytest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine

@pytest.fixture
def runtime_setup():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validation = ValidationEngine(computation)
    hint = HintEngine(computation)
    return CoreRuntime(computation, validation, hint)

def test_rendering_algebra(runtime_setup):
    runtime = runtime_setup
    runtime.set("x + x")
    result = runtime.check_step("2*x")
    
    # Check if rendering details exist
    assert "rendered" in result["details"]
    rendered = result["details"]["rendered"]
    assert "before" in rendered
    assert "after" in rendered
    
    # Algebra uses standard formatting (or simple string if SymPy not available/mocked)
    # Just ensure it's present and not empty
    assert rendered["before"] is not None
    assert rendered["after"] is not None

def test_rendering_calculus(runtime_setup):
    runtime = runtime_setup
    # Force category detection or manual set if possible, 
    # but check_step detects automatically.
    # We use a calculus-like expression
    # Function to check if sympy is installed
    try:
        import sympy
        # Use Derivative to prevent immediate evaluation to 2*x
        runtime.set("Derivative(x^2, x)")
    except ImportError:
        # Fallback if no sympy (though CoreRuntime likely depends on it)
        runtime.set("diff(x^2, x)")
    
    # The runtime detects category 'calculus' for 'diff' or 'Derivative'
    result = runtime.check_step("2*x")
    
    assert "rendered" in result["details"]
    rendered_before = result["details"]["rendered"]["before"]
    
    # Expect LaTeX or fallback formatting
    # If SymPy is available, Derivative -> \frac{d}{dx} ...
    print(f"DEBUG: Rendered Before: {rendered_before}")

    # We accept either standard LaTeX or our fallback
    assert "\\frac{d}{dx}" in rendered_before or "\\frac{d}{d x}" in rendered_before or "d/dx" in rendered_before or "\\partial" in rendered_before

def test_rendering_statistics(runtime_setup):
    # This might require forcing category if detection isn't smart enough for simple lists yet,
    # but let's try a distribution call if supported, or just use category update hack for testing.
    runtime = runtime_setup
    runtime.set("mean(data)")
    
    # Force category for test purposes if needed, but let's see if we can trigger via valid step
    # or just mocking the category since we really want to test the *renderer*, not detection
    runtime._current_category = runtime.category_identifier.identify("mean([1,2,3])") # statistics
    
    # Create a dummy result to test the engine directly if runtime integration is hard to trigger exactly
    from coherent.engine.renderers import RenderingEngine
    from coherent.engine.math_category import MathCategory
    
    engine = RenderingEngine()
    result = {
        "before": "mean(x)",
        "after": "mu",
        "details": {"category": "statistics"}
    }
    engine.render_result(result)
    
    assert "rendered" in result["details"]
    assert "📊" in result["details"]["rendered"]["before"]

def test_rendering_finalize(runtime_setup):
    runtime = runtime_setup
    runtime.set("x + 1")
    result = runtime.finalize("x + 1")
    
    assert "rendered" in result["details"]
    assert result["details"]["rendered"]["after"] is not None
