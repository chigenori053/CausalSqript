from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.evaluator import SymbolicEvaluationEngine
from pathlib import Path
import pytest

def test_pending_bounds():
    engine = SymbolicEngine()
    if not engine.has_sympy():
        print("Skipping test: SymPy not available")
        return

    registry = KnowledgeRegistry(Path("core/knowledge"), engine)
    eval_engine = SymbolicEvaluationEngine(engine, registry)
    
    # Problem: Definite Integral
    problem = "Integral(3*x**2, (x, 0, 2))"
    eval_engine.set(problem)
    print(f"Problem set: {problem}")
    
    # Step 1: Indefinite Integral (Pending Bounds)
    # x^3 is the antiderivative of 3x^2
    step1 = "x**3"
    result1 = eval_engine.check_step(step1)
    
    print(f"Step 1 ({step1}): Valid={result1['valid']}")
    print(f"Details: {result1.get('details')}")
    
    assert result1["valid"] is True
    assert result1.get("status") == "partial"
    assert result1["details"]["reason"] == "pending_bounds"
    
    # Verify state did NOT update (still the integral)
    # Actually check_step doesn't expose _current_expr directly, but we can infer it
    # by checking the next step against the original problem.
    
    # Step 2: Final Result
    step2 = "8"
    result2 = eval_engine.check_step(step2)
    print(f"Step 2 ({step2}): Valid={result2['valid']}")
    
    assert result2["valid"] is True
    # This matches because Integral(3x^2, 0, 2) == 8

if __name__ == "__main__":
    try:
        test_pending_bounds()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
