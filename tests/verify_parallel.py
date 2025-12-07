import time
import math
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.computation_engine import ComputationEngine

def verify_parallel():
    print("Initializing engines...")
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)

    # Create a scenario that requires some computation
    # We'll use a slightly complex expression and many scenarios
    expr = "x**2 + y**2"
    scenarios = {f"scenario_{i}": {"x": i, "y": i+1} for i in range(100)}

    print(f"Running evaluation for {len(scenarios)} scenarios...")
    start_time = time.time()
    results = comp_engine.evaluate_in_scenarios(expr, scenarios)
    end_time = time.time()

    print(f"Evaluation took {end_time - start_time:.4f} seconds")
    
    # Verify correctness for a few samples
    for i in [0, 10, 50, 99]:
        name = f"scenario_{i}"
        expected = i**2 + (i+1)**2
        actual = results[name]
        print(f"{name}: Expected {expected}, Got {actual}")
        assert actual == expected, f"Mismatch for {name}"

    print("Verification successful!")

if __name__ == "__main__":
    verify_parallel()
