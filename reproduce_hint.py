
import sys
import os
sys.path.append(os.getcwd())

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.hint_engine import HintEngine

def test_hint():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    hint_engine = HintEngine(comp_engine)

    user_expr = "x^2 + y^2"
    target_expr = "(x - y)^2"
    
    print(f"User: {user_expr}")
    print(f"Target: {target_expr}")
    
    # Test simplification manually
    diff_expr = f"({user_expr}) - ({target_expr})"
    try:
        simplified = sym_engine.simplify(diff_expr)
        print(f"Simplified Diff: {simplified}")
    except Exception as e:
        print(f"Simplification Error: {e}")

    # Test generate_hint
    hint = hint_engine.generate_hint(user_expr, target_expr)
    print(f"Hint Type: {hint.hint_type}")
    print(f"Hint Message: {hint.message}")

if __name__ == "__main__":
    test_hint()
