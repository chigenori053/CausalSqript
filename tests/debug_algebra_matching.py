import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from coherent.engine.symbolic_engine import SymbolicEngine

def debug_matching():
    print("--- Debugging Match Structure ---")
    engine = SymbolicEngine()
    
    expr = "7*a - 2*a + 4"
    pattern = "a * x + b * x"
    
    print(f"Expr: {expr}")
    print(f"Pattern: {pattern}")
    
    result = engine.match_structure(expr, pattern)
    
    if result:
        print("Match SUCCESS")
        for k, v in result.items():
            print(f"  {k}: {v} (Type: {type(v)})")
    else:
        print("Match FAILED (None)")

    # Also test strict 2-term match
    expr_2 = "7*a - 2*a"
    print(f"\nExpr (2 terms): {expr_2}")
    result_2 = engine.match_structure(expr_2, pattern)
    if result_2:
        print("Match SUCCESS")
        for k, v in result_2.items():
             print(f"  {k}: {v}")
    else:
        print("Match FAILED")

if __name__ == "__main__":
    debug_matching()
