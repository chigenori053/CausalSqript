import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from coherent.engine.symbolic_engine import SymbolicEngine

def debug_subs():
    print("--- Debugging Substitution ---")
    engine = SymbolicEngine()
    
    # Simulate ARITH-ADD-002: (a+b)+c -> a+(b+c)
    pattern_after = "a + (b + c)"
    
    # Bindings from previous debug output
    # 'c': 5*a, 'b': 2*a, 'a': -1*3*b + b
    # Note: -1*3*b + b is mathematically -2b.
    # We use string representations as they come from bindings.
    bindings = {
        'c': "5*a",
        'b': "2*a",
        'a': "-3*b + b" 
    }
    
    print(f"Pattern: {pattern_after}")
    print(f"Bindings: {bindings}")
    
    result = engine.substitute(pattern_after, bindings)
    print(f"Result (Subs): {result}")
    
    # Try xreplace
    from sympy.parsing.sympy_parser import parse_expr 
    # Use fallback-safe approach or just strings for local dict logic if needed
    # But here we just want to test xreplace logic parity with subs
    local_dict = {"e": 2.718, "pi": 3.14159} # Dummy or real syms
    internal = parse_expr(pattern_after, evaluate=False, local_dict=local_dict)
    
    subs = {}
    for k, v in bindings.items():
        val_internal = parse_expr(str(v), evaluate=False, local_dict=local_dict)
        # Use Symbol directly for xreplace
        import sympy
        subs[sympy.Symbol(k)] = val_internal
        
    res_xreplace = internal.xreplace(subs)
    print(f"Result (xreplace): {res_xreplace}")

    # Check simplified result
    simplified = engine.simplify(result)
    print(f"Result (Simplified): {simplified}")

if __name__ == "__main__":
    debug_subs()
