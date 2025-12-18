
from coherent.engine.symbolic_engine import SymbolicEngine
from sympy import Wild, parse_expr
import sympy

def debug_matching():
    engine = SymbolicEngine()
    
    # confusing input from test
    expr_str = "(1 + 1*i) * (1 - 1*i)"
    
    # Pattern from YAML
    pattern_str = "(a + b*i) * (c + d*i)"
    
    print(f"Expr string: {expr_str}")
    print(f"Pattern string: {pattern_str}")
    
    # Direct check using engine.match_structure logic (simplified)
    # We use engine.match_structure to see what happens
    
    bindings = engine.match_structure(expr_str, pattern_str)
    print(f"Bindings found: {bindings}")
    
    # Let's inspect internal representations
    local_dict = {"i": sympy.Symbol("i")}
    expr_internal = parse_expr(expr_str, evaluate=False, local_dict=local_dict)
    pattern_internal = parse_expr(pattern_str, evaluate=False, local_dict=local_dict)
    
    print(f"Internal Expr: {expr_internal} (Type: {type(expr_internal)})")
    print(f"Internal Expr Args: {expr_internal.args}")
    
    # Check the terms inside
    term1 = expr_internal.args[0] # (1 + 1*i)
    print(f"Term 1: {term1} (Type: {type(term1)})")
    print(f"Term 1 Args: {term1.args}") 
    
    # See if 1*i became i
    imag_part = term1.args[1] # expected 1*i
    print(f"Imag part of Term 1: {imag_part} (Type: {type(imag_part)})")
    
    print(f"Internal Pattern: {pattern_internal}")

    # Try matching manually with Wilds
    a = Wild("a")
    b = Wild("b")
    c = Wild("c")
    d = Wild("d")
    i = sympy.Symbol("i")
    
    pat_manual = (a + b*i) * (c + d*i)
    print(f"Manual Pattern: {pat_manual}")
    
    match_res = expr_internal.match(pat_manual)
    print(f"Manual Match Result: {match_res}")

    # Test with non-unity coefficients to see if it works
    expr_str_2 = "(1 + 2*i) * (3 + 4*i)"
    print(f"\nChecking expr: {expr_str_2}")
    bindings_2 = engine.match_structure(expr_str_2, pattern_str)
    print(f"Bindings 2: {bindings_2}")

if __name__ == "__main__":
    debug_matching()
