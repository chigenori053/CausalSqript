
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalscript.core.symbolic_engine import SymbolicEngine

def debug_match():
    engine = SymbolicEngine()
    
    def match(expr_str, pattern_str):
        print(f"\nMatching '{expr_str}' against '{pattern_str}'")
        matches = engine.match_structure(expr_str, pattern_str)
        if matches:
            print("  Matches:", matches)
        else:
            print("  No match")

    # Case 1: Power Definition
    match("(x - y)**2", "a**2")
    match("(x - y) * (x - y)", "a * a")
    match("(x - y) * (x - y)", "a**2") # Check if it matches Pow pattern
    
    # Case 2: Distributive
    match("x * (x + 1)", "a * (b + c)")
    match("x**2 + x", "a * b + a * c")
    
    # Case 4: Combine Like Terms
    match("x**2 - x*y + x*y", "a * x + b * x") 
    match("x**2", "(a + b) * x")

    # Case 5: Mul vs Add (False Positive Check)
    match("(x - y) * (x - y)", "a**2 - 2*a*b + b**2")

    # Case 6: Difference of Squares (Factoring)
    match("(x + 3) * (x - 3)", "(a + b) * (a - b)")

if __name__ == "__main__":
    debug_match()
