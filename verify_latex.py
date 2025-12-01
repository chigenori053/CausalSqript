import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.symbolic_engine import SymbolicEngine

def test_latex_evaluation():
    engine = SymbolicEngine()
    expr = "2**3 - 0"
    latex = engine.to_latex(expr)
    print(f"Expr: {expr}")
    print(f"LaTeX: {latex}")
    
    if latex == "8":
        print("FAIL: to_latex evaluated the expression!")
    else:
        print("PASS: to_latex preserved structure.")

if __name__ == "__main__":
    test_latex_evaluation()
