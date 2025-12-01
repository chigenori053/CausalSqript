from sympy import symbols, latex
from sympy.parsing.sympy_parser import parse_expr

def test_rendering():
    x, y = symbols('x y')
    
    exprs = [
        ("2 * 3", "Arithmetic"),
        ("2 * x", "Algebra Mixed"),
        ("x * y", "Algebra Pure"),
        ("2.5 * x", "Float Mixed"),
        ("1/2 * x", "Fraction Mixed")
    ]
    
    modes = [
        ("dot", r" \cdot "),
        ("times", r" \times "),
        ("implicit", "")
    ]
    
    print(f"{'Expression':<15} | {'Mode':<10} | {'LaTeX':<20}")
    print("-" * 50)
    
    for expr_str, cat in exprs:
        # Use evaluate=False to preserve structure
        expr = parse_expr(expr_str, evaluate=False)
        for mode_name, sym in modes:
            l = latex(expr, mul_symbol=sym)
            print(f"{expr_str:<15} | {mode_name:<10} | {l:<20}")

if __name__ == "__main__":
    test_rendering()
