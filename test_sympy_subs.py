import pytest
sympy = pytest.importorskip("sympy")
from sympy.parsing.sympy_parser import parse_expr

def test():
    expr = "(1 + 2) * (3 + 4)"
    target = "1 + 2"
    replacement = "3"
    
    internal_expr = parse_expr(expr, evaluate=False)
    internal_target = parse_expr(target, evaluate=False)
    internal_replacement = parse_expr(replacement, evaluate=False)
    
    from sympy import UnevaluatedExpr
    
    # Wrap replacement in UnevaluatedExpr
    replacement_node = UnevaluatedExpr(internal_replacement)
    res = internal_expr.subs(internal_target, replacement_node)
    print(f"Result (UnevaluatedExpr): {res} type: {type(res)}")
    
    # Check if 3+4 evaluated
    # internal_expr is Mul(Add(1,2), Add(3,4))
    # res should be Mul(3, Add(3,4)) if not evaluated
    
    expr2 = "sin(pi/3)**2 + cos(pi/3)**2"
    target2 = "sin(pi/3)**2"
    replacement2 = "3/4"
    
    internal_expr2 = parse_expr(expr2, evaluate=False)
    internal_target2 = parse_expr(target2, evaluate=False)
    internal_replacement2 = parse_expr(replacement2, evaluate=False)
    
    res2 = internal_expr2.subs(internal_target2, internal_replacement2)
    print(f"Result2: {res2}")

if __name__ == "__main__":
    test()
