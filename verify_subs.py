import sympy
from sympy import sympify, Subs

def test_subs():
    expr_str = "Subs(x**3, x, 2)"
    try:
        res = sympify(expr_str)
        print(f"Parsed: {res} (Type: {type(res)})")
        evaluated = res.doit()
        print(f"Evaluated: {evaluated}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_subs()
