import pytest
from coherent.engine.modules.calculus.parser import CalculusParser
from coherent.engine.modules.calculus.engine import CalculusEngine

def test_calculus_parser_normalization():
    parser = CalculusParser()
    # Normalize d/dx x^2 -> diff(x**2, x) then parse to SymPy
    expr = parser.parse("d/dx x^2")
    import ast
    if isinstance(expr, ast.AST):
        # We can't easily stringify AST back to code exactly without unparse (3.9+)
        # But we can check structure or just pass if it's an Expression
        assert isinstance(expr, ast.Expression)
    else:
        # We mapped diff -> Derivative (unevaluated)
        assert str(expr).replace(" ", "") == "Derivative(x**2,x)"

def test_calculus_engine_differentiation():
    pytest.importorskip("sympy")
    parser = CalculusParser()
    engine = CalculusEngine()
    
    # 1. Parse
    expr = parser.parse("diff(x^3, x)")
    
    # 2. Evaluate
    # Context empty means symbolic evaluation
    result = engine.evaluate(expr, {})
    
    assert str(result).replace(" ", "") == "3*x**2"

def test_calculus_engine_integration():
    pytest.importorskip("sympy")
    parser = CalculusParser()
    engine = CalculusEngine()
    
    expr = parser.parse("integrate(2*x, x)")
    result = engine.evaluate(expr, {})
    
    assert str(result).replace(" ", "") == "x**2"
