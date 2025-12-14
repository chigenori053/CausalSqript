import pytest
from causalscript.core.modules.geometry.parser import GeometryParser
from causalscript.core.modules.geometry.engine import GeometryEngine

def test_geometry_parser():
    # Point(0, 0)
    parser = GeometryParser()
    # Assuming CausalScriptInputParser normalizes or simply passes "Point(0,0)"
    # And SymbolicEngine to_internal converts it to SymPy Point if valid
    expr = parser.parse("Point(0, 0)")
    
    # If SymPy is available, we get a Point2D
    # If not, we get an AST Call
    import ast
    if isinstance(expr, ast.AST):
        assert isinstance(expr, (ast.Call, ast.Expression))
    else:
        # SymPy Point2D
        assert "Point" in str(type(expr))

def test_geometry_engine_distance():
    pytest.importorskip("sympy")
    parser = GeometryParser()
    engine = GeometryEngine()
    
    # Point(0,0).distance(Point(3,4)) -> 5
    expr = parser.parse("Point(0,0).distance(Point(3,4))")
    
    # Using internal symbolic engine evaluate
    result = engine.evaluate(expr, {})
    
    # 5 or 5.0
    assert float(result) == 5.0
