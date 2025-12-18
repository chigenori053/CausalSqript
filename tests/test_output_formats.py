import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.geometry_engine import GeometryEngine
from coherent.engine.function_analysis import FunctionAnalyzer

def test_to_latex():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)

    # Test simple expression
    assert comp_engine.to_latex("x + y") == "x + y"
    
    # Test power
    assert comp_engine.to_latex("x^2") == "x^{2}"
    
    # Test fraction (if SymPy available)
    if sym_engine.has_sympy():
        assert comp_engine.to_latex("1/2") == r"\frac{1}{2}"
        assert comp_engine.to_latex("pi") == r"\pi"
        assert comp_engine.to_latex("sqrt(x)") == r"\sqrt{x}"

def test_get_shape_data():
    try:
        geo_engine = GeometryEngine()
    except ImportError:
        pytest.skip("SymPy not available")

    # Point
    p1 = geo_engine.point(0, 0)
    data = geo_engine.get_shape_data(p1)
    assert data["type"] == "point"
    assert data["x"] == 0.0
    assert data["y"] == 0.0

    # Circle
    c1 = geo_engine.circle(p1, 5)
    data = geo_engine.get_shape_data(c1)
    assert data["type"] == "circle"
    assert data["center"] == [0.0, 0.0]
    assert data["radius"] == 5.0

    # Segment
    p2 = geo_engine.point(3, 4)
    s1 = geo_engine.segment(p1, p2)
    data = geo_engine.get_shape_data(s1)
    assert data["type"] == "segment"
    assert data["p1"] == [0.0, 0.0]
    assert data["p2"] == [3.0, 4.0]
    assert data["length"] == 5.0

def test_function_plot_data():
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    analyzer = FunctionAnalyzer(comp_engine)

    data = analyzer.generate_plot_data("x^2", start=-10, end=10, num_points=21)
    assert "x" in data
    assert "y" in data
    assert len(data["x"]) == 21
    assert len(data["y"]) == 21
    assert data["x"][0] == -10.0
    assert data["y"][0] == 100.0
    assert data["x"][10] == 0.0
    assert data["y"][10] == 0.0
