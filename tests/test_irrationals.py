"""Tests for Irrational Numbers and Square Roots."""

import pytest
try:
    import sympy
except ImportError:
    sympy = None

from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.input_parser import CoherentInputParser

@pytest.fixture
def engine():
    if sympy is None:
        pytest.skip("SymPy not installed")
    return SymbolicEngine()

def test_parser_constants():
    # Test pi
    assert CoherentInputParser.normalize("2pi").replace(" ", "") == "2*pi"
    assert CoherentInputParser.normalize("2π").replace(" ", "") == "2*pi"
    
    # Test e
    assert CoherentInputParser.normalize("e^x").replace(" ", "") == "e**x"
    
    # Test sqrt
    assert CoherentInputParser.normalize("√2").replace(" ", "") == "sqrt(2)"
    assert CoherentInputParser.normalize("sqrt(2)").replace(" ", "") == "sqrt(2)"

def test_symbolic_constants(engine):
    # Test pi evaluation
    expr = engine.to_internal("pi")
    assert expr == sympy.pi
    
    # Test e evaluation
    expr = engine.to_internal("e")
    assert expr == sympy.E
    
    # Test calculations
    # e^(ln(2)) = 2
    assert engine.simplify("e**(ln(2))") == "2"
    
    # sin(pi/2) = 1
    assert engine.simplify("sin(pi/2)") == "1"
    
    # sqrt(4) = 2
    assert engine.simplify("sqrt(4)") == "2"
    
    # sqrt(2)*sqrt(2) = 2
    assert engine.simplify("sqrt(2)*sqrt(2)") == "2"

def test_numeric_eval(engine):
    # pi approx 3.14
    val = engine.evaluate("pi", {})
    assert abs(val - 3.14159) < 0.001
    
    # e approx 2.718
    val = engine.evaluate("e", {})
    assert abs(val - 2.71828) < 0.001
    
    # sqrt(2) approx 1.414
    val = engine.evaluate("sqrt(2)", {})
    assert abs(val - 1.41421) < 0.001
