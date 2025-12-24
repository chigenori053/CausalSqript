
import pytest
from coherent.tools.library.complex_domain import ComplexDomain
from sympy import I, Symbol, simplify

def test_contains():
    dom = ComplexDomain()
    assert dom.contains("1 + 2i")
    assert dom.contains("3")
    assert dom.contains(3 + 4j) # Python complex
    assert not dom.contains("x + y") # Symbolic unknown
    
def test_contains_sympy():
    dom = ComplexDomain()
    assert dom.contains(I)
    assert dom.contains(1 + 2*I)

def test_distance():
    dom = ComplexDomain()
    z1 = "1 + 1i"
    z2 = "4 + 5i"
    # dist = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = 5
    d = dom.distance(z1, z2)
    assert abs(d - 5.0) < 1e-9

def test_canonicalize():
    dom = ComplexDomain()
    expr = "(1 + i)^2"
    # (1+i)^2 = 1 + 2i - 1 = 2i
    canon = dom.canonicalize(expr)
    # The output string should use 'i' instead of 'I'
    assert "2*i" in canon or "2i" in canon
    assert "I" not in canon

def test_polar():
    dom = ComplexDomain()
    expr = "1 + i"
    # r = sqrt(2), theta = pi/4
    # sqrt(2)*exp(i*pi/4)
    polar = dom.to_polar(expr)
    assert "exp" in polar
    assert "sqrt(2)" in polar or "1.414" in polar
    assert "pi/4" in polar

def test_euler_identity_canonical():
    dom = ComplexDomain()
    # e^(i*pi) + 1 = 0
    expr = "exp(i*pi) + 1"
    canon = dom.canonicalize(expr)
    assert canon == "0"

def test_string_inputs():
    dom = ComplexDomain()
    # Valid math strings
    assert dom.contains("3 + 4i")
    assert dom.contains("5")
    
    # Invalid strings (should not crash, should return False)
    assert not dom.contains("hello world")
    assert not dom.contains("just text")
    
    # Weird but technically symbolic? 
    # sympify("x") is not a complex NUMBER, it's a symbol.
    # contains() returns False for symbols.
    assert not dom.contains("x")

def test_operator_semantics():
    dom = ComplexDomain()
    # Test Multiplication (1+i)(1-i) = 1 - i^2 = 2
    res = dom.canonicalize("(1+i)*(1-i)")
    assert res == "2"
    
    # Test Addition
    res = dom.canonicalize("(2+3i) + (1-i)")
    # Should be 3 + 2i
    # Note: string format might vary "3 + 2*i" or "2*i + 3"
    assert "3" in res and "2*i" in res

def test_math_symbols():
    dom = ComplexDomain()
    # Test Unicode Pi and Sqrt
    # π = pi, √ = sqrt
    # e^(i*π)
    expr = "exp(i*π)"
    canon = dom.canonicalize(expr)
    # e^i*pi = -1
    assert canon == "-1"

    # Sqrt(-1) -> i
    # Note: input_parser maps √ to SQRT then sqrt
    expr2 = "√(-1)" 
    # This invokes sqrt(-1) -> i
    # We need to ensure input parser normalization happens inside parse_with_i or we call it explicitly.
    # ComplexDomain.canonicalize calls _ensure_sympy which calls parse_with_i which calls CausalScriptInputParser.normalize.
    # So this should work.
    canon2 = dom.canonicalize(expr2)
    assert canon2 == "i" or canon2 == "I"
