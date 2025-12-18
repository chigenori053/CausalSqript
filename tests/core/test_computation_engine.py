import pytest
from fractions import Fraction

from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.errors import InvalidExprError

@pytest.fixture
def engine():
    symbolic = SymbolicEngine()
    return ComputationEngine(symbolic)

def test_simplify(engine):
    assert engine.simplify("2 + 2") == "4"
    assert engine.simplify("x + x") == "2*x"

def test_numeric_eval(engine):
    assert engine.numeric_eval("3 * 4") == 12
    
    engine.bind("x", 10)
    assert engine.numeric_eval("x + 5") == 15
    
    assert engine.numeric_eval("y * 2", context={"y": 3}) == 6
    assert engine.numeric_eval("1/2 + 1/3") == Fraction(5, 6)

def test_bind_variable(engine):
    engine.bind("a", 100)
    assert engine.variables["a"] == 100
    assert engine.numeric_eval("a") == 100

def test_to_sympy(engine):
    # This depends on whether sympy is installed or fallback is used
    # If fallback, it returns an AST object
    # If sympy, it returns a sympy object
    result = engine.to_sympy("x + 1")
    assert result is not None

def test_expand(engine):
    """Test expansion of algebraic expressions."""
    # Expand (x + y)**2
    result = engine.expand("(x + y)**2")
    assert "x**2" in result
    assert "y**2" in result
    assert "x*y" in result
    
    # Expand (a + b)*(c + d)
    result = engine.expand("(a + b)*(c + d)")
    assert "a*c" in result
    assert "a*d" in result
    assert "b*c" in result
    assert "b*d" in result
    
    # Expand (x - 1)**3
    result = engine.expand("(x - 1)**3")
    assert "x**3" in result

def test_factor(engine):
    """Test factoring of algebraic expressions."""
    # Factor x**2 - y**2 (difference of squares)
    result = engine.factor("x**2 - y**2")
    assert "x - y" in result
    assert "x + y" in result
    
    # Factor x**2 + 2*x + 1 (perfect square)
    result = engine.factor("x**2 + 2*x + 1")
    assert "(x + 1)**2" in result or "(1 + x)**2" in result
    
    # Factor x**2 - 1
    result = engine.factor("x**2 - 1")
    assert "x - 1" in result
    assert "x + 1" in result

def test_substitute(engine):
    """Test variable substitution in expressions."""
    # Substitute both variables
    result = engine.substitute("x + y", {"x": 2, "y": 3})
    assert result == "5"
    
    # Substitute one variable
    result = engine.substitute("a*x + b", {"x": 5})
    assert "5*a" in result or "a*5" in result
    assert "b" in result
    
    # Substitute in a more complex expression
    result = engine.substitute("x**2 + 2*x + 1", {"x": 3})
    assert result == "16"  # 9 + 6 + 1 = 16

def test_expand_factor_roundtrip(engine):
    """Test that expand and factor are inverse operations."""
    # Start with factored form
    factored = "(x - y)*(x + y)"
    
    # Expand it
    expanded = engine.expand(factored)
    assert "x**2" in expanded
    assert "y**2" in expanded
    
    # Factor it back
    refactored = engine.factor(expanded)
    # Should contain the factors (order may vary)
    assert "x - y" in refactored or "x + y" in refactored

def test_simplify_with_variables(engine):
    """Test simplification of expressions with variables."""
    assert engine.simplify("x + x + x") == "3*x"
    assert engine.simplify("2*x + 3*x") == "5*x"
    assert engine.simplify("x*x") == "x**2"

def test_numeric_eval_with_context_override(engine):
    """Test that context parameter overrides bound variables."""
    engine.bind("x", 10)
    
    # Use bound value
    assert engine.numeric_eval("x + 5") == 15
    
    # Override with context
    assert engine.numeric_eval("x + 5", context={"x": 20}) == 25
    
    # Bound value should still be 10
    assert engine.variables["x"] == 10

def test_substitute_partial(engine):
    """Test partial substitution leaves other variables intact."""
    result = engine.substitute("x + y + z", {"x": 1, "y": 2})
    # Should have 1 + 2 + z = 3 + z
    assert "z" in result
    assert "3" in result or "1" in result  # Depends on simplification

def test_expand_already_expanded(engine):
    """Test expanding an already expanded expression."""
    result = engine.expand("x**2 + 2*x*y + y**2")
    # Should remain the same or equivalent
    assert "x**2" in result
    assert "y**2" in result

def test_factor_prime_expression(engine):
    """Test factoring an expression that cannot be factored further."""
    result = engine.factor("x + y")
    # Should remain the same
    assert "x" in result
    assert "y" in result
