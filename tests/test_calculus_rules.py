import pytest
from core.symbolic_engine import SymbolicEngine
from core.knowledge_registry import KnowledgeRegistry
from pathlib import Path

@pytest.fixture
def registry():
    engine = SymbolicEngine()
    # Assuming tests are run from project root
    knowledge_path = Path("core/knowledge")
    return KnowledgeRegistry(knowledge_path, engine)

def test_strict_matching_integral_vs_add(registry):
    """Verify that an Integral does not match an Add rule."""
    # ALG-OP-002 is x + x -> 2x (Add pattern)
    # Expression: Integral(x, x)
    
    expr = "Integral(x, x)"
    # Try to match against a known Add rule
    # We can manually check _match_node or use match()
    
    # Let's find the Add rule node first
    add_rule = registry.rules_by_id.get("ALG-OP-002")
    if not add_rule:
        pytest.skip("ALG-OP-002 not found")
        
    # Manually check strict matching logic via _match_node
    match = registry._match_node(add_rule, expr, "2*Integral(x, x)", False)
    assert match is None, "Integral should not match Add rule"

def test_integration_power_rule(registry):
    """Test Power Rule: Integral(x^2, x) -> x^3/3"""
    before = "Integral(x**2, x)"
    # Note: SymPy might format output differently, e.g. x**3/3
    after = "x**3 / 3" 
    
    match = registry.match(before, after, context_domains=["calculus"])
    assert match is not None
    assert match.id == "CALC-INT-POW"

def test_definite_integral_def(registry):
    """Test Definite Integral: Integral(x, (x, 0, 2)) -> ..."""
    before = "Integral(x, (x, 0, 2))"
    # The rule CALC-INT-DEF transforms to Subs(...) - Subs(...)
    # But usually the user writes the result of that.
    # Wait, the rule matches (before -> after).
    # If the user step is:
    # step: Integral(x, (x, 0, 2))
    # step: [x^2/2]_0^2  <-- This is Subs notation
    
    # Let's check if we can match the transformation to Subs
    # Pattern: Subs(Integral(f, x), x, b) - Subs(Integral(f, x), x, a)
    # For f=x, Integral(x, x) = x^2/2
    # So expected after: Subs(x**2/2, x, 2) - Subs(x**2/2, x, 0)
    
    # However, CALC-INT-DEF is a definition rule.
    # It maps Integral(f, (x, a, b)) -> Subs(Integral(f, x), x, b) - ...
    # It doesn't evaluate the inner integral.
    
    expected_after = "Subs(Integral(x, x), x, 2) - Subs(Integral(x, x), x, 0)"
    
    match = registry.match(before, expected_after, context_domains=["calculus"])
    assert match is not None
    assert match.id == "CALC-INT-DEF"

def test_constant_multiple(registry):
    """Test Constant Multiple: Integral(3*x^2, x) -> 3*Integral(x^2, x)"""
    before = "Integral(3*x**2, x)"
    after = "3 * Integral(x**2, x)"
    
    match = registry.match(before, after, context_domains=["calculus"])
    assert match is not None
    assert match.id == "CALC-INT-CONST"
