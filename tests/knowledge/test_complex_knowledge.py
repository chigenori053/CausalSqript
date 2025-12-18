
import pytest
from pathlib import Path
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry

@pytest.fixture
def complex_registry():
    engine = SymbolicEngine()
    root_path = Path("coherent/engine/knowledge")
    return KnowledgeRegistry(root_path, engine)

def test_complex_i_definition(complex_registry):
    engine = complex_registry.engine
    # i^2 -> -1
    step_input = "i**2"
    rules = complex_registry.match_rules(step_input)
    matched = [node for node, candidate in rules if node.id == "ALG-CPLX-DEF-I"]
    assert matched, "Should match i^2 = -1 rule"

def test_complex_addition(complex_registry):
    # (1 + 2*i) + (3 + 4*i)
    expr = "(1 + 2*i) + (3 + 4*i)"
    rules = complex_registry.match_rules(expr)
    matched = [node for node, candidate in rules if node.id == "ALG-CPLX-ADD"]
    assert matched, "Should match complex addition rule"

def test_complex_multiplication(complex_registry):
    # Using non-1 coefficients to ensure structural matching works reliably
    # (1 + 2*i) * (3 + 4*i)
    expr = "(1 + 2*i) * (3 + 4*i)"
    rules = complex_registry.match_rules(expr)
    matched = [node for node, candidate in rules if node.id == "ALG-CPLX-MUL"]
    assert matched, "Should match complex multiplication rule"

def test_complex_conjugate(complex_registry):
    expr = "conjugate(3 + 4*i)"
    rules = complex_registry.match_rules(expr)
    matched = [node for node, candidate in rules if node.id == "ALG-CPLX-CONJ"]
    assert matched, "Should match conjugate rule"

def test_complex_modulus(complex_registry):
    expr = "Abs(3 + 4*i)"
    rules = complex_registry.match_rules(expr)
    matched = [node for node, candidate in rules if node.id == "ALG-CPLX-ABS"]
    assert matched, "Should match modulus rule"
