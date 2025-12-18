import pytest

from coherent.engine.fraction_engine import FractionEngine
from coherent.engine import ast_nodes as ast


@pytest.fixture
def engine() -> FractionEngine:
    return FractionEngine()


def test_normalize_reduces_simple_fraction(engine: FractionEngine):
    expr = ast.Div(left=ast.Int(2), right=ast.Int(4))
    normalized = engine.normalize(expr)
    assert normalized == ast.Div(left=ast.Int(1), right=ast.Int(2))


def test_normalize_handles_symbolic_factors(engine: FractionEngine):
    x = ast.Sym("x")
    expr = ast.Div(left=ast.Mul(factors=[x, x]), right=x)
    normalized = engine.normalize(expr)
    assert normalized == x


def test_add_two_fractions(engine: FractionEngine):
    left = ast.Div(left=ast.Int(1), right=ast.Int(2))
    right = ast.Div(left=ast.Int(1), right=ast.Int(3))
    result = engine.add(left, right)
    assert result == ast.Div(left=ast.Int(5), right=ast.Int(6))


def test_multiply_two_fractions(engine: FractionEngine):
    left = ast.Div(left=ast.Int(3), right=ast.Int(4))
    right = ast.Int(2)
    result = engine.multiply(left, right)
    assert result == ast.Div(left=ast.Int(3), right=ast.Int(2))
